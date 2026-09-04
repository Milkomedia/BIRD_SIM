#include "mujoco_utils.hpp"
#include "params.hpp"
#include "MST.hpp"
#include "Mixer.hpp"
#include "utils.hpp"
#include "Servo.hpp"
#include "ELRS.hpp"
#include "mmap_manager.hpp"

#include <Eigen/Core>
#include <mujoco/mujoco.h>

#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

// ===== Main =====
int main(int argc, char** argv) {
  // ---------------- [ Initialize MuJoCo and viewer ] ----------------
  GLFWwindow* window = mj_utils::initialize(argc, argv);
  mjModel* m = mj_utils::g_model;
  mjData*  render_data = mj_utils::g_data;
  m->opt.timestep = static_cast<mjtNum>(std::chrono::duration<double>(param::SIM_DT_US).count());

  mjData* sim_data = mj_makeData(m);
  mj_copyData(sim_data, m, render_data);

  // ---------------- [ Sensor and actuator address ] ----------------
  const int imu_pos_sensor_adr  = m->sensor_adr[mj_name2id(m, mjOBJ_SENSOR, "imu_pos")];
  const int imu_vel_sensor_adr  = m->sensor_adr[mj_name2id(m, mjOBJ_SENSOR, "imu_vel")];
  const int imu_acc_sensor_adr  = m->sensor_adr[mj_name2id(m, mjOBJ_SENSOR, "imu_acc")];
  const int imu_quat_sensor_adr = m->sensor_adr[mj_name2id(m, mjOBJ_SENSOR, "imu_quat")];
  const int imu_gyro_sensor_adr = m->sensor_adr[mj_name2id(m, mjOBJ_SENSOR, "imu_gyro")];
  const int imu_angacc_sensor_adr = m->sensor_adr[mj_name2id(m, mjOBJ_SENSOR, "imu_angacc")];

  std::array<int, param::NUM_JOINTS> servo_qpos_address{};
  std::array<int, param::NUM_JOINTS> servo_qvel_address{};
  std::array<double, param::NUM_JOINTS> joint_passive_damping{};
  for (std::size_t i=0; i<param::NUM_JOINTS; ++i) {
    const int actuator_id = mj_utils::g_actuator_ids[i];
    const int joint_id = m->actuator_trnid[2 * actuator_id];
    const std::size_t motor_idx = param::MOTOR_MODEL_INDEX[i];
    servo_qpos_address[i] = m->jnt_qposadr[joint_id];
    servo_qvel_address[i] = m->jnt_dofadr[joint_id];
    joint_passive_damping[i] = static_cast<double>(m->dof_damping[servo_qvel_address[i]]);
    const double reduction_ratio = param::MOTOR_REDUCTION_RATIO[motor_idx];
    m->dof_armature[servo_qvel_address[i]] += static_cast<mjtNum>(param::MOTOR_ROTOR_INERTIA[motor_idx] * reduction_ratio * reduction_ratio);
  }
  // Propagate model-side armature changes, then recompute the initial dynamics.
  mj_setConst(m, sim_data);
  mj_forward(m, sim_data);

  constexpr std::array<const char*, MST::NUM_SURFACE_LOADS> surface_body_names = {"RWing3", "RWing4", "RWing6", "LWing3", "LWing4", "LWing6", "Tail2", "Tail2"};
  std::array<int, MST::NUM_SURFACE_LOADS> surface_body_ids{};
  for (std::size_t i=0; i<surface_body_ids.size(); ++i) {surface_body_ids[i] = mj_name2id(m, mjOBJ_BODY, surface_body_names[i]);}
  const int body_id = mj_name2id(m, mjOBJ_BODY, "body");

  const mjtNum* const subtree_com = sim_data->subtree_com + 3*body_id;
  const mjtNum* const sensor_pos  = sim_data->sensordata + imu_pos_sensor_adr;
  const mjtNum* const sensor_vel  = sim_data->sensordata + imu_vel_sensor_adr;
  const mjtNum* const sensor_acc  = sim_data->sensordata + imu_acc_sensor_adr;
  const mjtNum* const sensor_quat = sim_data->sensordata + imu_quat_sensor_adr;
  const mjtNum* const sensor_gyro = sim_data->sensordata + imu_gyro_sensor_adr;
  const mjtNum* const sensor_angacc = sim_data->sensordata + imu_angacc_sensor_adr;

  mj_utils::g_command_theta = param::INITIAL_DES_THETA;
  mjui_update(-1, -1, &mj_utils::g_ui, &mj_utils::g_ui_state, &mj_utils::g_context);

  // ---------------- [ Shared data ] ----------------
  ViewerData initial_viewer_data{};
  initial_viewer_data.theta_d = mj_utils::g_command_theta;
  initial_viewer_data.theta_t = mj_utils::g_command_theta_t;
  initial_viewer_data.sim_speed = mj_utils::g_sim_speed;
  initial_viewer_data.perturb = mj_utils::g_perturb;
  initial_viewer_data.paused = mj_utils::g_paused;
  initial_viewer_data.reset_epoch = mj_utils::g_reset_epoch;
  std::mutex viewer_mtx;
  ViewerData shared_viewer_data = initial_viewer_data;

  std::mutex sim_mtx;
  SimData shared_sim_data{std::vector<mjtNum>(static_cast<std::size_t>(m->nq)), std::vector<mjtNum>(static_cast<std::size_t>(m->nv))};
  std::copy_n(sim_data->qpos, shared_sim_data.qpos.size(), shared_sim_data.qpos.begin());
  std::copy_n(sim_data->qvel, shared_sim_data.qvel.size(), shared_sim_data.qvel.begin());
  shared_sim_data.time = sim_data->time;

  // ---------------- [ ELRS initialization ] ----------------
  ELRS elrs;
  ELRS::Channels elrs_channels{};
  bool elrs_enabled = false;

  if (!elrs.begin()) {std::fprintf(stderr, "[ELRS] UART open failed: device=%s, error=%s\n", ELRS::DEVICE, elrs.last_error().c_str());}
  else {
    const std::chrono::steady_clock::time_point probe_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    do {
      (void)elrs.update(elrs_channels);
      if (elrs.is_connected()) {break;}
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    while (std::chrono::steady_clock::now() < probe_deadline);

    std::printf("[ELRS] %s, CRSF=%s\n", ELRS::DEVICE, elrs.is_connected() ? "connected" : "waiting");
    if (elrs.is_connected()) {elrs_enabled = true;}
  }

  // ---------------- [ Simulation thread ] ----------------
  static std::atomic<bool> g_stop{false};
  
  std::thread th_sim([&]() {
    std::vector<Actuator::Servo> servo;
    servo.reserve(param::NUM_JOINTS);

    // --- Servo configuration ---
    std::array<Actuator::MotorParameters, param::NUM_MOTOR_MODELS> motor_parameters{};
    for (std::size_t i=0; i<motor_parameters.size(); ++i) {
      motor_parameters[i].ohm = param::MOTOR_OHM[i];
      motor_parameters[i].h = param::MOTOR_H[i];
      motor_parameters[i].Kt = param::MOTOR_KT[i];
      motor_parameters[i].Ke = param::MOTOR_KE[i];
      motor_parameters[i].reduction_ratio = param::MOTOR_REDUCTION_RATIO[i];
      motor_parameters[i].efficiency = param::MOTOR_EFFICIENCY[i];
      motor_parameters[i].max_torque = param::MOTOR_MAX_TORQUE[i];
      motor_parameters[i].viscous_friction = param::MOTOR_VISCOUS_FRICTION[i];
      motor_parameters[i].esc_time_constant = param::MOTOR_ESC_TIME_CONSTANT[i];
      motor_parameters[i].kP = param::MOTOR_KP[i];
      motor_parameters[i].kD = param::MOTOR_KD[i];
      motor_parameters[i].dt = param::SIM_DT_SEC;
    }
    for (std::size_t i=0; i<param::NUM_JOINTS; ++i) {servo.emplace_back(motor_parameters[param::MOTOR_MODEL_INDEX[i]]);}

    MST mst{};
    Mixer mixer{};
    bird_mmap::MMapLogger mmap_logger{};
    mmap_logger.open();
    SimState sim_state{};
    State s{};
    initialize_mass_estimate(s, sim_state, m, sim_data, body_id);
    Command cmd{};
    cmd.theta = param::INITIAL_DES_THETA;
    std::array<double, param::NUM_JOINTS> elrs_theta = param::INITIAL_DES_THETA;
    double elrs_flapping_phase = 0.0;
    ViewerData viewer_data = initial_viewer_data;
    std::vector<mjtNum> snapshot_qpos(static_cast<std::size_t>(m->nq));
    std::vector<mjtNum> snapshot_qvel(static_cast<std::size_t>(m->nv));
    std::vector<mjtNum> added_mass_jac(6*static_cast<std::size_t>(m->nv));
    std::vector<mjtNum> added_mass_inertia_jac(6*static_cast<std::size_t>(m->nv));
    std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS> applied_aero_pos{};
    std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS> applied_aero_force{};
    std::array<double, param::NUM_JOINTS> servo_torque{};
    std::array<double, param::NUM_JOINTS> damping_torque{};
    mjtNum snapshot_time = sim_data->time;
    std::uint64_t handled_reset_epoch = viewer_data.reset_epoch;
    std::uint64_t sim_step = 0;
    double sim_step_credit = 0.0;
    bool manual_mode = false;
    std::chrono::steady_clock::duration snapshot_elapsed = std::chrono::steady_clock::duration::zero();

    std::chrono::steady_clock::time_point next_tick = std::chrono::steady_clock::now();

    // =====================================================
    // ================ [ Simulation loop ] ================
    // =====================================================
    while (!g_stop.load(std::memory_order_acquire)) {
      next_tick += param::SIM_DT_US;

      { // Consume the newest viewer command without blocking the sim thread.
        std::unique_lock<std::mutex> viewer_lk(viewer_mtx, std::try_to_lock);
        if (viewer_lk.owns_lock()) {viewer_data = shared_viewer_data;}
      }

      // --- Check reset request ---
      const bool reset_requested = viewer_data.reset_epoch != handled_reset_epoch;
      if (reset_requested) {
        mj_resetData(m, sim_data);
        mj_forward(m, sim_data);
        for (Actuator::Servo& item : servo) {item.reset();}
        mst.reset();
        mj_utils::g_manus_trajectory.reset();
        applied_aero_pos.fill(Eigen::Vector3d::Zero());
        applied_aero_force.fill(Eigen::Vector3d::Zero());
        servo_torque.fill(0.0);
        damping_torque.fill(0.0);
        sim_step = 0;
        sim_step_credit = 0.0;
        elrs_flapping_phase = 0.0;
        handled_reset_epoch = viewer_data.reset_epoch;
      }

      // Keep MuJoCo's fixed timestep and scale only how often a step runs in wall time.
      bool advance_sim = false;
      if (!reset_requested && !viewer_data.paused) {
        sim_step_credit += std::clamp(static_cast<double>(viewer_data.sim_speed), 0.0, 1.0);
        if (sim_step_credit >= 1.0-1.0e-12) {
          sim_step_credit = std::max(0.0, sim_step_credit-1.0);
          advance_sim = true;
        }
      }
      else if (viewer_data.paused) {sim_step_credit = 0.0;}

      snapshot_elapsed += param::SIM_DT_US;
      const bool snapshot_due = reset_requested || snapshot_elapsed >= param::RENDER_DT_US;
      if (snapshot_due) {snapshot_elapsed = reset_requested ? std::chrono::steady_clock::duration::zero() : snapshot_elapsed - param::RENDER_DT_US;}
      const bool log_due = advance_sim && (sim_step + 1) % bird_mmap::LOG_DECIMATION == 0;

      // --- ELRS RC command ---
      if (elrs_enabled) {
        (void)elrs.update(elrs_channels);

        manual_mode = (elrs_channels[5] == 1810);
        if (manual_mode) {
          // ELRS MIN = 172, MAX = 1810.
          cmd.u(0) = param::MAX_FREQ * static_cast<double>(elrs_channels[10]- 172) / 1638.0; // flapping_frequency
          cmd.u(1) = param::MAX_FLAPPING_AMPLITUDE * static_cast<double>(elrs_channels[2]- 172) / 1638.0; // mean_flapping_amplitude
          cmd.u(2) = param::MAX_FLAPPING_DIFFERENCE * (static_cast<double>(elrs_channels[0]) - 992.0) / (elrs_channels[0] < 992 ? 820.0 : 818.0); // flapping_difference
          cmd.u(3) = param::MAX_PITCHING_AMPLITUDE * static_cast<double>(elrs_channels[1]- 172) / 1638.0; // pitching_amplitude
          cmd.u(4) = param::MAX_PITCHING_DIFFERENCE * (static_cast<double>(elrs_channels[3]) - 992.0) / (elrs_channels[3] < 992 ? 820.0 : 818.0); // pitching_difference
          cmd.u(5) = param::MAX_SWEEP_BIAS * (static_cast<double>(elrs_channels[11]) - 992.0) / (elrs_channels[11] < 992 ? 820.0 : 818.0); // sweep_bias

          const double cycle_ratio = elrs_flapping_phase / (2.0 * M_PI);
          double cosR1; double sinR1;
          if (cycle_ratio < param::R1) {cosR1 = -std::cos(M_PI * cycle_ratio / param::R1); sinR1 = std::sin(M_PI * cycle_ratio / param::R1);}
          else{cosR1 = std::cos(M_PI * (cycle_ratio - param::R1) / (1.0 - param::R1)); sinR1 = std::sin(M_PI * (param::R1 - cycle_ratio) / (1.0 - param::R1));}
          double one_minus_cosR2 = 0.0;
          if (cycle_ratio > param::R2) {one_minus_cosR2 = 1.0 - std::cos(2.0 * M_PI * (cycle_ratio - param::R2) / (1.0 - param::R2));}

          const double flapping_right = param::FLAPPING_DELTA_0 + (cmd.u(1) + 0.5*cmd.u(2)) * cosR1;
          const double flapping_left  = param::FLAPPING_DELTA_0 + (cmd.u(1) - 0.5*cmd.u(2)) * cosR1;
          const double pitching_right = param::PITCHING_DELTA_0 + 0.5 * (cmd.u(3) + 0.5*cmd.u(4)) * one_minus_cosR2;
          const double pitching_left  = param::PITCHING_DELTA_0 + 0.5 * (cmd.u(3) - 0.5*cmd.u(4)) * one_minus_cosR2;
          const double sweep   = cmd.u(5) + param::SWEEP_AMPLITUDE * sinR1;
          const double folding = param::FOLDING_DELTA_0 + 0.5 * param::FOLDING_AMPLITUDE * one_minus_cosR2;

          elrs_theta[0] = param::INITIAL_DES_THETA[0] + flapping_right;
          elrs_theta[1] = param::INITIAL_DES_THETA[1] + pitching_right;
          elrs_theta[2] = param::INITIAL_DES_THETA[2] + sweep;
          elrs_theta[3] = param::INITIAL_DES_THETA[3] + folding;
          elrs_theta[4] = J5_model(elrs_theta[2]);
          elrs_theta[5] = param::INITIAL_DES_THETA[5] - 2.0 * folding;

          elrs_theta[6]  = param::INITIAL_DES_THETA[6] + flapping_left;
          elrs_theta[7]  = param::INITIAL_DES_THETA[7] + pitching_left;
          elrs_theta[8]  = param::INITIAL_DES_THETA[8] + sweep;
          elrs_theta[9]  = param::INITIAL_DES_THETA[9] + folding;
          elrs_theta[10] = J5_model(elrs_theta[8]);
          elrs_theta[11] = param::INITIAL_DES_THETA[11] - 2.0 * folding;

          for (std::size_t i=param::NUM_WING_JOINTS; i<param::NUM_JOINTS; ++i) {elrs_theta[i] = static_cast<double>(viewer_data.theta_d[i]);}
        }
      }
      if (!elrs_enabled) {for(std::size_t i=0; i<param::NUM_JOINTS; ++i) {cmd.theta[i] = static_cast<double>(viewer_data.theta_d[i]);}}

      sim_state.vel_f = Eigen::Vector3d(-8.0, 0.0, 0.0);
      cmd.theta_t = static_cast<double>(viewer_data.theta_t);

      // --- Control and servo dynamics ---
      if (!reset_requested) {
        mju_zero(sim_data->xfrc_applied, 6*m->nbody);
        mju_zero(sim_data->qfrc_applied, m->nv);
        mjv_applyPerturbPose(m, sim_data, &viewer_data.perturb, viewer_data.paused);

        if (advance_sim) {
          // --- MuJoCo simulation step1 ---
          mj_step1(m, sim_data);

          { // --- Position/velocity state at t_k ---
            sim_state.pos(0) = static_cast<double>( sensor_pos[0]);
            sim_state.pos(1) = static_cast<double>(-sensor_pos[1]);
            sim_state.pos(2) = static_cast<double>(-sensor_pos[2]);
            sim_state.vel(0) = static_cast<double>( sensor_vel[0]);
            sim_state.vel(1) = static_cast<double>(-sensor_vel[1]);
            sim_state.vel(2) = static_cast<double>(-sensor_vel[2]);
            sim_state.w(0) = static_cast<double>( sensor_gyro[0]);
            sim_state.w(1) = static_cast<double>(-sensor_gyro[1]);
            sim_state.w(2) = static_cast<double>(-sensor_gyro[2]);
            sim_state.R = quat_to_R(sensor_quat);

            const Eigen::Vector3d Gpc_NED(static_cast<double>( subtree_com[0]), static_cast<double>(-subtree_com[1]), static_cast<double>(-subtree_com[2]));
            sim_state.bpc = sim_state.R.transpose() * (Gpc_NED - sim_state.pos);

            for (std::size_t i=0; i<param::NUM_JOINTS; ++i) {
              sim_state.theta[i] = static_cast<double>(sim_data->qpos[servo_qpos_address[i]]);
              sim_state.theta_dot[i] = static_cast<double>(sim_data->qvel[servo_qvel_address[i]]);
            }
          }

          FK(sim_state.theta, sim_state.bTj);
          update_state_estimate(s, sim_state);

          // --- Flight-control loop: once every five simulation steps ---
          Eigen::Matrix<double, 6, 6> B;
          Eigen::Matrix<double, 6, 1> w_hat;
          if (sim_step % 5 == 0) {
            // B rows: [Fx, Fy, Fz, Mx, My, Mz]
            // B cols: [f, Af_bar, Af_delta, Ap_bar, Ap_delta, sweep]
            mixer.update_B(s, cmd.u, B, w_hat);

            // std::printf(
            //   "[Mixer] mean F=[%.6g %.6g %.6g],  M=[%.6g %.6g %.6g],  ||B||_F=%.6g\n",
            //   w_hat(0), w_hat(1), w_hat(2),
            //   w_hat(3), w_hat(4), w_hat(5),
            //   B.norm()
            // );
            // std::printf("\n|%.6g %.6g %.6g %.6g %.6g %.6g|\n", B(0,0), B(0,1), B(0,2), B(0,3), B(0,4), B(0,5));
            // std::printf("|%.6g %.6g %.6g %.6g %.6g %.6g|\n", B(1,0), B(1,1), B(1,2), B(1,3), B(1,4), B(1,5));
            // std::printf("|%.6g %.6g %.6g %.6g %.6g %.6g|\n", B(2,0), B(2,1), B(2,2), B(2,3), B(2,4), B(2,5));
            // std::printf("|%.6g %.6g %.6g %.6g %.6g %.6g|\n", B(3,0), B(3,1), B(3,2), B(3,3), B(3,4), B(3,5));
            // std::printf("|%.6g %.6g %.6g %.6g %.6g %.6g|\n", B(4,0), B(4,1), B(4,2), B(4,3), B(4,4), B(4,5));
            // std::printf("|%.6g %.6g %.6g %.6g %.6g %.6g|\n", B(5,0), B(5,1), B(5,2), B(5,3), B(5,4), B(5,5));

            // Future incremental-QP allocation and joint-trajectory logic goes here.
          }

          // Manual mode has final authority over any automatic joint command.
          if (manual_mode) {cmd.theta = elrs_theta;}

          elrs_flapping_phase += 2.0 * M_PI * cmd.u(0) * param::SIM_DT_SEC;
          if (elrs_flapping_phase >= 2.0 * M_PI) {elrs_flapping_phase -= 2.0 * M_PI;}

          // Servo is simulator-agnostic: feed it only the joint state.
          for (std::size_t i=0; i<param::NUM_JOINTS; ++i) {
            servo[i].desired_rad = cmd.theta[i];
            servo[i].step(s.theta[i], s.theta_dot[i]);
            sim_data->ctrl[mj_utils::g_actuator_ids[i]] = static_cast<mjtNum>(servo[i].motor_state.torque);
          }

          mst.update_dynamics(sim_state, cmd.theta_t, log_due);

          const Eigen::Matrix3d GRb_FLU = param::NED_TO_FLU * sim_state.R;
          const Eigen::Vector3d Gpb_FLU = param::NED_TO_FLU * sim_state.pos;
          mj_utils::g_manus_trajectory.sample_if_due(static_cast<double>(sim_data->time), mst.copy_strip_state().p_m, GRb_FLU, Gpb_FLU);

          // Transfer MST's current added inertia to MuJoCo's generalized mass.
          add_spatial_inertias(m, sim_data, mst.added_mass_positions(), mst.added_mass_matrices(), surface_body_ids, GRb_FLU, Gpb_FLU, added_mass_jac.data(), added_mass_inertia_jac.data());

          // Apply current t_k aerodynamic wrenches.
          mjv_applyPerturbForce(m, sim_data, &viewer_data.perturb);
          const std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS>& bp = mst.positions();
          const std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS>& bF = mst.forces();
          const std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS>& bT = mst.torques();

          for (std::size_t i=0; i<MST::NUM_AERO_LOADS; ++i) {
            const Eigen::Vector3d GF = GRb_FLU * bF[i];
            const Eigen::Vector3d GT = GRb_FLU * bT[i];
            const Eigen::Vector3d Gp = Gpb_FLU + GRb_FLU * bp[i];
            const mjtNum force[3]  = {static_cast<mjtNum>(GF(0)), static_cast<mjtNum>(GF(1)), static_cast<mjtNum>(GF(2))};
            const mjtNum torque[3] = {static_cast<mjtNum>(GT(0)), static_cast<mjtNum>(GT(1)), static_cast<mjtNum>(GT(2))};
            const mjtNum point[3]  = {static_cast<mjtNum>(Gp(0)), static_cast<mjtNum>(Gp(1)), static_cast<mjtNum>(Gp(2))};
            const int target_body_id = i < surface_body_ids.size() ? surface_body_ids[i] : body_id;
            mj_applyFT(m, sim_data, force, torque, point, target_body_id, sim_data->qfrc_applied);
          }

          const double sample_time = static_cast<double>(sim_data->time);
          if (snapshot_due) {
            std::copy_n(sim_data->qpos, snapshot_qpos.size(), snapshot_qpos.begin());
            std::copy_n(sim_data->qvel, snapshot_qvel.size(), snapshot_qvel.begin());
            snapshot_time = sim_data->time;
            applied_aero_pos = mst.positions();
            applied_aero_force = mst.forces();
          }

          // --- MuJoCo simulation step2 ---
          mj_step2(m, sim_data);

          { // --- Acceleration state solved at t_k ---
            sim_state.acc(0) = static_cast<double>( sensor_acc[0]);
            sim_state.acc(1) = static_cast<double>(-sensor_acc[1]);
            sim_state.acc(2) = static_cast<double>(-sensor_acc[2]);
            sim_state.w_dot = sim_state.R.transpose() * Eigen::Vector3d(static_cast<double>(sensor_angacc[0]), static_cast<double>(-sensor_angacc[1]), static_cast<double>(-sensor_angacc[2]));
            for (std::size_t i=0; i<param::NUM_JOINTS; ++i) {sim_state.theta_ddot[i] = static_cast<double>(sim_data->qacc[servo_qvel_address[i]]);}

            // Keep qddot-dependent added-mass telemetry synchronized to every log sample.
            if (snapshot_due || log_due) {mst.update_visualization(sim_state, cmd.theta_t);}
          }

          ++sim_step;
          if (log_due) {
            for (std::size_t i=0; i<param::NUM_JOINTS; ++i) {
              servo_torque[i] = servo[i].motor_state.torque;
              damping_torque[i] = -joint_passive_damping[i] * sim_state.theta_dot[i];
            }
            mmap_logger.push(sample_time, sim_step, handled_reset_epoch, sim_state, cmd, mst, servo_torque, damping_torque);
          }
        }
      }
      if (snapshot_due && !advance_sim) {
        std::copy_n(sim_data->qpos, snapshot_qpos.size(), snapshot_qpos.begin());
        std::copy_n(sim_data->qvel, snapshot_qvel.size(), snapshot_qvel.begin());
        snapshot_time = sim_data->time;
      
        sim_state.pos(0) = static_cast<double>( sensor_pos[0]);
        sim_state.pos(1) = static_cast<double>(-sensor_pos[1]);
        sim_state.pos(2) = static_cast<double>(-sensor_pos[2]);
        sim_state.vel(0) = static_cast<double>( sensor_vel[0]);
        sim_state.vel(1) = static_cast<double>(-sensor_vel[1]);
        sim_state.vel(2) = static_cast<double>(-sensor_vel[2]);
        sim_state.acc(0) = static_cast<double>( sensor_acc[0]);
        sim_state.acc(1) = static_cast<double>(-sensor_acc[1]);
        sim_state.acc(2) = static_cast<double>(-sensor_acc[2]);
        sim_state.w(0) = static_cast<double>( sensor_gyro[0]);
        sim_state.w(1) = static_cast<double>(-sensor_gyro[1]);
        sim_state.w(2) = static_cast<double>(-sensor_gyro[2]);
        sim_state.R = quat_to_R(sensor_quat);
        sim_state.w_dot = sim_state.R.transpose() * Eigen::Vector3d(static_cast<double>(sensor_angacc[0]), static_cast<double>(-sensor_angacc[1]), static_cast<double>(-sensor_angacc[2]));

        const Eigen::Vector3d Gpc_NED(static_cast<double>( subtree_com[0]), static_cast<double>(-subtree_com[1]), static_cast<double>(-subtree_com[2]));
        sim_state.bpc = sim_state.R.transpose() * (Gpc_NED - sim_state.pos);

        for (std::size_t i=0; i<param::NUM_JOINTS; ++i) {
          sim_state.theta[i] = static_cast<double>(sim_data->qpos[servo_qpos_address[i]]);
          sim_state.theta_dot[i] = static_cast<double>(sim_data->qvel[servo_qvel_address[i]]);
          sim_state.theta_ddot[i] = static_cast<double>(sim_data->qacc[servo_qvel_address[i]]);
        }

        FK(sim_state.theta, sim_state.bTj);
        update_state_estimate(s, sim_state);
        mst.update_visualization(sim_state, cmd.theta_t);
        applied_aero_pos = mst.positions();
        applied_aero_force = mst.forces();
      }

      // --- Publish a render snapshot at the viewer rate ---
      if (snapshot_due) {
        std::unique_lock<std::mutex> snapshot_lk(sim_mtx, std::try_to_lock);
        if (snapshot_lk.owns_lock()) {
          shared_sim_data.qpos.swap(snapshot_qpos);
          shared_sim_data.qvel.swap(snapshot_qvel);
          shared_sim_data.time = snapshot_time;
          shared_sim_data.state = sim_state;
          shared_sim_data.strip_state = mst.copy_strip_state();
          shared_sim_data.aero_pos = applied_aero_pos;
          shared_sim_data.aero_force = applied_aero_force;
          shared_sim_data.theta_d = cmd.theta;
        }
      }

      // --- SIM_DT_US absolute-time delay ---
      const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
      const std::chrono::steady_clock::time_point sleep_target = next_tick - param::SPIN_MARGIN_US;
      if (now < next_tick) {
        if (now < sleep_target) {std::this_thread::sleep_until(sleep_target);}
        while (!g_stop.load(std::memory_order_relaxed) && std::chrono::steady_clock::now() < next_tick) {}
      }
    }
  });

  // ---------------- [ Viewer thread (main thread) ] ----------------
  const std::chrono::steady_clock::duration render_period = param::RENDER_DT_US;
  std::uint64_t handled_viewer_reset_epoch = mj_utils::g_reset_epoch;
  SimState copied_state{};
  MST::StripState copied_strip_state{};
  std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS> copied_aero_pos{};
  std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS> copied_aero_force{};

  while (!glfwWindowShouldClose(window)) {
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

    // --- Process viewer input without simulation-state access ---
    glfwPollEvents();

    if (mj_utils::g_reset_epoch != handled_viewer_reset_epoch) {
      mj_utils::g_command_theta = param::INITIAL_DES_THETA;
      mj_utils::g_command_theta_t = param::INITIAL_THETA_T;
      mjui_update(-1, -1, &mj_utils::g_ui, &mj_utils::g_ui_state, &mj_utils::g_context);
      handled_viewer_reset_epoch = mj_utils::g_reset_epoch;
    }

    ViewerData viewer_data{};
    for (std::size_t i=0; i<param::NUM_JOINTS; ++i) {viewer_data.theta_d[i] = mj_utils::g_command_theta[i];}
    viewer_data.theta_t = mj_utils::g_command_theta_t;
    viewer_data.sim_speed = mj_utils::g_sim_speed;
    viewer_data.perturb = mj_utils::g_perturb;
    viewer_data.paused = mj_utils::g_paused;
    viewer_data.reset_epoch = mj_utils::g_reset_epoch;
    
    {
      std::lock_guard<std::mutex> viewer_lk(viewer_mtx);
      shared_viewer_data = viewer_data;
    }

    { // Copy data
      std::lock_guard<std::mutex> snapshot_lk(sim_mtx);
      std::copy_n(shared_sim_data.qpos.begin(), shared_sim_data.qpos.size(), render_data->qpos);
      std::copy_n(shared_sim_data.qvel.begin(), shared_sim_data.qvel.size(), render_data->qvel);
      render_data->time = shared_sim_data.time;
      copied_state = shared_sim_data.state;
      copied_strip_state = shared_sim_data.strip_state;
      copied_aero_pos = shared_sim_data.aero_pos;
      copied_aero_force = shared_sim_data.aero_force;
      if (elrs_enabled) {for(std::size_t i=0; i<param::NUM_WING_JOINTS; ++i) {mj_utils::g_command_theta[i] = static_cast<mjtNum>(shared_sim_data.theta_d[i]);}}
    }
    if (elrs_enabled) {mjui_update(-1, -1, &mj_utils::g_ui, &mj_utils::g_ui_state, &mj_utils::g_context);}
    mj_forward(m, render_data);

    const Eigen::Vector3d camera_target = param::NED_TO_FLU * copied_state.pos;
    for (int axis = 0; axis < 3; ++axis) {mj_utils::g_camera.lookat[axis] = static_cast<mjtNum>(camera_target(axis)) + mj_utils::g_camera_pan_offset[axis];}
    
    mjv_updateScene(m, render_data, &mj_utils::g_option, &mj_utils::g_perturb, &mj_utils::g_camera, mjCAT_ALL, &mj_utils::g_scene);
    mj_utils::highlight_selected_body();

    // Joint frames
    Eigen::Matrix4d GTb = Eigen::Matrix4d::Identity();
    GTb.block<3, 3>(0, 0) = param::NED_TO_FLU * copied_state.R;
    GTb.block<3, 1>(0, 3) = param::NED_TO_FLU * copied_state.pos;
    mj_utils::append_frame(GTb);
    const Eigen::Vector3d Gpc = GTb.block<3, 1>(0, 3) + GTb.block<3, 3>(0, 0) * copied_state.bpc;
    mj_utils::append_com_marker(Gpc);
    for (const Eigen::Matrix4d& bTj : copied_state.bTj) {mj_utils::append_frame(GTb * bTj);}
    mj_utils::g_manus_trajectory.render(static_cast<double>(render_data->time), copied_strip_state.p_m, GTb);

    { // State arrows and strip frames
      const Eigen::Matrix3d GRb = GTb.block<3, 3>(0, 0);
      const Eigen::Vector3d Gpb = GTb.block<3, 1>(0, 3);
      for (std::size_t wing=0; wing<2; ++wing) {
        const MST::StripRotation<1>& humerus_rotation = copied_strip_state.humerus_rotation[wing];
        const MST::StripRotation<param::NR>& radius_rotation = copied_strip_state.radius_rotation[wing];
        const MST::StripRotation<1>& manus_rotation = copied_strip_state.manus_rotation[wing];
        const Eigen::Matrix3d& bRhi = humerus_rotation.bRri[0];
        const Eigen::Matrix3d& bRmi = manus_rotation.bRri[0];

        if (mj_utils::g_arrow_view == mj_utils::ARROW_VIEW_WING || mj_utils::g_arrow_view == mj_utils::ARROW_VIEW_BOTH) {
          switch (mj_utils::g_arrow_quantity) {
            case mj_utils::ARROW_A:
              mj_utils::append_segment_vector<param::NH>(copied_strip_state.p_h, copied_strip_state.a_h, wing*param::NH, GRb, Gpb, bRhi, mj_utils::A_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::A_ARROW_COLOR);
              mj_utils::append_segment_vector<param::NR>(copied_strip_state.p_r, copied_strip_state.a_r, wing*param::NR, GRb, Gpb, radius_rotation.bRri, mj_utils::A_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::A_ARROW_COLOR);
              mj_utils::append_segment_vector<param::NM>(copied_strip_state.p_m, copied_strip_state.a_m, wing*param::NM, GRb, Gpb, bRmi, mj_utils::A_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::A_ARROW_COLOR);
              break;

            case mj_utils::ARROW_W:
              mj_utils::append_segment_vector<param::NH>(copied_strip_state.p_h, copied_strip_state.w_h[wing], wing*param::NH, GRb, Gpb, bRhi, mj_utils::W_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::W_ARROW_COLOR);
              mj_utils::append_segment_vector<param::NR>(copied_strip_state.p_r, copied_strip_state.w_r, wing*param::NR, GRb, Gpb, radius_rotation.bRri, mj_utils::W_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::W_ARROW_COLOR);
              mj_utils::append_segment_vector<param::NM>(copied_strip_state.p_m, copied_strip_state.w_m[wing], wing*param::NM, GRb, Gpb, bRmi, mj_utils::W_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::W_ARROW_COLOR);
              break;

            case mj_utils::ARROW_WDOT:
              mj_utils::append_segment_vector<param::NH>(copied_strip_state.p_h, copied_strip_state.wdot_h[wing], wing*param::NH, GRb, Gpb, bRhi, mj_utils::WDOT_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::WDOT_ARROW_COLOR);
              mj_utils::append_segment_vector<param::NR>(copied_strip_state.p_r, copied_strip_state.wdot_r, wing*param::NR, GRb, Gpb, radius_rotation.bRri, mj_utils::WDOT_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::WDOT_ARROW_COLOR);
              mj_utils::append_segment_vector<param::NM>(copied_strip_state.p_m, copied_strip_state.wdot_m[wing], wing*param::NM, GRb, Gpb, bRmi, mj_utils::WDOT_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::WDOT_ARROW_COLOR);
              break;

            case mj_utils::ARROW_V:
            default:
              mj_utils::append_segment_vector<param::NH>(copied_strip_state.p_h, copied_strip_state.v_h, wing*param::NH, GRb, Gpb, bRhi, mj_utils::V_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::V_ARROW_COLOR);
              mj_utils::append_segment_vector<param::NR>(copied_strip_state.p_r, copied_strip_state.v_r, wing*param::NR, GRb, Gpb, radius_rotation.bRri, mj_utils::V_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::V_ARROW_COLOR);
              mj_utils::append_segment_vector<param::NM>(copied_strip_state.p_m, copied_strip_state.v_m, wing*param::NM, GRb, Gpb, bRmi, mj_utils::V_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::V_ARROW_COLOR);
              break;
          }
        }

        mj_utils::append_humerus_strip_frames(copied_strip_state.p_h, wing*param::NH, GTb, bRhi, humerus_rotation.cos_psi[0]);
        mj_utils::append_radius_strip_frames(copied_strip_state.p_r, wing*param::NR, GTb, radius_rotation.bRri, radius_rotation.cos_psi);
        mj_utils::append_manus_strip_frames(copied_strip_state.p_m, wing*param::NM, GTb, bRmi, manus_rotation.cos_psi[0]);
      }

      for (std::size_t section=0; section<2; ++section) {
        const std::size_t idx0 = section*param::NT;
        const Eigen::Matrix3d& bRt = copied_strip_state.bR_t[section];

        if (mj_utils::g_arrow_view == mj_utils::ARROW_VIEW_TAIL || mj_utils::g_arrow_view == mj_utils::ARROW_VIEW_BOTH) {
          switch (mj_utils::g_arrow_quantity) {
            case mj_utils::ARROW_A:
              mj_utils::append_segment_vector<param::NT>(copied_strip_state.p_t, copied_strip_state.a_t, idx0, GRb, Gpb, bRt, mj_utils::A_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::A_ARROW_COLOR);
              break;

            case mj_utils::ARROW_W:
              mj_utils::append_segment_vector<param::NT>(copied_strip_state.p_t, copied_strip_state.w_t[section], idx0, GRb, Gpb, bRt, mj_utils::W_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::W_ARROW_COLOR);
              break;

            case mj_utils::ARROW_WDOT:
              mj_utils::append_segment_vector<param::NT>(copied_strip_state.p_t, copied_strip_state.wdot_t[section], idx0, GRb, Gpb, bRt, mj_utils::WDOT_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::WDOT_ARROW_COLOR);
              break;

            case mj_utils::ARROW_V:
            default:
              mj_utils::append_segment_vector<param::NT>(copied_strip_state.p_t, copied_strip_state.v_t, idx0, GRb, Gpb, bRt, mj_utils::V_ARROW_SCALE, mj_utils::STATE_ARROW_WIDTH, mj_utils::V_ARROW_COLOR);
              break;
          }
        }
      }
      mj_utils::append_tail_strip_frames(copied_strip_state.p_t, copied_strip_state.bR_t, copied_strip_state.theta_t, GTb);

      for (std::size_t i=0; i<MST::NUM_AERO_LOADS; ++i) {
        const Eigen::Vector3d origin = Gpb + GRb*copied_aero_pos[i];
        const Eigen::Vector3d force = GRb*copied_aero_force[i];
        mj_utils::append_arrow(origin, force, mj_utils::AERO_FORCE_ARROW_SCALE, mj_utils::AERO_FORCE_ARROW_WIDTH, mj_utils::AERO_FORCE_ARROW_COLOR);
      }
    }

    // --- Render ---
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    mjr_render(viewport, &mj_utils::g_scene, &mj_utils::g_context);

    char status[128] = {};
    std::snprintf(status, sizeof(status), "%.2fs", static_cast<double>(render_data->time));
    mjr_overlay(mjFONT_SHADOW, mjGRID_TOPLEFT, viewport, status, "", &mj_utils::g_context);
    mj_utils::render_ui(window);
    glfwSwapBuffers(window);

    std::this_thread::sleep_until(now + render_period);
  }

  g_stop.store(true, std::memory_order_release);
  if (th_sim.joinable()) th_sim.join();

  mj_deleteData(sim_data);
  mj_utils::shutdown(window);
  return 0;
}
