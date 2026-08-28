#include "mujoco_utils.hpp"
#include "params.hpp"
#include "MST.hpp"
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
    bird_mmap::MMapLogger mmap_logger{};
    mmap_logger.open();
    State s{};
    Command cmd{};
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
    double flapping_phase = 0.0;
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
        applied_aero_pos.fill(Eigen::Vector3d::Zero());
        applied_aero_force.fill(Eigen::Vector3d::Zero());
        servo_torque.fill(0.0);
        damping_torque.fill(0.0);
        sim_step = 0;
        sim_step_credit = 0.0;
        flapping_phase = 0.0;
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

        constexpr double ELRS_MIN = 172.0;
        constexpr double ELRS_CENTER = 992.0;
        constexpr double ELRS_MAX = 1810.0;
        constexpr double TWO_PI = 2.0 * M_PI;
        constexpr double R1 = 0.4;
        constexpr double R2 = 0.3;
        constexpr double MAX_THETA0_OFFSET = M_PI / 6.0;
        constexpr double MAX_FLAPPING_AMPLITUDE = M_PI / 3.0;
        constexpr double MAX_FOLDING_ANGLE = M_PI / 2.0;
        constexpr double MAX_PITCHING_ANGLE = M_PI / 3.0;
        constexpr double MAX_FLAPPING_FREQUENCY = 5.0;
        constexpr double FREQUENCY_STOP_DEADBAND = 8.0;

        const double flapping_bias_channel = std::clamp(static_cast<double>(elrs_channels[1]), ELRS_MIN, ELRS_MAX);
        const double flapping_channel = std::clamp(static_cast<double>(elrs_channels[11]), ELRS_MIN, ELRS_MAX);
        const double folding_channel = std::clamp(static_cast<double>(elrs_channels[10]), ELRS_MIN, ELRS_MAX);
        const double frequency_channel = std::clamp(static_cast<double>(elrs_channels[2]), ELRS_MIN, ELRS_MAX);

        const double flapping_offset = flapping_bias_channel < ELRS_CENTER ? MAX_THETA0_OFFSET * (flapping_bias_channel - ELRS_CENTER) / (ELRS_CENTER - ELRS_MIN) : MAX_THETA0_OFFSET * (flapping_bias_channel - ELRS_CENTER) / (ELRS_MAX - ELRS_CENTER);
        const double flapping_amplitude = MAX_FLAPPING_AMPLITUDE * (flapping_channel - ELRS_MIN) / (ELRS_MAX - ELRS_MIN);
        const double folding_amplitude = MAX_FOLDING_ANGLE * (folding_channel - ELRS_MIN) / (ELRS_MAX - ELRS_MIN);
        const double pitching_amplitude = MAX_PITCHING_ANGLE * (folding_channel - ELRS_MIN) / (ELRS_MAX - ELRS_MIN);

        // The deadband prevents slow phase drift caused by idle-stick jitter.
        const double frequency_start = ELRS_MIN + FREQUENCY_STOP_DEADBAND;
        const double flapping_frequency = MAX_FLAPPING_FREQUENCY * std::clamp((frequency_channel - frequency_start) / (ELRS_MAX - frequency_start), 0.0, 1.0);
        const double cycle_ratio = flapping_phase / TWO_PI;

        // 1. Asymmetric flapping motion.
        double flapping_delta;
        if (cycle_ratio < R1) {flapping_delta = flapping_amplitude * std::cos(M_PI * cycle_ratio / R1);}
        else {flapping_delta = -flapping_amplitude * std::cos(M_PI * (cycle_ratio - R1) / (1.0 - R1));}

        // 2. non-folding, folding, and unfolding motion.
        double theta2 = 0.0;
        double pitching = 0.0;
        if (cycle_ratio >= R2) {
          theta2 = 0.5 * folding_amplitude * (1.0 - std::cos(TWO_PI * (cycle_ratio - R2) / (1.0 - R2)));
          pitching = 0.5 * pitching_amplitude * (1.0 - std::cos(TWO_PI * (cycle_ratio - R2) / (1.0 - R2)));
        }

        cmd.theta[0] = param::INITIAL_DES_THETA[0] + flapping_offset - flapping_delta;
        cmd.theta[1] = param::INITIAL_DES_THETA[1] + pitching;
        cmd.theta[2] = param::INITIAL_DES_THETA[2] - theta2;
        cmd.theta[3] = param::INITIAL_DES_THETA[3] + param::KIN_GAIN * theta2;
        cmd.theta[4] = J5_model(cmd.theta[2]);
        cmd.theta[5] = param::INITIAL_DES_THETA[5] - 1.4*param::KIN_GAIN * theta2;

        cmd.theta[6]  = param::INITIAL_DES_THETA[6] + flapping_offset - flapping_delta;
        cmd.theta[7]  = param::INITIAL_DES_THETA[7] + pitching;
        cmd.theta[8]  = param::INITIAL_DES_THETA[8] - theta2;
        cmd.theta[9]  = param::INITIAL_DES_THETA[9] + param::KIN_GAIN * theta2;
        cmd.theta[10] = J5_model(cmd.theta[8]);
        cmd.theta[11] = param::INITIAL_DES_THETA[11] - 1.4*param::KIN_GAIN * theta2;

        for (std::size_t i=param::NUM_WING_JOINTS; i<param::NUM_JOINTS; ++i) {cmd.theta[i] = param::INITIAL_DES_THETA[i];}

        if (advance_sim) {
          flapping_phase += TWO_PI * flapping_frequency * param::SIM_DT_SEC;
          if (flapping_phase >= TWO_PI) {flapping_phase -= TWO_PI;}
        }

        s.vel_f = Eigen::Vector3d((static_cast<double>(elrs_channels[14])-992.0)*(2.0/1638.0)*10.0, 0.0, 0.0);
        
        // std::printf("[ELRS] %llu", static_cast<unsigned long long>(elrs.valid_frames()));
        // for (std::size_t i=0; i<elrs_channels.size(); ++i) {std::printf(" %4u", static_cast<unsigned int>(elrs_channels[i]));}
        // std::putchar('\n'); std::fflush(stdout);
      }
      if (!elrs_enabled) {
        // for(std::size_t i=0; i<param::NUM_JOINTS; ++i) {cmd.theta[i] = static_cast<double>(viewer_data.theta_d[i]);}
        constexpr double FLAP_FREQ = 3.0;  // [Hz]
        constexpr double FLAP_AMP  = M_PI / 2.0;  // [rad]

        const double flap = FLAP_AMP * std::sin(2.0 * M_PI * FLAP_FREQ * static_cast<double>(sim_data->time));
        cmd.theta[0] = param::INITIAL_DES_THETA[0] + flap;
        cmd.theta[1] = param::INITIAL_DES_THETA[1];
        cmd.theta[2] = static_cast<double>(viewer_data.theta_d[2]);
        cmd.theta[3] = static_cast<double>(viewer_data.theta_d[3]);
        cmd.theta[4] = J5_model(cmd.theta[2]);
        cmd.theta[5] = static_cast<double>(viewer_data.theta_d[5]);
        cmd.theta[6] = param::INITIAL_DES_THETA[6] + flap;
        cmd.theta[7] = static_cast<double>(viewer_data.theta_d[7]);
        cmd.theta[8] = param::INITIAL_DES_THETA[8];
        cmd.theta[9] = static_cast<double>(viewer_data.theta_d[9]);
        cmd.theta[10] = J5_model(cmd.theta[8]);
        cmd.theta[11] = static_cast<double>(viewer_data.theta_d[11]);
        for (std::size_t i=param::NUM_WING_JOINTS; i<param::NUM_JOINTS; ++i) {cmd.theta[i] = static_cast<double>(viewer_data.theta_d[i]);}

        s.vel_f = Eigen::Vector3d(-10.0, 0.0, 0.0);
      }
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
            s.pos(0) = static_cast<double>( sensor_pos[0]);
            s.pos(1) = static_cast<double>(-sensor_pos[1]);
            s.pos(2) = static_cast<double>(-sensor_pos[2]);
            s.vel(0) = static_cast<double>( sensor_vel[0]);
            s.vel(1) = static_cast<double>(-sensor_vel[1]);
            s.vel(2) = static_cast<double>(-sensor_vel[2]);
            s.w(0) = static_cast<double>( sensor_gyro[0]);
            s.w(1) = static_cast<double>(-sensor_gyro[1]);
            s.w(2) = static_cast<double>(-sensor_gyro[2]);
            s.R = quat_to_R(sensor_quat);

            for (std::size_t i=0; i<param::NUM_JOINTS; ++i) {
              s.theta[i] = static_cast<double>(sim_data->qpos[servo_qpos_address[i]]);
              s.theta_dot[i] = static_cast<double>(sim_data->qvel[servo_qvel_address[i]]);
            }
          }

          // Servo is simulator-agnostic: feed it only the joint state.
          for (std::size_t i=0; i<param::NUM_JOINTS; ++i) {
            servo[i].desired_rad = cmd.theta[i];
            servo[i].step(s.theta[i], s.theta_dot[i]);
            sim_data->ctrl[mj_utils::g_actuator_ids[i]] = static_cast<mjtNum>(servo[i].motor_state.torque);
          }

          FK(s.theta, s.bTj);
          mst.update_dynamics(s, cmd.theta_t, log_due);

          const Eigen::Matrix3d GRb_FLU = param::NED_TO_FLU * s.R;
          const Eigen::Vector3d Gpb_FLU = param::NED_TO_FLU * s.pos;

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
            s.acc(0) = static_cast<double>( sensor_acc[0]);
            s.acc(1) = static_cast<double>(-sensor_acc[1]);
            s.acc(2) = static_cast<double>(-sensor_acc[2]);
            s.w_dot = s.R.transpose() * Eigen::Vector3d(static_cast<double>(sensor_angacc[0]), static_cast<double>(-sensor_angacc[1]), static_cast<double>(-sensor_angacc[2]));
            for (std::size_t i=0; i<param::NUM_JOINTS; ++i) {s.theta_ddot[i] = static_cast<double>(sim_data->qacc[servo_qvel_address[i]]);}

            // Keep qddot-dependent added-mass telemetry synchronized to every log sample.
            if (snapshot_due || log_due) {mst.update_visualization(s, cmd.theta_t);}
          }

          ++sim_step;
          if (log_due) {
            for (std::size_t i=0; i<param::NUM_JOINTS; ++i) {
              servo_torque[i] = servo[i].motor_state.torque;
              damping_torque[i] = -joint_passive_damping[i] * s.theta_dot[i];
            }
            mmap_logger.push(sample_time, sim_step, handled_reset_epoch, s, cmd, mst, servo_torque, damping_torque);
          }
        }
      }
      if (snapshot_due && !advance_sim) {
        std::copy_n(sim_data->qpos, snapshot_qpos.size(), snapshot_qpos.begin());
        std::copy_n(sim_data->qvel, snapshot_qvel.size(), snapshot_qvel.begin());
        snapshot_time = sim_data->time;
      
        s.pos(0) = static_cast<double>( sensor_pos[0]);
        s.pos(1) = static_cast<double>(-sensor_pos[1]);
        s.pos(2) = static_cast<double>(-sensor_pos[2]);
        s.vel(0) = static_cast<double>( sensor_vel[0]);
        s.vel(1) = static_cast<double>(-sensor_vel[1]);
        s.vel(2) = static_cast<double>(-sensor_vel[2]);
        s.acc(0) = static_cast<double>( sensor_acc[0]);
        s.acc(1) = static_cast<double>(-sensor_acc[1]);
        s.acc(2) = static_cast<double>(-sensor_acc[2]);
        s.w(0) = static_cast<double>( sensor_gyro[0]);
        s.w(1) = static_cast<double>(-sensor_gyro[1]);
        s.w(2) = static_cast<double>(-sensor_gyro[2]);
        s.R = quat_to_R(sensor_quat);
        s.w_dot = s.R.transpose() * Eigen::Vector3d(static_cast<double>(sensor_angacc[0]), static_cast<double>(-sensor_angacc[1]), static_cast<double>(-sensor_angacc[2]));

        for (std::size_t i=0; i<param::NUM_JOINTS; ++i) {
          s.theta[i] = static_cast<double>(sim_data->qpos[servo_qpos_address[i]]);
          s.theta_dot[i] = static_cast<double>(sim_data->qvel[servo_qvel_address[i]]);
          s.theta_ddot[i] = static_cast<double>(sim_data->qacc[servo_qvel_address[i]]);
        }

        FK(s.theta, s.bTj);
        mst.update_visualization(s, cmd.theta_t);
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
          shared_sim_data.state = s;
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
  State copied_state{};
  MST::StripState copied_strip_state{};
  std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS> copied_aero_pos{};
  std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS> copied_aero_force{};

  while (!glfwWindowShouldClose(window)) {
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

    // --- Process viewer input without simulation-state access ---
    glfwPollEvents();

    if (mj_utils::g_reset_epoch != handled_viewer_reset_epoch) {
      mj_utils::g_command_theta = param::INITIAL_DES_THETA;
      mj_utils::g_command_theta_t = 0.0;
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
      if (elrs_enabled){for(std::size_t i=0; i<param::NUM_JOINTS; ++i){mj_utils::g_command_theta[i] = static_cast<mjtNum>(shared_sim_data.theta_d[i]);}}
    }
    if (elrs_enabled) {mjui_update(-1, -1, &mj_utils::g_ui, &mj_utils::g_ui_state, &mj_utils::g_context);}
    mj_forward(m, render_data);

    const Eigen::Vector3d camera_target = param::NED_TO_FLU * copied_state.pos;
    for (int axis = 0; axis < 3; ++axis) {mj_utils::g_camera.lookat[axis] = static_cast<mjtNum>(camera_target(axis));}
    
    mjv_updateScene(m, render_data, &mj_utils::g_option, &mj_utils::g_perturb, &mj_utils::g_camera, mjCAT_ALL, &mj_utils::g_scene);
    mj_utils::highlight_selected_body();

    // Joint frames
    Eigen::Matrix4d GTb = Eigen::Matrix4d::Identity();
    GTb.block<3, 3>(0, 0) = param::NED_TO_FLU * copied_state.R;
    GTb.block<3, 1>(0, 3) = param::NED_TO_FLU * copied_state.pos;
    mj_utils::append_frame(GTb);
    for (const Eigen::Matrix4d& bTj : copied_state.bTj) {mj_utils::append_frame(GTb * bTj);}

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
