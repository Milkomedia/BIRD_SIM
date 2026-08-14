#include "mujoco_utils.hpp"
#include "params.hpp"
#include "MST.hpp"
#include "utils.hpp"
#include "Servo.hpp"
#include "ELRS.hpp"

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

static std::atomic<bool> g_stop{false};

// ===== Main =====
int main(int argc, char** argv) {
  // ---------------- [ Initialize MuJoCo and viewer ] ----------------
  GLFWwindow* window = mj_utils::initialize(argc, argv);
  mjModel* m = mj_utils::g_model;
  mjData*  render_data = mj_utils::g_data;
  m->opt.timestep = static_cast<mjtNum>(std::chrono::duration<double>(param::SIM_DT_US).count());

  mjData* sim_data = mj_makeData(m);
  mj_copyData(sim_data, m, render_data);
  mj_forward(m, sim_data);

  // ---------------- [ Sensor and actuator address ] ----------------
  const int imu_pos_sensor_adr  = m->sensor_adr[mj_name2id(m, mjOBJ_SENSOR, "imu_pos")];
  const int imu_vel_sensor_adr  = m->sensor_adr[mj_name2id(m, mjOBJ_SENSOR, "imu_vel")];
  const int imu_acc_sensor_adr  = m->sensor_adr[mj_name2id(m, mjOBJ_SENSOR, "imu_acc")];
  const int imu_quat_sensor_adr = m->sensor_adr[mj_name2id(m, mjOBJ_SENSOR, "imu_quat")];
  const int imu_gyro_sensor_adr = m->sensor_adr[mj_name2id(m, mjOBJ_SENSOR, "imu_gyro")];
  const int imu_angacc_sensor_adr = m->sensor_adr[mj_name2id(m, mjOBJ_SENSOR, "imu_angacc")];

  std::array<int, 12> servo_qpos_address{};
  std::array<int, 12> servo_qvel_address{};
  for (std::size_t i=0; i<12; ++i) {
    const int joint_id = m->actuator_trnid[2 * mj_utils::g_actuator_ids[i]];
    servo_qpos_address[i] = m->jnt_qposadr[joint_id];
    servo_qvel_address[i] = m->jnt_dofadr[joint_id];
  }

  constexpr std::array<const char*, 6> wing_plate_names = {"RWing3", "RWing4", "RWing6", "LWing3", "LWing4", "LWing6"};
  std::array<int, 6> wing_plate_ids{};
  for (std::size_t i=0; i<wing_plate_ids.size(); ++i) {wing_plate_ids[i] = mj_name2id(m, mjOBJ_BODY, wing_plate_names[i]);}
  const int body_id = mj_name2id(m, mjOBJ_BODY, "body");

  const mjtNum* const sensor_pos  = sim_data->sensordata + imu_pos_sensor_adr;
  const mjtNum* const sensor_vel  = sim_data->sensordata + imu_vel_sensor_adr;
  const mjtNum* const sensor_acc  = sim_data->sensordata + imu_acc_sensor_adr;
  const mjtNum* const sensor_quat = sim_data->sensordata + imu_quat_sensor_adr;
  const mjtNum* const sensor_gyro = sim_data->sensordata + imu_gyro_sensor_adr;
  const mjtNum* const sensor_angacc = sim_data->sensordata + imu_angacc_sensor_adr;

  mj_utils::g_command_theta = param::INITIAL_DES_THETA;
  mjui_update(-1, -1, &mj_utils::g_ui, &mj_utils::g_ui_state, &mj_utils::g_context);

  // ---------------- [ Shared mailboxes ] ----------------
  std::mutex viewer_mtx;
  ViewerData shared_viewer_data{};
  shared_viewer_data.theta_d = mj_utils::g_command_theta;
  shared_viewer_data.perturb = mj_utils::g_perturb;

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
  std::thread th_sim([&]() {
    std::vector<Actuator::Servo> servo;
    servo.reserve(12);

    // --- Servo configuration ---
    std::array<Actuator::MotorParameters, 6> motor_parameters{};
    for (std::size_t i=0; i<motor_parameters.size(); ++i) {
      motor_parameters[i].ohm = param::MOTOR_OHM[i];
      motor_parameters[i].h = param::MOTOR_H[i];
      motor_parameters[i].Kt = param::MOTOR_KT[i];
      motor_parameters[i].Ke = param::MOTOR_KE[i];
      motor_parameters[i].reduction_ratio = param::MOTOR_REDUCTION_RATIO[i];
      motor_parameters[i].efficiency = param::MOTOR_EFFICIENCY[i];
      motor_parameters[i].max_voltage = param::MOTOR_MAX_VOLTAGE[i];
      motor_parameters[i].max_current = param::MOTOR_MAX_CURRENT[i];
      motor_parameters[i].max_torque = param::MOTOR_MAX_TORQUE[i];
      motor_parameters[i].viscous_friction = param::MOTOR_VISCOUS_FRICTION[i];
      motor_parameters[i].kP = param::MOTOR_KP[i];
      motor_parameters[i].kD = param::MOTOR_KD[i];
      motor_parameters[i].dt = param::MOTOR_DT[i];
    }
    for (std::size_t i=0; i<12; ++i) {servo.emplace_back(m, mj_utils::kActuatorNames[i], motor_parameters[i % motor_parameters.size()]);}

    MST mst{};
    State s{};
    Command cmd{};
    ViewerData viewer_data{};
    std::vector<mjtNum> snapshot_qpos(static_cast<std::size_t>(m->nq));
    std::vector<mjtNum> snapshot_qvel(static_cast<std::size_t>(m->nv));
    std::vector<mjtNum> added_mass_jac(6*static_cast<std::size_t>(m->nv));
    std::vector<mjtNum> added_mass_inertia_jac(6*static_cast<std::size_t>(m->nv));
    std::array<Eigen::Vector3d, 7> applied_aero_pos{};
    std::array<Eigen::Vector3d, 7> applied_aero_force{};
    mjtNum snapshot_time = sim_data->time;
    std::uint64_t handled_reset_epoch = viewer_data.reset_epoch;
    std::chrono::steady_clock::duration snapshot_elapsed = std::chrono::steady_clock::duration::zero();

    std::chrono::steady_clock::time_point next_tick = std::chrono::steady_clock::now();

    // =====================================================
    // ================ [ Simulation loop ] ================
    // =====================================================
    while (!g_stop.load(std::memory_order_acquire)) {
      next_tick += param::SIM_DT_US;

      { // --- Copy viewer data ---
        std::unique_lock<std::mutex> viewer_data_lk(viewer_mtx, std::try_to_lock);
        if (viewer_data_lk.owns_lock()) {viewer_data = shared_viewer_data;}
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
        handled_reset_epoch = viewer_data.reset_epoch;
      }

      snapshot_elapsed += param::SIM_DT_US;
      const bool snapshot_due = reset_requested || snapshot_elapsed >= param::RENDER_DT_US;
      if (snapshot_due) {snapshot_elapsed = reset_requested ? std::chrono::steady_clock::duration::zero() : snapshot_elapsed - param::RENDER_DT_US;}

      // --- ELRS RC parse ---
      if (elrs_enabled && elrs.update(elrs_channels)) {
        constexpr double ELRS_TO_RAD = M_PI / 1638.0;
        constexpr double ELRS_TO_ONE = 2.0 / 1638.0;
        cmd.theta[0]  = param::INITIAL_DES_THETA[0] - (static_cast<double>(elrs_channels[1]) - 992.0) * ELRS_TO_RAD;
        cmd.theta[1]  = param::INITIAL_DES_THETA[1] - (static_cast<double>(elrs_channels[3]) - 992.0) * (0.25 * ELRS_TO_RAD);
        cmd.theta[2]  = param::INITIAL_DES_THETA[2] - (static_cast<double>(elrs_channels[2]) - 172.0) * (0.50 * ELRS_TO_RAD);
        cmd.theta[3]  = param::INITIAL_DES_THETA[3] + param::KIN_GAIN * (static_cast<double>(elrs_channels[2]) - 172.0) * (0.50 * ELRS_TO_RAD);
        cmd.theta[4]  = J5_model(cmd.theta[2]);
        cmd.theta[5]  = param::INITIAL_DES_THETA[5] - param::KIN_GAIN * (static_cast<double>(elrs_channels[2]) - 172.0) * (0.50 * ELRS_TO_RAD);
        cmd.theta[6]  = param::INITIAL_DES_THETA[6] - (static_cast<double>(elrs_channels[1]) - 992.0) * ELRS_TO_RAD;
        cmd.theta[7]  = param::INITIAL_DES_THETA[7] - (static_cast<double>(elrs_channels[3]) - 992.0) * (0.25 * ELRS_TO_RAD);
        cmd.theta[8]  = param::INITIAL_DES_THETA[8] - (static_cast<double>(elrs_channels[2]) - 172.0) * (0.50 * ELRS_TO_RAD);
        cmd.theta[9]  = param::INITIAL_DES_THETA[9] + param::KIN_GAIN * (static_cast<double>(elrs_channels[2]) - 172.0) * (0.50 * ELRS_TO_RAD);
        cmd.theta[10] = J5_model(cmd.theta[8]);
        cmd.theta[11] = param::INITIAL_DES_THETA[11] - param::KIN_GAIN * (static_cast<double>(elrs_channels[2]) - 172.0) * (0.50 * ELRS_TO_RAD);

        // manual external wind gust
        const double vel_x = (static_cast<double>(elrs_channels[10]) - 992.0) * ELRS_TO_ONE * 20.0;
        const double vel_y = (static_cast<double>(elrs_channels[15]) - 992.0) * ELRS_TO_ONE * 10.0;
        const double vel_z = (static_cast<double>(elrs_channels[11]) - 992.0) * ELRS_TO_ONE * 10.0;
        s.vel_f = Eigen::Vector3d(vel_x, vel_y, vel_z);
        
        // std::printf("[ELRS] %llu", static_cast<unsigned long long>(elrs.valid_frames()));
        // for (std::size_t i=0; i<elrs_channels.size(); ++i) {std::printf(" %4u", static_cast<unsigned int>(elrs_channels[i]));}
        // std::putchar('\n'); std::fflush(stdout);
      }
      if (!elrs_enabled) {for(std::size_t i=0; i<12; ++i){cmd.theta[i] = static_cast<double>(viewer_data.theta_d[i]);}}

      // --- Control and servo dynamics ---
      if (!reset_requested) {

        // --- MuJoCo simulation step1 ---
        mju_zero(sim_data->xfrc_applied, 6*m->nbody);
        mju_zero(sim_data->qfrc_applied, m->nv);
        mjv_applyPerturbPose(m, sim_data, &viewer_data.perturb, viewer_data.paused);
        if (!viewer_data.paused) {mj_step1(m, sim_data);}

        // --- Servo dyn calc ---
        for(std::size_t i=0; i<12; ++i) {
          servo[i].desired_rad = cmd.theta[i];
          servo[i].step(*sim_data);
        }

        // --- MuJoCo simulation step2 ---
        mjv_applyPerturbForce(m, sim_data, &viewer_data.perturb);

        if (!viewer_data.paused) {
          for (std::size_t i=0; i<12; ++i) {sim_data->ctrl[servo[i].actuatorId()] = static_cast<mjtNum>(servo[i].motor_state.torque);}

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

            for (std::size_t i=0; i<12; ++i) {
              s.theta[i] = static_cast<double>(sim_data->qpos[servo_qpos_address[i]]);
              s.theta_dot[i] = static_cast<double>(sim_data->qvel[servo_qvel_address[i]]);
            }

            FK(s.theta, s.bTj);
            mst.update_dynamics(s);
          }

          const Eigen::Matrix3d GRb_FLU = param::NED_TO_FLU * s.R;
          const Eigen::Vector3d Gpb_FLU = param::NED_TO_FLU * s.pos;

          // Transfer MST's current added inertia to MuJoCo's generalized mass.
          add_spatial_inertias(m, sim_data, mst.added_mass_positions(), mst.added_mass_matrices(), wing_plate_ids, GRb_FLU, Gpb_FLU, added_mass_jac.data(), added_mass_inertia_jac.data());

          // Apply current t_k aerodynamic wrenches.
          const std::array<Eigen::Vector3d, 6>& bp = mst.positions();
          const std::array<Eigen::Vector3d, 6>& bF = mst.forces();
          const std::array<Eigen::Vector3d, 6>& bT = mst.torques();

          for (std::size_t i=0; i<6; ++i) {
            const Eigen::Vector3d GF = GRb_FLU * bF[i];
            const Eigen::Vector3d GT = GRb_FLU * bT[i];
            const Eigen::Vector3d Gp = Gpb_FLU + GRb_FLU * bp[i];
            const mjtNum force[3]  = {static_cast<mjtNum>(GF(0)), static_cast<mjtNum>(GF(1)), static_cast<mjtNum>(GF(2))};
            const mjtNum torque[3] = {static_cast<mjtNum>(GT(0)), static_cast<mjtNum>(GT(1)), static_cast<mjtNum>(GT(2))};
            const mjtNum point[3]  = {static_cast<mjtNum>(Gp(0)), static_cast<mjtNum>(Gp(1)), static_cast<mjtNum>(Gp(2))};
            mj_applyFT(m, sim_data, force, torque, point, wing_plate_ids[i], sim_data->qfrc_applied);
          }

          {
            const std::array<Eigen::Vector3d, 3>& body_elipsoid = mst.body_elipsoid();
            const Eigen::Vector3d GF = GRb_FLU * body_elipsoid[1];
            const Eigen::Vector3d GT = GRb_FLU * body_elipsoid[2];
            const Eigen::Vector3d Gp = Gpb_FLU + GRb_FLU * body_elipsoid[0];
            const mjtNum force[3]  = {static_cast<mjtNum>(GF(0)), static_cast<mjtNum>(GF(1)), static_cast<mjtNum>(GF(2))};
            const mjtNum torque[3] = {static_cast<mjtNum>(GT(0)), static_cast<mjtNum>(GT(1)), static_cast<mjtNum>(GT(2))};
            const mjtNum point[3]  = {static_cast<mjtNum>(Gp(0)), static_cast<mjtNum>(Gp(1)), static_cast<mjtNum>(Gp(2))};
            mj_applyFT(m, sim_data, force, torque, point, body_id, sim_data->qfrc_applied);
          }

          if (snapshot_due) {
            std::copy_n(sim_data->qpos, snapshot_qpos.size(), snapshot_qpos.begin());
            std::copy_n(sim_data->qvel, snapshot_qvel.size(), snapshot_qvel.begin());
            snapshot_time = sim_data->time;
            std::copy_n(mst.positions().begin(), 6, applied_aero_pos.begin());
            std::copy_n(mst.forces().begin(), 6, applied_aero_force.begin());
            const std::array<Eigen::Vector3d, 3>& body_elipsoid = mst.body_elipsoid();
            applied_aero_pos[6] = body_elipsoid[0];
            applied_aero_force[6] = body_elipsoid[1];
          }

          mj_step2(m, sim_data);

          { // --- Acceleration state solved at t_k ---
            s.acc(0) = static_cast<double>( sensor_acc[0]);
            s.acc(1) = static_cast<double>(-sensor_acc[1]);
            s.acc(2) = static_cast<double>(-sensor_acc[2]);
            s.w_dot = s.R.transpose() * Eigen::Vector3d(static_cast<double>(sensor_angacc[0]), static_cast<double>(-sensor_angacc[1]), static_cast<double>(-sensor_angacc[2]));
            for (std::size_t i=0; i<12; ++i) {s.theta_ddot[i] = static_cast<double>(sim_data->qacc[servo_qvel_address[i]]);}
            if (snapshot_due) {mst.update_visualization(s);}
          }
        }
      }
      if (snapshot_due && (reset_requested || viewer_data.paused)) {
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

        for (std::size_t i=0; i<12; ++i) {
          s.theta[i] = static_cast<double>(sim_data->qpos[servo_qpos_address[i]]);
          s.theta_dot[i] = static_cast<double>(sim_data->qvel[servo_qvel_address[i]]);
          s.theta_ddot[i] = static_cast<double>(sim_data->qacc[servo_qvel_address[i]]);
        }

        FK(s.theta, s.bTj);
        mst.update_visualization(s);
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
  std::array<Eigen::Vector3d, 7> copied_aero_pos{};
  std::array<Eigen::Vector3d, 7> copied_aero_force{};

  while (!glfwWindowShouldClose(window)) {
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

    // --- Process viewer input without simulation-state access ---
    glfwPollEvents();

    if (mj_utils::g_reset_epoch != handled_viewer_reset_epoch) {
      mj_utils::g_command_theta = param::INITIAL_DES_THETA;
      mjui_update(-1, -1, &mj_utils::g_ui, &mj_utils::g_ui_state, &mj_utils::g_context);
      handled_viewer_reset_epoch = mj_utils::g_reset_epoch;
    }

    ViewerData viewer_data{};
    for (std::size_t i=0; i<12; ++i) {viewer_data.theta_d[i] = mj_utils::g_command_theta[i];}
    viewer_data.perturb = mj_utils::g_perturb;
    viewer_data.paused = mj_utils::g_paused;
    viewer_data.reset_epoch = mj_utils::g_reset_epoch;
    
    { // Publish data
      std::lock_guard<std::mutex> viewer_data_lk(viewer_mtx);
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
      if (elrs_enabled){for(std::size_t i=0; i<12; ++i){mj_utils::g_command_theta[i] = static_cast<mjtNum>(shared_sim_data.theta_d[i]);}}
    }
    if (elrs_enabled) {mjui_update(-1, -1, &mj_utils::g_ui, &mj_utils::g_ui_state, &mj_utils::g_context);}
    mj_forward(m, render_data);
    
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

        switch (mj_utils::g_arrow_quantity) {
          case mj_utils::ARROW_NONE:
            break;

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

        mj_utils::append_humerus_strip_frames(copied_strip_state.p_h, wing*param::NH, GTb, bRhi, humerus_rotation.cos_psi[0]);
        mj_utils::append_radius_strip_frames(copied_strip_state.p_r, wing*param::NR, GTb, radius_rotation.bRri, radius_rotation.cos_psi);
        mj_utils::append_manus_strip_frames(copied_strip_state.p_m, wing*param::NM, GTb, bRmi, manus_rotation.cos_psi[0]);
      }

      for (std::size_t i=0; i<7; ++i) {
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
