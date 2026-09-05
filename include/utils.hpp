#pragma once

#include "params.hpp"
#include "MST.hpp"
#include "ELRS.hpp"

#include <mujoco/mujoco.h>
#include <vector>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <Eigen/Dense>

struct State {
  Eigen::Vector3d pos   = Eigen::Vector3d::Zero();       // [m], body origin, world NED
  Eigen::Vector3d vel   = Eigen::Vector3d::Zero();       // [m/s], body origin, world NED
  Eigen::Matrix3d R     = Eigen::Matrix3d::Identity();   // [SO3], body FRD -> world NED
  Eigen::Vector3d w     = Eigen::Vector3d::Zero();       // [rad/s] body FRD
  Eigen::Vector3d bpc   = Eigen::Vector3d::Zero();       // [m], body origin -> estimated system CoM, body FRD
  Eigen::Matrix3d MoI   = 1.0e-3f * Eigen::Matrix3d::Identity(); // [kg.m^2], about estimated system CoM
  std::array<double, param::NUM_JOINTS> theta{};         // [rad], J1-J14
  std::array<double, param::NUM_JOINTS> theta_dot{};     // [rad/s], J1-J14
  Eigen::Vector3d vel_f = Eigen::Vector3d::Zero();       // [m/s], world NED
};

struct Command {
  // Flight control
  Eigen::Vector3d pos   = Eigen::Vector3d::Zero();     // [m]
  Eigen::Vector3d vel   = Eigen::Vector3d::Zero();     // [m/s]
  Eigen::Matrix3d R     = Eigen::Matrix3d::Identity(); // [SO3]
  Eigen::Vector3d w     = Eigen::Vector3d::Zero();     // [rad/s]

  // Desired control input. order : [f, Af_bar, Af_delta, Ap_bar, Ap_delta, sweep]
  Eigen::Matrix<double, 6, 1> u = (Eigen::Matrix<double, 6, 1>() << param::MIN_FREQ, param::MIN_FLAPPING_AMPLITUDE, 0.0, param::MIN_PITCHING_AMPLITUDE, 0.0, 0.0).finished();

  // Joint control
  std::array<double, param::NUM_JOINTS> theta = param::INITIAL_DES_THETA; // [rad]
  double theta_t = 0.0; // [rad]
};

struct Phase {
  enum Type : std::uint8_t {
    PAUSED = 0,
    JOINT_MANUAL = 1,
    INPUT_MANUAL = 2,
    WRENCH_MANUAL = 3
  };

  Type type = JOINT_MANUAL;
};

struct SimState {
  Eigen::Vector3d pos   = Eigen::Vector3d::Zero();       // [m], body origin, world NED
  Eigen::Vector3d bpc   = Eigen::Vector3d::Zero();       // [m], body origin -> system CoM, body FRD
  Eigen::Vector3d vel   = Eigen::Vector3d::Zero();       // [m/s], body origin, world NED
  Eigen::Vector3d acc   = Eigen::Vector3d::Zero();       // [m/s^2], body origin, world NED
  Eigen::Matrix3d R     = Eigen::Matrix3d::Identity();   // [SO3], body FRD -> world NED
  Eigen::Vector3d w     = Eigen::Vector3d::Zero();       // [rad/s] body FRD
  Eigen::Vector3d w_dot = Eigen::Vector3d::Zero();       // [rad/s^2] body FRD
  Eigen::Matrix3d MoI   = 1.0e-3f * Eigen::Matrix3d::Identity(); // [kg.m^2]
  Eigen::Vector3d vel_f = Eigen::Vector3d::Zero();       // [m/s], world NED
  std::array<double, param::NUM_JOINTS> theta{};         // [rad], J1-J14
  std::array<double, param::NUM_JOINTS> theta_dot{};     // [rad/s], J1-J14
  std::array<double, param::NUM_JOINTS> theta_ddot{};    // [rad/s^2], J1-J14
  std::array<Eigen::Matrix4d, param::NUM_JOINTS> bTj{};  // [SE3], joint frames, body FRD
};

struct ViewerData {
  std::array<mjtNum, param::NUM_JOINTS> theta_d{};
  mjtNum theta_t = 0.0;
  mjtNum sim_speed = 1.0; // [0, 1], simulation time / wall time
  mjvPerturb perturb{};
  std::uint64_t reset_epoch = 0;
};

struct SimData {
  std::vector<mjtNum> qpos;
  std::vector<mjtNum> qvel;
  mjtNum time = 0;
  SimState state{};
  MST::StripState strip_state{};
  std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS> aero_pos{};   // [m], body FRD; last = body ellipsoid
  std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS> aero_force{}; // [N], body FRD; last = body ellipsoid
  std::array<double, param::NUM_JOINTS> theta_d{};
};

static inline void initialize_mass_estimate(State& state, SimState& sim_state, const mjModel* model, const mjData* data, const int root_body_id) noexcept {
  if (model->body_subtreemass[root_body_id] <= 0.0) {return;}

  const Eigen::Vector3d com_world(
    static_cast<double>(data->subtree_com[3*root_body_id]),
    static_cast<double>(data->subtree_com[3*root_body_id+1]),
    static_cast<double>(data->subtree_com[3*root_body_id+2])
  );

  Eigen::Matrix3d moi_world = Eigen::Matrix3d::Zero();
  for (int body=1; body<model->nbody; ++body) {
    int ancestor = body;
    while (ancestor > 0 && ancestor != root_body_id) {ancestor = model->body_parentid[ancestor];}
    if (ancestor != root_body_id) {continue;}

    const double mass = static_cast<double>(model->body_mass[body]);
    const Eigen::Vector3d body_com(
      static_cast<double>(data->xipos[3*body]),
      static_cast<double>(data->xipos[3*body+1]),
      static_cast<double>(data->xipos[3*body+2])
    );
    const Eigen::Vector3d offset = body_com-com_world;

    Eigen::Matrix3d world_from_inertial;
    for (int row=0; row<3; ++row) {
      for (int col=0; col<3; ++col) {world_from_inertial(row, col) = static_cast<double>(data->ximat[9*body+3*row+col]);}
    }
    const Eigen::Vector3d principal_inertia(
      static_cast<double>(model->body_inertia[3*body]),
      static_cast<double>(model->body_inertia[3*body+1]),
      static_cast<double>(model->body_inertia[3*body+2])
    );
    moi_world.noalias() += world_from_inertial*principal_inertia.asDiagonal()*world_from_inertial.transpose();
    moi_world.noalias() += mass*(offset.squaredNorm()*Eigen::Matrix3d::Identity()-offset*offset.transpose());
  }

  const Eigen::Vector3d body_origin(
    static_cast<double>(data->xpos[3*root_body_id]),
    static_cast<double>(data->xpos[3*root_body_id+1]),
    static_cast<double>(data->xpos[3*root_body_id+2])
  );
  Eigen::Matrix3d world_from_body_frd;
  for (int row=0; row<3; ++row) {
    for (int col=0; col<3; ++col) {world_from_body_frd(row, col) = static_cast<double>(data->xmat[9*root_body_id+3*row+col]);}
  }
  world_from_body_frd *= param::NED_TO_FLU;

  state.bpc = world_from_body_frd.transpose()*(com_world-body_origin);
  state.MoI = world_from_body_frd.transpose()*moi_world*world_from_body_frd;
  state.MoI = 0.5*(state.MoI+state.MoI.transpose());
  sim_state.bpc = state.bpc;
  sim_state.MoI = state.MoI;
}

static inline void update_state_estimate(State& state, const SimState& sim_state) noexcept {
  // This is the estimator boundary. Measurement noise and delay can be added here later.
  state.pos = sim_state.pos;
  state.vel = sim_state.vel;
  state.R = sim_state.R;
  state.w = sim_state.w;
  state.theta = sim_state.theta;
  state.theta_dot = sim_state.theta_dot;
  state.vel_f = sim_state.vel_f;
}

static inline Eigen::Matrix3d quat_to_R(const mjtNum* q) {
  // q: body FLU -> world FLU
  // R: body FRD -> world NED
  const double w = q[0];
  const double x = q[1];
  const double y = q[2];
  const double z = q[3];

  const double x2 = 2.0 * x;
  const double y2 = 2.0 * y;
  const double z2 = 2.0 * z;

  const double xx = x * x2;
  const double yy = y * y2;
  const double zz = z * z2;
  const double xy = x * y2;
  const double xz = x * z2;
  const double yz = y * z2;
  const double wx = w * x2;
  const double wy = w * y2;
  const double wz = w * z2;

  Eigen::Matrix3d R;
  R << 1.0 - yy - zz,  -xy + wz,        -xz - wy,
       -xy - wz,        1.0 - xx - zz,   yz - wx,
       -xz + wy,         yz + wx,        1.0 - xx - yy;

  return R;
}

static inline void FK(const std::array<double, param::NUM_JOINTS>& theta, std::array<Eigen::Matrix4d, param::NUM_JOINTS>& frame_poses) {
  // Wing
  for (std::size_t wing=0; wing<2; ++wing) {
    Eigen::Matrix3d bRj = Eigen::Matrix3d::Identity();
    Eigen::Vector3d bpj = Eigen::Vector3d::Zero();

    for (std::size_t j=wing*param::NUM_WING_JOINTS_PER_WING; j<wing*param::NUM_WING_JOINTS_PER_WING+param::NUM_WING_JOINTS_PER_WING; ++j) {
      const Eigen::Matrix4d& fixed_transform = param::JOINT_FIXED_TRANSFORM[j];
      bpj += bRj * fixed_transform.block<3, 1>(0, 3);
      bRj = bRj * fixed_transform.block<3, 3>(0, 0);

      const double c = std::cos(theta[j]);
      const double s = std::sin(theta[j]);
      const Eigen::Vector3d y = bRj.col(1);
      const Eigen::Vector3d z = bRj.col(2);
      bRj.col(1) =  c*y + s*z;
      bRj.col(2) = -s*y + c*z;

      frame_poses[j].setIdentity();
      frame_poses[j].block<3, 3>(0, 0) = bRj;
      frame_poses[j].block<3, 1>(0, 3) = bpj;
    }
  }

  // Tail
  Eigen::Matrix3d bRj = Eigen::Matrix3d::Identity();
  Eigen::Vector3d bpj = Eigen::Vector3d::Zero();
  for (std::size_t j=param::NUM_WING_JOINTS; j<param::NUM_WING_JOINTS+param::NUM_TAIL_JOINTS; ++j) {
    const Eigen::Matrix4d& fixed_transform = param::JOINT_FIXED_TRANSFORM[j];
    bpj += bRj * fixed_transform.block<3, 1>(0, 3);
    bRj = bRj * fixed_transform.block<3, 3>(0, 0);

    const double c = std::cos(theta[j]);
    const double s = std::sin(theta[j]);
    const Eigen::Vector3d y = bRj.col(1);
    const Eigen::Vector3d z = bRj.col(2);
    bRj.col(1) =  c*y + s*z;
    bRj.col(2) = -s*y + c*z;

    frame_poses[j].setIdentity();
    frame_poses[j].block<3, 3>(0, 0) = bRj;
    frame_poses[j].block<3, 1>(0, 3) = bpj;
  }
}

static inline double J5_model(const double J3) {return -0.1356*J3*J3*J3 - 0.2059*J3*J3 + 0.1409*J3 + 0.1719;}

static inline void update_elrs_command(const ELRS::Channels& channels, const Phase& phase, Command& cmd, Eigen::Matrix<double, 6, 1>& wrench_bar) noexcept {
  const double ch0  = std::clamp(static_cast<double>(channels[0]),  param::ELRS_MIN, param::ELRS_MAX);
  const double ch1  = std::clamp(static_cast<double>(channels[1]),  param::ELRS_MIN, param::ELRS_MAX);
  const double ch2  = std::clamp(static_cast<double>(channels[2]),  param::ELRS_MIN, param::ELRS_MAX);
  const double ch3  = std::clamp(static_cast<double>(channels[3]),  param::ELRS_MIN, param::ELRS_MAX);
  const double ch10 = std::clamp(static_cast<double>(channels[10]), param::ELRS_MIN, param::ELRS_MAX);
  const double ch11 = std::clamp(static_cast<double>(channels[11]), param::ELRS_MIN, param::ELRS_MAX);

  const double s0  = ch0  < param::ELRS_CENTER ? (ch0 -param::ELRS_CENTER)/(param::ELRS_CENTER-param::ELRS_MIN) : (ch0 -param::ELRS_CENTER)/(param::ELRS_MAX-param::ELRS_CENTER);
  const double s1  = ch1  < param::ELRS_CENTER ? (ch1 -param::ELRS_CENTER)/(param::ELRS_CENTER-param::ELRS_MIN) : (ch1 -param::ELRS_CENTER)/(param::ELRS_MAX-param::ELRS_CENTER);
  const double s2  = ch2  < param::ELRS_CENTER ? (ch2 -param::ELRS_CENTER)/(param::ELRS_CENTER-param::ELRS_MIN) : (ch2 -param::ELRS_CENTER)/(param::ELRS_MAX-param::ELRS_CENTER);
  const double s3  = ch3  < param::ELRS_CENTER ? (ch3 -param::ELRS_CENTER)/(param::ELRS_CENTER-param::ELRS_MIN) : (ch3 -param::ELRS_CENTER)/(param::ELRS_MAX-param::ELRS_CENTER);
  const double s10 = ch10 < param::ELRS_CENTER ? (ch10-param::ELRS_CENTER)/(param::ELRS_CENTER-param::ELRS_MIN) : (ch10-param::ELRS_CENTER)/(param::ELRS_MAX-param::ELRS_CENTER);
  const double s11 = ch11 < param::ELRS_CENTER ? (ch11-param::ELRS_CENTER)/(param::ELRS_CENTER-param::ELRS_MIN) : (ch11-param::ELRS_CENTER)/(param::ELRS_MAX-param::ELRS_CENTER);

  wrench_bar(0) = param::ELRS_WRENCH_FX_AT_MIN + (param::ELRS_WRENCH_FX_AT_MAX-param::ELRS_WRENCH_FX_AT_MIN)*0.5*(s10+1.0); // Fx [N]
  wrench_bar(1) = 0.0;                                                                                               // Fy [N]
  wrench_bar(2) = param::ELRS_WRENCH_FZ_AT_MIN + (param::ELRS_WRENCH_FZ_AT_MAX-param::ELRS_WRENCH_FZ_AT_MIN)*0.5*(s2+1.0);   // Fz [N]
  wrench_bar(3) = param::ELRS_WRENCH_MX_MAX*s3;                                                                       // Mx [N.m]
  wrench_bar(4) = param::ELRS_WRENCH_MY_MAX*s1;                                                                       // My [N.m]
  wrench_bar(5) = param::ELRS_WRENCH_MZ_MAX*s0;                                                                       // Mz [N.m]

  if (phase.type != Phase::INPUT_MANUAL) {return;}

  cmd.u(0) = param::MIN_FREQ + (param::MAX_FREQ-param::MIN_FREQ)*0.5*(s10+1.0);
  cmd.u(1) = param::MIN_FLAPPING_AMPLITUDE + (param::MAX_FLAPPING_AMPLITUDE-param::MIN_FLAPPING_AMPLITUDE)*0.5*(s2+1.0);
  cmd.u(2) = param::MAX_FLAPPING_DIFFERENCE*s0;
  cmd.u(3) = param::MIN_PITCHING_AMPLITUDE + (param::MAX_PITCHING_AMPLITUDE-param::MIN_PITCHING_AMPLITUDE)*0.5*(s1+1.0);
  cmd.u(4) = param::MAX_PITCHING_DIFFERENCE*s3;
  cmd.u(5) = param::MIN_SWEEP_BIAS + (param::MAX_SWEEP_BIAS-param::MIN_SWEEP_BIAS)*0.5*(s11+1.0);
}

static inline void update_joint_command(Command& cmd, double& flapping_phase) noexcept {
  const double cycle_ratio = flapping_phase/(2.0*M_PI);
  double cosR1; double sinR1;
  if (cycle_ratio < param::R1) {cosR1 = -std::cos(M_PI*cycle_ratio/param::R1); sinR1 = std::sin(M_PI*cycle_ratio/param::R1);}
  else {cosR1 = std::cos(M_PI*(cycle_ratio-param::R1)/(1.0-param::R1)); sinR1 = std::sin(M_PI*(param::R1-cycle_ratio)/(1.0-param::R1));}
  double one_minus_cosR2 = 0.0;
  if (cycle_ratio > param::R2) {one_minus_cosR2 = 1.0-std::cos(2.0*M_PI*(cycle_ratio-param::R2)/(1.0-param::R2));}

  const double flapping_right = param::FLAPPING_DELTA_0 + (cmd.u(1)+0.5*cmd.u(2))*cosR1;
  const double flapping_left  = param::FLAPPING_DELTA_0 + (cmd.u(1)-0.5*cmd.u(2))*cosR1;
  const double pitching_right = param::PITCHING_DELTA_0 + 0.5*(cmd.u(3)+0.5*cmd.u(4))*one_minus_cosR2;
  const double pitching_left  = param::PITCHING_DELTA_0 + 0.5*(cmd.u(3)-0.5*cmd.u(4))*one_minus_cosR2;
  const double sweep = cmd.u(5) + param::SWEEP_AMPLITUDE*sinR1;
  const double folding = param::FOLDING_DELTA_0 + 0.5*param::FOLDING_AMPLITUDE*one_minus_cosR2;

  cmd.theta[0] = param::INITIAL_DES_THETA[0] + flapping_right;
  cmd.theta[1] = param::INITIAL_DES_THETA[1] + pitching_right;
  cmd.theta[2] = param::INITIAL_DES_THETA[2] + sweep;
  cmd.theta[3] = param::INITIAL_DES_THETA[3] + folding;
  cmd.theta[4] = J5_model(cmd.theta[2]);
  cmd.theta[5] = param::INITIAL_DES_THETA[5] - 2.0*folding;

  cmd.theta[6]  = param::INITIAL_DES_THETA[6] + flapping_left;
  cmd.theta[7]  = param::INITIAL_DES_THETA[7] + pitching_left;
  cmd.theta[8]  = param::INITIAL_DES_THETA[8] + sweep;
  cmd.theta[9]  = param::INITIAL_DES_THETA[9] + folding;
  cmd.theta[10] = J5_model(cmd.theta[8]);
  cmd.theta[11] = param::INITIAL_DES_THETA[11] - 2.0*folding;

  flapping_phase += 2.0*M_PI*cmd.u(0)*param::SIM_DT_SEC;
  if (flapping_phase >= 2.0*M_PI) {flapping_phase -= 2.0*M_PI;}
}

template <std::size_t N>
inline void add_spatial_inertias(const mjModel* m, mjData* d, const std::array<Eigen::Vector3d, N>& local_positions, const std::array<Eigen::Matrix<double, 6, 6>, N>& local_spatial_inertias, const std::array<int, N>& body_ids, const Eigen::Matrix3d& world_R_body, const Eigen::Vector3d& world_p_body, mjtNum* jac, mjtNum* inertia_jac) {
  const int nv = m->nv;
  mjtNum* const jacp = jac;
  mjtNum* const jacr = jac + 3*nv;
  const Eigen::Matrix3d world_R_body_T = world_R_body.transpose();
  Eigen::Matrix<double, 6, 6> world_spatial_inertia;

  for (std::size_t i=0; i<N; ++i) {
    const int body_id = body_ids[i];
    if (body_id <= 0) {continue;}

    const Eigen::Vector3d point = world_p_body + world_R_body*local_positions[i];
    const mjtNum point_mj[3] = {static_cast<mjtNum>(point(0)), static_cast<mjtNum>(point(1)), static_cast<mjtNum>(point(2))};
    mj_jac(m, d, jacp, jacr, point_mj, body_id);

    // Rotate the 6x6 spatial inertia once instead of rotating every Jacobian column.
    for (int block_row=0; block_row<2; ++block_row) {
      for (int block_col=0; block_col<2; ++block_col) {
        world_spatial_inertia.block<3, 3>(3*block_row, 3*block_col).noalias() = world_R_body * local_spatial_inertias[i].template block<3, 3>(3*block_row, 3*block_col) * world_R_body_T;
      }
    }

    for (int row=0; row<6; ++row) {
      for (int col=0; col<nv; ++col) {
        mjtNum value = 0;
        for (int k=0; k<6; ++k) {value += static_cast<mjtNum>(world_spatial_inertia(row,k))*jac[k*nv+col];}
        inertia_jac[row*nv+col] = value;
      }
    }

    #if mjVERSION_HEADER >= 3011000
    // MuJoCo >= 3.11: inertia matrix is stored directly in lower-triangular CSR.
    for (int row=0; row<nv; ++row) {
      const int row_adr = m->M_rowadr[row];
      const int row_nnz = m->M_rownnz[row];

      for (int k=0; k<row_nnz; ++k) {
        const int address = row_adr + k;
        const int col = m->M_colind[address];

        mjtNum value = 0;
        for (int j=0; j<6; ++j) {value += jac[j*nv+row]*inertia_jac[j*nv+col];}
        d->M[address] += value;
      }
    }
    #else
    // MuJoCo <= 3.10: legacy tree-sparse qM layout.
    for (int row=0; row<nv; ++row) {
      int address = m->dof_Madr[row];
      int col = row;
      while (col >= 0) {
        mjtNum value = 0;
        for (int k=0; k<6; ++k) {value += jac[k*nv+row]*inertia_jac[k*nv+col];}
        d->qM[address++] += value;
        col = m->dof_parentid[col];
      }
    }
    #endif
  }

  mj_factorM(m, d);
  if (d->nefc > 0) {mj_projectConstraint(m, d);}
}
