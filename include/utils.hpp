#pragma once

#include "params.hpp"
#include "MST.hpp"

#include <mujoco/mujoco.h>
#include <vector>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <Eigen/Dense>

struct State {
  Eigen::Vector3d pos   = Eigen::Vector3d::Zero();       // [m], world NED
  Eigen::Vector3d vel   = Eigen::Vector3d::Zero();       // [m/s], world NED
  Eigen::Vector3d acc   = Eigen::Vector3d::Zero();       // [m/s^2], world NED
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

struct Command {
  Eigen::Vector3d pos   = Eigen::Vector3d::Zero();     // [m]
  Eigen::Vector3d vel   = Eigen::Vector3d::Zero();     // [m/s]
  Eigen::Matrix3d R     = Eigen::Matrix3d::Identity(); // [SO3]
  Eigen::Vector3d w     = Eigen::Vector3d::Zero();     // [rad/s]
  std::array<double, param::NUM_JOINTS> theta{};       // [rad]
  double theta_t = 0.0;                                // [rad]
};

struct ViewerData {
  std::array<mjtNum, param::NUM_JOINTS> theta_d{};
  mjtNum theta_t = 0.0;
  mjtNum sim_speed = 1.0; // [0, 1], simulation time / wall time
  mjvPerturb perturb{};
  bool paused = false;
  std::uint64_t reset_epoch = 0;
};

struct SimData {
  std::vector<mjtNum> qpos;
  std::vector<mjtNum> qvel;
  mjtNum time = 0;
  State state{};
  MST::StripState strip_state{};
  std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS> aero_pos{};   // [m], body FRD; last = body ellipsoid
  std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS> aero_force{}; // [N], body FRD; last = body ellipsoid
  std::array<double, param::NUM_JOINTS> theta_d{};
};

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

static inline double J5_model(const double J3) {
  // return 0.18702 + 0.308759 * J3 - 0.0707378 / (J3 + 1.05901);
  return -0.1356*J3*J3*J3 - 0.2059*J3*J3 + 0.1409*J3 + 0.1719;
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
  }

  mj_factorM(m, d);
  if (d->nefc > 0) {mj_projectConstraint(m, d);}
}
