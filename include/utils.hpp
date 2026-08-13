#pragma once

#include "params.hpp"
#include "MST.hpp"

#include <mujoco/mujoco.h>
#include <vector>
#include <cstdio>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>
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
  std::array<double, 12> theta{};                        // [rad]
  std::array<double, 12> theta_dot{};                    // [rad/s]
  std::array<double, 12> theta_ddot{};                   // [rad/s^2]
  std::array<Eigen::Matrix4d, 12> bTj{};                 // [SE3], wing pose, body FRD
};

struct Command {
  Eigen::Vector3d pos   = Eigen::Vector3d::Zero();     // [m]
  Eigen::Vector3d vel   = Eigen::Vector3d::Zero();     // [m/s]
  Eigen::Vector3d acc   = Eigen::Vector3d::Zero();     // [m/s^2]
  Eigen::Matrix3d R     = Eigen::Matrix3d::Identity(); // [SO3]
  Eigen::Vector3d w     = Eigen::Vector3d::Zero();     // [rad/s]
  Eigen::Vector3d w_dot = Eigen::Vector3d::Zero();     // [rad/s^2]
  std::array<double, 12> theta{};                      // [rad]
};

struct ViewerData {
  std::array<mjtNum, 12> theta_d{};
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
  std::array<Eigen::Vector3d, 6> aero_pos{};   // [m], body FRD
  std::array<Eigen::Vector3d, 6> aero_force{}; // [N], body FRD
  std::array<double, 12> theta_d{};
};

static inline Eigen::Matrix3d hat(const Eigen::Vector3d v){
  Eigen::Matrix3d V;
  V.setZero();

  V(2,1) = v(0);
  V(1,2) = -V(2, 1);
  V(0,2) = v(1);
  V(2,0) = -V(0, 2);
  V(1,0) = v(2);
  V(0,1) = -V(1, 0);

  return V;
}

static inline Eigen::Vector3d vee(const Eigen::Matrix3d V){
  Eigen::Vector3d v;
  Eigen::Matrix3d E;

  v.setZero();
  E = V + V.transpose();
  
  if(E.norm() > 1.e-6) {std::fprintf(stderr, "VEE: E.norm() = %.6f\n", E.norm());}

  v(0) = V(2, 1);
  v(1) = V(0, 2);
  v(2) = V(1, 0);

  return  v;
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

static inline void FK(const std::array<double, 12>& theta, std::array<Eigen::Matrix4d, 12>& frame_poses) {
  for (std::size_t wing=0; wing<2; ++wing) {
    Eigen::Matrix3d bRj = Eigen::Matrix3d::Identity();
    Eigen::Vector3d bpj = Eigen::Vector3d::Zero();

    for (std::size_t j=wing*6; j<wing*6+6; ++j) {
      const Eigen::Matrix4d& fixed_transform = param::WING_FIXED_TRANSFORM[j];
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
}

static inline double J5_model(const double J3) {
  // return 0.18702 + 0.308759 * J3 - 0.0707378 / (J3 + 1.05901);
  return -0.1356*J3*J3*J3 - 0.2059*J3*J3 + 0.1409*J3 + 0.1719;
}

static inline std::int64_t now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
