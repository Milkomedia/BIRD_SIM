#pragma once

#include "params.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstddef>

struct State;

class Mixer {
public:
  static constexpr std::size_t NSTRIP_REDUCTION = 2;
  static constexpr std::size_t N_PHASE  = 24;

  static_assert(NSTRIP_REDUCTION > 0, "NSTRIP_REDUCTION must be greater than zero.");
  static_assert(N_PHASE > 0, "N_PHASE must be greater than zero.");
  static_assert(NSTRIP_REDUCTION <= param::NH && NSTRIP_REDUCTION <= param::NR && NSTRIP_REDUCTION <= param::NM, "NSTRIP_REDUCTION must produce at least one strip per section.");
  static constexpr std::size_t NH = (param::NH + NSTRIP_REDUCTION - 1) / NSTRIP_REDUCTION;
  static constexpr std::size_t NR = (param::NR + NSTRIP_REDUCTION - 1) / NSTRIP_REDUCTION;
  static constexpr std::size_t NM = (param::NM + NSTRIP_REDUCTION - 1) / NSTRIP_REDUCTION;
  static constexpr std::size_t TOTAL_SAMPLES  = 2*(NH + NR + NM) * N_PHASE;

  Mixer() noexcept {reset();}
  void reset() noexcept;

  // prev_input = [f, Af_bar, Af_delta, Ap_bar, Ap_delta, sweep].
  const Eigen::Matrix<double, 6, 1>& update(const State& state, const Eigen::Matrix<double, 6, 1>& prev_input) noexcept;

private:
  static constexpr std::size_t HUMERUS_SAMPLE_COUNT = 2 * N_PHASE * NH;
  static constexpr std::size_t RADIUS_SAMPLE_COUNT = 2 * N_PHASE * NR;
  static constexpr std::size_t MANUS_SAMPLE_COUNT = 2 * N_PHASE * NM;
  static constexpr std::size_t RADIUS_SAMPLE_BEGIN = HUMERUS_SAMPLE_COUNT;
  static constexpr std::size_t MANUS_SAMPLE_BEGIN = HUMERUS_SAMPLE_COUNT + RADIUS_SAMPLE_COUNT;

  // Cached strip kinematics; all vectors and rotations are expressed in body FRD.
  std::array<Eigen::Vector3d, TOTAL_SAMPLES> bp_ac_{};
  std::array<Eigen::Vector3d, TOTAL_SAMPLES> bv_ac_{};
  std::array<Eigen::Matrix3d, TOTAL_SAMPLES> bRsi_{};
  std::array<double, TOTAL_SAMPLES> c_{};
  std::array<double, TOTAL_SAMPLES> area_{};

  Eigen::Matrix<double, 6, 1> wrench_ = Eigen::Matrix<double, 6, 1>::Zero();

  void rebuild_kinematics(const Eigen::Matrix<double, 6, 1>& prev_input) noexcept;
  void accumulate_range(std::size_t begin, std::size_t end, const Eigen::Vector3d& RtVrel, const Eigen::Vector3d& b_omega, const Eigen::Vector3d& bpc, const double (&CD)[176][14], const double (&CL)[176][14], const double (&CM)[176][14]) noexcept;

  static inline void rotate_x(Eigen::Matrix3d& rotation, const double angle) noexcept {
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const Eigen::Vector3d y = rotation.col(1);
    const Eigen::Vector3d z = rotation.col(2);
    rotation.col(1) = c*y + s*z;
    rotation.col(2) = -s*y + c*z;
  }

  static inline double j5_angle(const double j3) noexcept {return -0.1356*j3*j3*j3 -0.2059*j3*j3 + 0.1409*j3 + 0.1719;}
  static inline double j5_slope(const double j3) noexcept {return -0.4068*j3*j3 - 0.4118*j3 + 0.1409;}
};
