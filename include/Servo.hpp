#pragma once

#include <mujoco/mujoco.h>

#include <cstdint>
#include <string>

namespace Actuator {

struct MotorParameters {
  double ohm = 0.5; // resistance [ohm]
  double h = 0.001; // inductance [H]

  double Kt = 0.05; // torque constant [Nm/A]
  double Ke = 0.05; // back EMF constant [V/(rad/s)]

  double reduction_ratio = 20.0;
  double efficiency = 0.85;

  double max_voltage = 24.0; // [V]
  double max_current = 20.0; // [A]
  double max_torque = 10.0; // [Nm]

  double viscous_friction = 0.002; // [Nm/(rad/s)]

  double kP = 0.0;   // [Nm/rad]
  double kD = 0.0;   // [Nm/(rad/s)]
  double dt = 0.001; // [sec]
};

struct MotorState {
  double rad = 0.0;
  double rad_s = 0.0;
  double torque = 0.0; // [Nm]
};

class Servo {
 public:
  double desired_rad = 0.0;
  MotorState motor_state;

  Servo(
    const mjModel* model,
    const std::string& actuator_name,
    MotorParameters motor_parameters = {}
  );

  void reset();
  void step(const mjData& data);

  std::int32_t actuatorId() const noexcept;

 private:
  MotorParameters motor_;

  std::int32_t actuator_id_ = -1;
  std::int32_t joint_id_ = -1;
  std::int32_t qpos_address_ = -1;
  std::int32_t qvel_address_ = -1;

  double ctrl_min_ = 0.0;
  double ctrl_max_ = 0.0;

  double current_ = 0.0;
  double current_decay_ = 0.6;
  double previous_time_ = 0.0;
};

} // namespace Actuator