#pragma once

namespace Actuator {

struct MotorParameters {
  double ohm = 0.5; // resistance [ohm]
  double h = 0.001; // inductance [H]

  double Kt = 0.05; // torque constant [Nm/A]
  double Ke = 0.05; // back EMF constant [V/(rad/s)]

  double reduction_ratio = 20.0;
  double efficiency = 0.85;

  double max_torque = 10.0; // [Nm]

  double viscous_friction = 0.002; // [Nm/(rad/s)]

  double esc_time_constant = 0.0; // current-command time constant [sec]

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

  explicit Servo(MotorParameters motor_parameters);

  void reset();
  void step(double theta, double theta_dot);

 private:
  MotorParameters motor_;

  double current_ = 0.0;
  double current_command_ = 0.0;
  double current_decay_ = 0.6;
  double esc_command_decay_ = 0.0;
  double torque_per_amp_ = 0.0;
  double inv_torque_per_amp_ = 0.0;
  double back_emf_per_joint_rad_s_ = 0.0;
  double inv_ohm_ = 0.0;
};

} // namespace Actuator
