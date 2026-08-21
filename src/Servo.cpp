#include "Servo.hpp"

#include <algorithm>
#include <cmath>

namespace Actuator {

Servo::Servo(const MotorParameters motor_parameters)
  : motor_(motor_parameters),
    current_decay_(std::exp(-motor_.ohm * motor_.dt / motor_.h)),
    esc_command_decay_(motor_.esc_time_constant > 0.0 ? std::exp(-motor_.dt / motor_.esc_time_constant) : 0.0),
    torque_per_amp_(motor_.Kt * motor_.reduction_ratio * motor_.efficiency),
    inv_torque_per_amp_(1.0 / torque_per_amp_),
    back_emf_per_joint_rad_s_(motor_.Ke * motor_.reduction_ratio),
    inv_ohm_(1.0 / motor_.ohm) {}

void Servo::step(const double theta, const double theta_dot) {
  motor_state.rad = theta;
  motor_state.rad_s = theta_dot;

  const double position_error = desired_rad - motor_state.rad;
  const double desired_torque = std::clamp(
    motor_.kP * position_error - motor_.kD * motor_state.rad_s,
    -motor_.max_torque, motor_.max_torque
  );

  const double desired_current = desired_torque * inv_torque_per_amp_;
  // Exact update of tau_esc * i_cmd_dot + i_cmd = i_desired.
  current_command_ = desired_current + (current_command_ - desired_current) * esc_command_decay_;

  const double back_emf = back_emf_per_joint_rad_s_ * motor_state.rad_s;
  const double voltage = motor_.ohm * current_command_ + back_emf;
  const double steady_state_current = (voltage - back_emf) * inv_ohm_;

  // Exact zero-order-hold update of L * i_dot = V - R*i - Ke*omega_m.
  current_ = steady_state_current + (current_ - steady_state_current) * current_decay_;

  const double electromagnetic_torque = torque_per_amp_ * current_;
  const double friction_torque = motor_.viscous_friction * motor_state.rad_s;
  motor_state.torque = std::clamp(electromagnetic_torque - friction_torque, -motor_.max_torque, motor_.max_torque);
}

void Servo::reset() {
  motor_state.rad = 0.0;
  motor_state.rad_s = 0.0;
  motor_state.torque = 0.0;
  current_ = 0.0;
  current_command_ = 0.0;
}

} // namespace Actuator
