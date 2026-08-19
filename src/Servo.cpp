#include "Servo.hpp"

#include <algorithm>
#include <cmath>

namespace Actuator {

Servo::Servo(const MotorParameters motor_parameters, const double control_min_torque, const double control_max_torque)
  : motor_(motor_parameters),
    current_decay_(std::exp(-motor_.ohm * motor_.dt / motor_.h)),
    torque_per_amp_(motor_.Kt * motor_.reduction_ratio * motor_.efficiency),
    inv_torque_per_amp_(1.0 / torque_per_amp_),
    back_emf_per_joint_rad_s_(motor_.Ke * motor_.reduction_ratio),
    inv_ohm_(1.0 / motor_.ohm),
    minimum_torque_(std::max(control_min_torque, -motor_.max_torque)),
    maximum_torque_(std::min(control_max_torque, motor_.max_torque)) {}

void Servo::step(const double theta, const double theta_dot) {
  motor_state.rad = theta;
  motor_state.rad_s = theta_dot;

  const double position_error = desired_rad - motor_state.rad;
  const double desired_torque = std::clamp(
    motor_.kP * position_error - motor_.kD * motor_state.rad_s,
    -motor_.max_torque, motor_.max_torque
  );

  const double desired_current = std::clamp(desired_torque * inv_torque_per_amp_, -motor_.max_current, motor_.max_current);
  const double back_emf = back_emf_per_joint_rad_s_ * motor_state.rad_s;
  const double voltage = std::clamp(motor_.ohm * desired_current + back_emf, -motor_.max_voltage, motor_.max_voltage);
  const double steady_state_current = (voltage - back_emf) * inv_ohm_;

  current_ = steady_state_current + (current_ - steady_state_current) * current_decay_;
  current_ = std::clamp(current_, -motor_.max_current, motor_.max_current);

  const double electromagnetic_torque = torque_per_amp_ * current_;
  const double friction_torque = motor_.viscous_friction * motor_state.rad_s;
  motor_state.torque = std::clamp(electromagnetic_torque - friction_torque, minimum_torque_, maximum_torque_);
}

void Servo::reset() {
  motor_state.rad = 0.0;
  motor_state.rad_s = 0.0;
  motor_state.torque = 0.0;
  current_ = 0.0;
}

} // namespace Actuator
