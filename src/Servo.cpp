#include "Servo.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Actuator {

Servo::Servo(const mjModel* model, const std::string& actuator_name, MotorParameters motor_parameters)
  : motor_(motor_parameters), current_decay_(std::exp(-motor_.ohm * motor_.dt / motor_.h)) {
    
  actuator_id_ = static_cast<std::int32_t>(mj_name2id(model, mjOBJ_ACTUATOR, actuator_name.c_str()));
  if (actuator_id_ < 0) {throw std::runtime_error("Actuator not found: " + actuator_name);}

  joint_id_ = static_cast<std::int32_t>(model->actuator_trnid[2 * actuator_id_]);
  qpos_address_ = static_cast<std::int32_t>(model->jnt_qposadr[joint_id_]);
  qvel_address_ = static_cast<std::int32_t>(model->jnt_dofadr[joint_id_]);

  if (model->actuator_ctrllimited[actuator_id_]) {
    ctrl_min_ = static_cast<double>(model->actuator_ctrlrange[2 * actuator_id_]);
    ctrl_max_ = static_cast<double>(model->actuator_ctrlrange[2 * actuator_id_ + 1]);
  }
  else {
    ctrl_min_ = -motor_.max_torque;
    ctrl_max_ = motor_.max_torque;
  }
}

void Servo::step(const mjData& data) {
  if (static_cast<double>(data.time) < previous_time_) {reset();}
  previous_time_ = static_cast<double>(data.time);

  motor_state.rad = static_cast<double>(data.qpos[qpos_address_]);
  motor_state.rad_s = static_cast<double>(data.qvel[qvel_address_]);

  const double position_error = desired_rad - motor_state.rad;

  const double desired_torque = std::clamp(
    motor_.kP * position_error - motor_.kD * motor_state.rad_s,
    -motor_.max_torque, motor_.max_torque
  );

  const double desired_current = std::clamp(desired_torque / (motor_.Kt * motor_.reduction_ratio * motor_.efficiency ), -motor_.max_current, motor_.max_current);
  const double motor_rad_s = motor_.reduction_ratio * motor_state.rad_s;
  const double back_emf = motor_.Ke * motor_rad_s;

  const double voltage = std::clamp(
    motor_.ohm * desired_current + back_emf,
    -motor_.max_voltage, motor_.max_voltage
  );

  const double steady_state_current = (voltage - back_emf) / motor_.ohm;

  current_ = steady_state_current + (current_ - steady_state_current) * current_decay_;
  current_ = std::clamp(current_, -motor_.max_current, motor_.max_current);

  const double electromagnetic_torque = motor_.Kt * current_ * motor_.reduction_ratio * motor_.efficiency;

  const double minimum_torque = std::max(ctrl_min_, -motor_.max_torque);
  const double maximum_torque = std::min(ctrl_max_, motor_.max_torque);
  const double friction_torque = motor_.viscous_friction * motor_state.rad_s;

  motor_state.torque = std::clamp(electromagnetic_torque - friction_torque, minimum_torque, maximum_torque);
}

void Servo::reset() {
  motor_state.rad = 0.0;
  motor_state.rad_s = 0.0;
  motor_state.torque = 0.0;

  current_ = 0.0;
  previous_time_ = 0.0;
}

std::int32_t Servo::actuatorId() const noexcept {
  return actuator_id_;
}

} // namespace Actuator