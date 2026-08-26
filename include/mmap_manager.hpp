#pragma once

#include "params.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

struct State;
struct Command;
class MST;

namespace bird_mmap {

inline constexpr std::uint32_t LOG_HZ = 250;
inline constexpr std::uint32_t LOG_SECONDS = 3;
inline constexpr std::uint32_t CAPACITY = LOG_HZ * LOG_SECONDS;
inline constexpr std::size_t NUM_JOINTS = param::NUM_JOINTS;
inline constexpr std::size_t NUM_SEGMENTS = 8;
inline constexpr std::size_t NUM_STRIPS = 2*(param::NH + param::NR + param::NM + param::NT);
inline constexpr std::size_t LOG_DECIMATION = static_cast<std::size_t>(1.0 / (param::SIM_DT_SEC * static_cast<double>(LOG_HZ)) + 0.5);

static_assert(LOG_DECIMATION > 0, "Invalid mmap logging decimation.");
static_assert(LOG_DECIMATION * LOG_HZ * std::chrono::duration_cast<std::chrono::microseconds>(param::SIM_DT_US).count() == 1000000, "LOG_HZ must divide the simulation rate exactly.");

struct SampleData {
  double time = 0.0;
  std::uint64_t step = 0;
  std::uint64_t reset_epoch = 0;

  float state_pos[3]{};
  float state_vel[3]{};
  float state_R[9]{};
  float state_w[3]{};
  float cmd_pos[3]{};
  float cmd_vel[3]{};
  float cmd_R[9]{};
  float cmd_w[3]{};
  float cmd_theta_t = 0.0f;

  float joint_theta[NUM_JOINTS]{};
  float joint_theta_dot[NUM_JOINTS]{};
  float joint_theta_ddot[NUM_JOINTS]{};
  float joint_theta_cmd[NUM_JOINTS]{};
  float servo_torque[NUM_JOINTS]{};
  float damping_torque[NUM_JOINTS]{};

  float segment_pos[NUM_SEGMENTS][3]{};
  float segment_force[NUM_SEGMENTS][3]{};
  float segment_torque[NUM_SEGMENTS][3]{};
  float body_elipsoid_pos[3]{};
  float body_elipsoid_force[3]{};
  float body_elipsoid_torque[3]{};

  float strip_alpha[NUM_STRIPS]{};
  float strip_alpha_dot[NUM_STRIPS]{};
  float strip_speed[NUM_STRIPS]{};
  float strip_Re[NUM_STRIPS]{};
  float strip_Cd[NUM_STRIPS]{};
  float strip_Cl_lut[NUM_STRIPS]{};
  float strip_Cl_dynamic[NUM_STRIPS]{};
  float strip_Cl_wagner[NUM_STRIPS]{};
  float strip_Cm[NUM_STRIPS]{};
  float strip_wagner_input[NUM_STRIPS]{};
  float strip_wagner_z1[NUM_STRIPS]{};
  float strip_wagner_z2[NUM_STRIPS]{};
  float strip_wagner_output[NUM_STRIPS]{};
  float strip_X_eq[NUM_STRIPS]{};
  float strip_X[NUM_STRIPS]{};
  float strip_X_target[NUM_STRIPS]{};
  float strip_tau1[NUM_STRIPS]{};
  float strip_tau2[NUM_STRIPS]{};
  float strip_stall_active[NUM_STRIPS]{};

  float strip_lut_force[NUM_STRIPS][3]{};
  float strip_dynamic_force[NUM_STRIPS][3]{};
  float strip_wagner_force[NUM_STRIPS][3]{};
  float strip_added_bias_force[NUM_STRIPS][3]{};
  float strip_added_full_force[NUM_STRIPS][3]{};
  float strip_lut_moment[NUM_STRIPS][3]{};
  float strip_added_bias_moment[NUM_STRIPS][3]{};
  float strip_added_full_moment[NUM_STRIPS][3]{};
};

struct alignas(8) MMapHeader {
  char magic[8]{};
  std::uint32_t version = 0;
  std::uint32_t header_size = 0;
  std::uint32_t descriptor_size = 0;
  std::uint32_t descriptor_count = 0;
  std::uint32_t sample_size = 0;
  std::uint32_t slot_size = 0;
  std::uint32_t capacity = 0;
  std::uint32_t log_hz = 0;
  std::uint32_t nh = 0;
  std::uint32_t nr = 0;
  std::uint32_t nm = 0;
  std::uint32_t reserved0 = 0;
  std::uint32_t nt = 0;
  std::uint64_t write_count = 0;
  std::uint64_t start_time_ns = 0;
  std::uint64_t schema_hash = 0;
  std::uint64_t session_id = 0;
  std::uint64_t sim_dt_ns = 0;
  std::uint64_t log_dt_ns = 0;
  std::uint64_t data_offset = 0;
  std::uint64_t reserved1 = 0;
};

struct ChannelDescriptor {
  char name[40]{};
  char unit[16]{};
  char frame[16]{};
  std::uint32_t offset = 0;
  std::uint16_t rows = 0;
  std::uint16_t cols = 0;
  std::uint8_t dtype = 0;
  std::uint8_t reserved[15]{};
};

struct alignas(8) Slot {
  std::uint64_t seq = 0;
  SampleData data{};
};

static_assert(sizeof(MMapHeader) == 128, "MMapHeader ABI changed.");
static_assert(offsetof(MMapHeader, reserved0) == 52 && offsetof(MMapHeader, nt) == 56, "MMapHeader strip-count layout changed.");
static_assert(offsetof(MMapHeader, write_count) == 64, "MMapHeader write_count offset changed.");
static_assert(offsetof(ChannelDescriptor, offset) == 72 && offsetof(ChannelDescriptor, dtype) == 80, "ChannelDescriptor field layout changed.");
static_assert(sizeof(ChannelDescriptor) == 96, "ChannelDescriptor ABI changed.");
static_assert(offsetof(ChannelDescriptor, offset) == 72 && offsetof(ChannelDescriptor, dtype) == 80, "ChannelDescriptor field layout changed.");
static_assert(std::is_standard_layout<SampleData>::value, "SampleData must have a stable standard layout.");
static_assert(offsetof(SampleData, time) == 0 && offsetof(SampleData, step) == 8 && offsetof(SampleData, reset_epoch) == 16 && offsetof(SampleData, state_pos) == 24, "SampleData prefix layout changed.");
static_assert(offsetof(Slot, data) == 8, "Slot data offset changed.");
static_assert(alignof(Slot) == 8 && sizeof(Slot) % 8 == 0, "Slot must be 8-byte aligned.");

class MMapLogger {
public:
  explicit MMapLogger(const std::string& path = "/tmp/bird_sim.mmap");
  ~MMapLogger();

  MMapLogger(const MMapLogger&) = delete;
  MMapLogger& operator=(const MMapLogger&) = delete;

  void open();
  void close();
  void push(double time, std::uint64_t step, std::uint64_t reset_epoch, const State& s, const Command& cmd, const MST& mst, const std::array<double, NUM_JOINTS>& servo_torque, const std::array<double, NUM_JOINTS>& damping_torque);

private:
  std::string path_;
  int fd_ = -1;
  std::size_t map_size_ = 0;
  std::uint8_t* base_ = nullptr;
  MMapHeader* header_ = nullptr;
  Slot* slots_ = nullptr;
};

} // namespace bird_mmap
