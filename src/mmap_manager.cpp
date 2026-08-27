#include "mmap_manager.hpp"

#include "MST.hpp"
#include "utils.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#if defined(__linux__) || defined(__APPLE__)
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#else
  #error "mmap_manager supports POSIX systems only."
#endif

namespace bird_mmap {
namespace {

constexpr char MAGIC[8] = {'B', 'I', 'R', 'D', 'L', 'O', 'G', '1'};
constexpr std::uint32_t VERSION = 2;
constexpr std::uint8_t DTYPE_FLOAT32 = 1;

static_assert(NUM_SEGMENTS+1 == MST::NUM_AERO_LOADS, "MST and mmap aerodynamic load counts differ.");
static_assert(NUM_STRIPS == MST::NUM_STRIPS, "MST and mmap strip counts differ.");

#define BIRD_CHANNELS(X) \
  X("state.pos",                    "m",       "world NED", offsetof(SampleData, state_pos),                  3, 1) \
  X("state.vel",                    "m/s",     "world NED", offsetof(SampleData, state_vel),                  3, 1) \
  X("state.R",                      "1",       "FRD->NED",  offsetof(SampleData, state_R),                    3, 3) \
  X("state.w",                      "rad/s",   "body FRD",  offsetof(SampleData, state_w),                    3, 1) \
  X("cmd.pos",                      "m",       "world NED", offsetof(SampleData, cmd_pos),                    3, 1) \
  X("cmd.vel",                      "m/s",     "world NED", offsetof(SampleData, cmd_vel),                    3, 1) \
  X("cmd.R",                        "1",       "FRD->NED",  offsetof(SampleData, cmd_R),                      3, 3) \
  X("cmd.w",                        "rad/s",   "body FRD",  offsetof(SampleData, cmd_w),                      3, 1) \
  X("cmd.theta_t",                  "rad",     "tail",      offsetof(SampleData, cmd_theta_t),                1, 1) \
  X("joint.theta",                  "rad",     "joint",     offsetof(SampleData, joint_theta),                NUM_JOINTS, 1) \
  X("joint.theta_dot",              "rad/s",   "joint",     offsetof(SampleData, joint_theta_dot),            NUM_JOINTS, 1) \
  X("joint.theta_ddot",             "rad/s2",  "joint",     offsetof(SampleData, joint_theta_ddot),           NUM_JOINTS, 1) \
  X("joint.theta_cmd",              "rad",     "joint",     offsetof(SampleData, joint_theta_cmd),            NUM_JOINTS, 1) \
  X("servo.torque",                 "N.m",     "joint",     offsetof(SampleData, servo_torque),               NUM_JOINTS, 1) \
  X("joint.damping_torque",         "N.m",     "joint",     offsetof(SampleData, damping_torque),             NUM_JOINTS, 1) \
  X("segment.pos",                  "m",       "body FRD",  offsetof(SampleData, segment_pos),                NUM_SEGMENTS, 3) \
  X("segment.force",                "N",       "body FRD",  offsetof(SampleData, segment_force),              NUM_SEGMENTS, 3) \
  X("segment.torque",               "N.m",     "body FRD",  offsetof(SampleData, segment_torque),             NUM_SEGMENTS, 3) \
  X("body.ellipsoid_pos",           "m",       "body FRD",  offsetof(SampleData, body_elipsoid_pos),           3, 1) \
  X("body.ellipsoid_force",         "N",       "body FRD",  offsetof(SampleData, body_elipsoid_force),         3, 1) \
  X("body.ellipsoid_torque",        "N.m",     "body FRD",  offsetof(SampleData, body_elipsoid_torque),        3, 1) \
  X("strip.alpha",                  "rad",     "strip",     offsetof(SampleData, strip_alpha),                 NUM_STRIPS, 1) \
  X("strip.alpha_dot",              "rad/s",   "strip",     offsetof(SampleData, strip_alpha_dot),             NUM_STRIPS, 1) \
  X("strip.speed",                  "m/s",     "strip",     offsetof(SampleData, strip_speed),                 NUM_STRIPS, 1) \
  X("tail.wake_delta_speed",        "m/s",     "tail strip", offsetof(SampleData, tail_wake_delta_speed),      NUM_TAIL_STRIPS, 1) \
  X("strip.Re",                     "1",       "strip",     offsetof(SampleData, strip_Re),                    NUM_STRIPS, 1) \
  X("strip.Cd",                     "1",       "strip",     offsetof(SampleData, strip_Cd),                    NUM_STRIPS, 1) \
  X("strip.Cl_lut",                 "1",       "strip",     offsetof(SampleData, strip_Cl_lut),                NUM_STRIPS, 1) \
  X("strip.Cl_dynamic",             "1",       "strip",     offsetof(SampleData, strip_Cl_dynamic),            NUM_STRIPS, 1) \
  X("strip.Cl_wagner",              "1",       "strip",     offsetof(SampleData, strip_Cl_wagner),             NUM_STRIPS, 1) \
  X("strip.Cm",                     "1",       "strip",     offsetof(SampleData, strip_Cm),                    NUM_STRIPS, 1) \
  X("strip.wagner_input",           "m/s",     "strip",     offsetof(SampleData, strip_wagner_input),          NUM_STRIPS, 1) \
  X("strip.wagner_z1",              "m/s",     "strip",     offsetof(SampleData, strip_wagner_z1),             NUM_STRIPS, 1) \
  X("strip.wagner_z2",              "m/s",     "strip",     offsetof(SampleData, strip_wagner_z2),             NUM_STRIPS, 1) \
  X("strip.wagner_output",          "m/s",     "strip",     offsetof(SampleData, strip_wagner_output),         NUM_STRIPS, 1) \
  X("strip.X_eq",                   "1",       "strip",     offsetof(SampleData, strip_X_eq),                  NUM_STRIPS, 1) \
  X("strip.X",                      "1",       "strip",     offsetof(SampleData, strip_X),                     NUM_STRIPS, 1) \
  X("strip.X_target",               "1",       "strip",     offsetof(SampleData, strip_X_target),              NUM_STRIPS, 1) \
  X("strip.tau1",                   "s",       "strip",     offsetof(SampleData, strip_tau1),                  NUM_STRIPS, 1) \
  X("strip.tau2",                   "s",       "strip",     offsetof(SampleData, strip_tau2),                  NUM_STRIPS, 1) \
  X("strip.stall_active",           "bool",    "strip",     offsetof(SampleData, strip_stall_active),          NUM_STRIPS, 1) \
  X("strip.lut_force",              "N",       "body FRD",  offsetof(SampleData, strip_lut_force),             NUM_STRIPS, 3) \
  X("strip.dynamic_force",          "N",       "body FRD",  offsetof(SampleData, strip_dynamic_force),         NUM_STRIPS, 3) \
  X("strip.wagner_force",           "N",       "body FRD",  offsetof(SampleData, strip_wagner_force),          NUM_STRIPS, 3) \
  X("strip.added_bias_force",       "N",       "body FRD",  offsetof(SampleData, strip_added_bias_force),      NUM_STRIPS, 3) \
  X("strip.added_full_force",       "N",       "body FRD",  offsetof(SampleData, strip_added_full_force),      NUM_STRIPS, 3) \
  X("strip.lut_moment",             "N.m",     "body FRD",  offsetof(SampleData, strip_lut_moment),            NUM_STRIPS, 3) \
  X("strip.added_bias_moment",      "N.m",     "body FRD",  offsetof(SampleData, strip_added_bias_moment),     NUM_STRIPS, 3) \
  X("strip.added_full_moment",      "N.m",     "body FRD",  offsetof(SampleData, strip_added_full_moment),     NUM_STRIPS, 3)

#define COUNT_CHANNEL(...) + 1
constexpr std::size_t NUM_CHANNELS = 0 BIRD_CHANNELS(COUNT_CHANNEL);
#undef COUNT_CHANNEL

void copy_string(char* dst, const std::size_t size, const char* src) {
  std::strncpy(dst, src, size-1);
  dst[size-1] = '\0';
}

std::array<ChannelDescriptor, NUM_CHANNELS> make_descriptors() {
  std::array<ChannelDescriptor, NUM_CHANNELS> descriptors{};
  std::size_t index = 0;
#define ADD_CHANNEL(NAME, UNIT, FRAME, OFFSET, ROWS, COLS) \
  do { \
    ChannelDescriptor& descriptor = descriptors[index++]; \
    copy_string(descriptor.name, sizeof(descriptor.name), NAME); \
    copy_string(descriptor.unit, sizeof(descriptor.unit), UNIT); \
    copy_string(descriptor.frame, sizeof(descriptor.frame), FRAME); \
    descriptor.offset = static_cast<std::uint32_t>(OFFSET); \
    descriptor.rows = static_cast<std::uint16_t>(ROWS); \
    descriptor.cols = static_cast<std::uint16_t>(COLS); \
    descriptor.dtype = DTYPE_FLOAT32; \
  } while (false);
  BIRD_CHANNELS(ADD_CHANNEL)
#undef ADD_CHANNEL
  return descriptors;
}

std::uint64_t hash_descriptors(const std::array<ChannelDescriptor, NUM_CHANNELS>& descriptors) {
  constexpr std::uint64_t OFFSET = 1469598103934665603ull;
  constexpr std::uint64_t PRIME = 1099511628211ull;
  std::uint64_t hash = OFFSET;
  const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(descriptors.data());
  for (std::size_t i=0; i<sizeof(descriptors); ++i) {hash = (hash ^ bytes[i]) * PRIME;}
  hash = (hash ^ static_cast<std::uint64_t>(sizeof(SampleData))) * PRIME;
  return hash;
}

std::uint64_t atomic_load(const std::uint64_t* value) {return __atomic_load_n(value, __ATOMIC_ACQUIRE);}
void atomic_store(std::uint64_t* value, const std::uint64_t data) {__atomic_store_n(value, data, __ATOMIC_RELEASE);}

template <std::size_t N_DST, std::size_t N_SRC>
void copy_vector(float (&dst)[N_DST][3], const std::array<Eigen::Vector3d, N_SRC>& src) {
  for (std::size_t i=0; i<N_DST; ++i) {
    dst[i][0] = static_cast<float>(src[i].x());
    dst[i][1] = static_cast<float>(src[i].y());
    dst[i][2] = static_cast<float>(src[i].z());
  }
}

void copy_vector(float (&dst)[3], const Eigen::Vector3d& src) {
  dst[0] = static_cast<float>(src.x());
  dst[1] = static_cast<float>(src.y());
  dst[2] = static_cast<float>(src.z());
}

template <std::size_t N>
void copy_scalar(float (&dst)[N], const std::array<double, N>& src) {
  for (std::size_t i=0; i<N; ++i) {dst[i] = static_cast<float>(src[i]);}
}

} // namespace

MMapLogger::MMapLogger(const std::string& path) : path_(path) {}

MMapLogger::~MMapLogger() {close();}

void MMapLogger::open() {
  if (base_) {return;}

  const std::array<ChannelDescriptor, NUM_CHANNELS> descriptors = make_descriptors();
  const std::size_t descriptor_bytes = sizeof(descriptors);
  const std::size_t data_offset = (sizeof(MMapHeader) + descriptor_bytes + 7u) & ~std::size_t{7u};
  map_size_ = data_offset + static_cast<std::size_t>(CAPACITY)*sizeof(Slot);

  fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT, 0666);
  if (fd_ < 0) {throw std::runtime_error("bird mmap: open failed: " + path_);}
  if (::ftruncate(fd_, static_cast<off_t>(map_size_)) != 0) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("bird mmap: ftruncate failed");
  }

  void* mapped = ::mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (mapped == MAP_FAILED) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("bird mmap: mmap failed");
  }

  base_ = static_cast<std::uint8_t*>(mapped);
  std::memset(base_, 0, map_size_);
  header_ = reinterpret_cast<MMapHeader*>(base_);
  slots_ = reinterpret_cast<Slot*>(base_ + data_offset);

  std::memcpy(header_->magic, MAGIC, sizeof(MAGIC));
  header_->version = VERSION;
  header_->header_size = sizeof(MMapHeader);
  header_->descriptor_size = sizeof(ChannelDescriptor);
  header_->descriptor_count = static_cast<std::uint32_t>(NUM_CHANNELS);
  header_->sample_size = sizeof(SampleData);
  header_->slot_size = sizeof(Slot);
  header_->capacity = CAPACITY;
  header_->log_hz = LOG_HZ;
  header_->nh = param::NH;
  header_->nr = param::NR;
  header_->nm = param::NM;
  header_->nt = param::NT;
  header_->start_time_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
  header_->schema_hash = hash_descriptors(descriptors);
  header_->session_id = header_->start_time_ns;
  header_->sim_dt_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(param::SIM_DT_US).count());
  header_->log_dt_ns = 1000000000ull/LOG_HZ;
  header_->data_offset = data_offset;
  std::memcpy(base_ + sizeof(MMapHeader), descriptors.data(), descriptor_bytes);
  atomic_store(&header_->write_count, 0);
}

void MMapLogger::close() {
  if (!base_) {return;}
  ::msync(base_, map_size_, MS_ASYNC);
  ::munmap(base_, map_size_);
  ::close(fd_);
  base_ = nullptr;
  header_ = nullptr;
  slots_ = nullptr;
  fd_ = -1;
  map_size_ = 0;
}

void MMapLogger::push(double time, std::uint64_t step, std::uint64_t reset_epoch, const State& s, const Command& cmd, const MST& mst, const std::array<double, NUM_JOINTS>& servo_torque, const std::array<double, NUM_JOINTS>& damping_torque) {
  if (!base_) {open();}

  const std::uint64_t write_count = atomic_load(&header_->write_count);
  Slot& slot = slots_[write_count % CAPACITY];
  const std::uint64_t sequence = atomic_load(&slot.seq);
  atomic_store(&slot.seq, sequence + 1);

  SampleData& data = slot.data;
  data.time = time;
  data.step = step;
  data.reset_epoch = reset_epoch;

  copy_vector(data.state_pos, s.pos);
  copy_vector(data.state_vel, s.vel);
  copy_vector(data.state_w, s.w);
  copy_vector(data.cmd_pos, cmd.pos);
  copy_vector(data.cmd_vel, cmd.vel);
  copy_vector(data.cmd_w, cmd.w);
  data.cmd_theta_t = static_cast<float>(cmd.theta_t);
  for (std::size_t row=0; row<3; ++row) {
    for (std::size_t col=0; col<3; ++col) {
      data.state_R[3*row+col] = static_cast<float>(s.R(row, col));
      data.cmd_R[3*row+col] = static_cast<float>(cmd.R(row, col));
    }
  }

  for (std::size_t i=0; i<NUM_JOINTS; ++i) {
    data.joint_theta[i] = static_cast<float>(s.theta[i]);
    data.joint_theta_dot[i] = static_cast<float>(s.theta_dot[i]);
    data.joint_theta_ddot[i] = static_cast<float>(s.theta_ddot[i]);
    data.joint_theta_cmd[i] = static_cast<float>(cmd.theta[i]);
    data.servo_torque[i] = static_cast<float>(servo_torque[i]);
    data.damping_torque[i] = static_cast<float>(damping_torque[i]);
  }

  const std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS>& aero_pos = mst.positions();
  const std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS>& aero_force = mst.forces();
  const std::array<Eigen::Vector3d, MST::NUM_AERO_LOADS>& aero_torque = mst.torques();
  copy_vector(data.segment_pos, aero_pos);
  copy_vector(data.segment_force, aero_force);
  copy_vector(data.segment_torque, aero_torque);
  copy_vector(data.body_elipsoid_pos, aero_pos.back());
  copy_vector(data.body_elipsoid_force, aero_force.back());
  copy_vector(data.body_elipsoid_torque, aero_torque.back());

  const MST::AeroTelemetry& aero = mst.aero_telemetry();
  copy_scalar(data.strip_alpha, aero.alpha);
  copy_scalar(data.strip_alpha_dot, aero.alpha_dot);
  copy_scalar(data.strip_speed, aero.speed);
  copy_scalar(data.tail_wake_delta_speed, aero.tail_wake_delta_speed);
  copy_scalar(data.strip_Re, aero.Re);
  copy_scalar(data.strip_Cd, aero.Cd);
  copy_scalar(data.strip_Cl_lut, aero.Cl_lut);
  copy_scalar(data.strip_Cl_dynamic, aero.Cl_dynamic);
  copy_scalar(data.strip_Cl_wagner, aero.Cl_wagner);
  copy_scalar(data.strip_Cm, aero.Cm);
  copy_scalar(data.strip_wagner_input, aero.wagner_input);
  copy_scalar(data.strip_wagner_z1, aero.wagner_z1);
  copy_scalar(data.strip_wagner_z2, aero.wagner_z2);
  copy_scalar(data.strip_wagner_output, aero.wagner_output);
  copy_scalar(data.strip_X_eq, aero.X_eq);
  copy_scalar(data.strip_X, aero.X);
  copy_scalar(data.strip_X_target, aero.X_target);
  copy_scalar(data.strip_tau1, aero.tau1);
  copy_scalar(data.strip_tau2, aero.tau2);
  copy_scalar(data.strip_stall_active, aero.stall_active);
  copy_vector(data.strip_lut_force, aero.lut_force);
  copy_vector(data.strip_dynamic_force, aero.dynamic_force);
  copy_vector(data.strip_wagner_force, aero.wagner_force);
  copy_vector(data.strip_added_bias_force, aero.added_bias_force);
  copy_vector(data.strip_added_full_force, aero.added_full_force);
  copy_vector(data.strip_lut_moment, aero.lut_moment);
  copy_vector(data.strip_added_bias_moment, aero.added_bias_moment);
  copy_vector(data.strip_added_full_moment, aero.added_full_moment);

  atomic_store(&slot.seq, sequence + 2);
  atomic_store(&header_->write_count, write_count + 1);
}

} // namespace bird_mmap
