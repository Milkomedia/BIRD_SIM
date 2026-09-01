#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

class ELRS {
public:
  static constexpr const char* DEVICE = "/dev/cu.usbserial-DU0E4O7H";
  static constexpr unsigned long BAUD_RATE = 420000UL;
  static constexpr std::size_t CHANNEL_COUNT = 16U;

  using Channels = std::array<std::uint16_t, CHANNEL_COUNT>;

  ELRS() = default;
  ~ELRS();

  ELRS(const ELRS&) = delete;
  ELRS& operator=(const ELRS&) = delete;

  bool begin();
  void close();

  // Reads all currently available bytes.
  // Returns true when at least one new valid RC channel frame was decoded.
  bool update(Channels& channels);

  [[nodiscard]] bool is_open() const noexcept;
  [[nodiscard]] bool is_connected() const noexcept;
  [[nodiscard]] const std::string& last_error() const noexcept;

  [[nodiscard]] std::uint64_t valid_frames() const noexcept;
  [[nodiscard]] std::uint64_t crc_errors() const noexcept;
  [[nodiscard]] std::uint64_t length_errors() const noexcept;
  [[nodiscard]] std::uint64_t dropped_bytes() const noexcept;

private:
  static constexpr std::uint8_t CRSF_ADDRESS_FLIGHT_CONTROLLER = 0xC8U;
  static constexpr std::uint8_t CRSF_FRAMETYPE_RC_CHANNELS_PACKED = 0x16U;
  static constexpr std::uint8_t CRSF_MIN_LENGTH = 2U;
  static constexpr std::uint8_t CRSF_MAX_LENGTH = 62U;
  static constexpr std::uint8_t CRSF_RC_PAYLOAD_SIZE = 22U;
  static constexpr std::size_t STREAM_BUFFER_SIZE = 128U;
  static constexpr auto CONNECTION_TIMEOUT = std::chrono::seconds(1);

  int fd_ = -1;
  std::array<std::uint8_t, STREAM_BUFFER_SIZE> stream_{};
  std::size_t stream_size_ = 0U;
  Channels channels_{};

  bool has_valid_frame_ = false;
  std::chrono::steady_clock::time_point last_valid_frame_time_{};
  std::string last_error_{};

  std::uint64_t valid_frames_ = 0U;
  std::uint64_t crc_errors_ = 0U;
  std::uint64_t length_errors_ = 0U;
  std::uint64_t dropped_bytes_ = 0U;

  bool configure_uart();
  bool parse_stream();
  void discard_prefix(std::size_t count);

  static std::uint8_t crc8_d5(const std::uint8_t* data, std::size_t length);
  static void unpack_rc_channels(const std::uint8_t* payload, Channels& channels);
};

/*
*ch[0]  : yaw   [172, 1810] left,right
*ch[1]  : pitch [172, 1810] down, up
*ch[2]  : z     [172, 1810] down, up
*ch[3]  : roll  [172, 1810] left,right
 ch[4]  : SF    [172, 1810] down, up
 ch[5]  : SA    [172, 992, 1810] up, mid, down
 ch[6]  : SB    [172, 992, 1810] up, mid, down
 ch[7]  : SC    [172, 992, 1810] up, mid, down
 ch[8]  : SD    [172, 992, 1810] up, mid, down
 ch[9]  : SE    [172, 992, 1810] down, mid, up
*ch[10] : S1    [172, 1810] ccw, cw
*ch[11] : S2    [172, 1810] ccw, cw
*ch[12] : SG    [172, 992, 1810] down, mid, up
*ch[13] : SH    [172, 1810] down, up
 ch[14] : LS    [172, 1810] down, up
 ch[15] : RS    [172, 1810] down, up
*/