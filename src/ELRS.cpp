#include "ELRS.hpp"

#include <IOKit/serial/ioss.h>

#include <sys/ioctl.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

ELRS::~ELRS() {
  close();
}

bool ELRS::begin() {
  close();
  last_error_.clear();

  fd_ = ::open(DEVICE, O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) {
    last_error_ = std::string("open failed: ") + std::strerror(errno);
    return false;
  }

  if (!configure_uart()) {
    close();
    return false;
  }

  stream_size_ = 0U;
  channels_.fill(0U);
  has_valid_frame_ = false;

  if (::tcflush(fd_, TCIOFLUSH) != 0) {
    last_error_ = std::string("tcflush failed: ") + std::strerror(errno);
    close();
    return false;
  }

  return true;
}

void ELRS::close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }

  stream_size_ = 0U;
  has_valid_frame_ = false;
}

bool ELRS::update(Channels& channels) {
  if (fd_ < 0) {last_error_ = "serial device is not open"; return false;}

  bool rc_frame_updated = false;
  std::array<std::uint8_t, 512U> read_buffer;

  for (;;) {
    const ssize_t received = ::read(fd_, read_buffer.data(), read_buffer.size());

    if (received > 0) {
      std::size_t offset = 0U;
      const std::size_t received_size = static_cast<std::size_t>(received);
      while (offset < received_size) {
        if (stream_size_ == stream_.size()) {
          rc_frame_updated = parse_stream() || rc_frame_updated;
          if (stream_size_ == stream_.size()) {
            discard_prefix(1U);
            ++dropped_bytes_;
          }
        }

        const std::size_t copy_size = std::min(received_size-offset, stream_.size()-stream_size_);
        std::memcpy(stream_.data()+stream_size_, read_buffer.data()+offset, copy_size);
        stream_size_ += copy_size;
        offset += copy_size;
        rc_frame_updated = parse_stream() || rc_frame_updated;
      }
      continue;
    }

    if (received == 0) {break;}
    if (errno == EINTR) {continue;}
    if (errno == EAGAIN || errno == EWOULDBLOCK) {break;}

    last_error_ = std::string("read failed: ") + std::strerror(errno);
    close();
    return false;
  }

  if (rc_frame_updated) {channels = channels_;}

  return rc_frame_updated;
}

bool ELRS::is_open() const noexcept {return fd_ >= 0;}

bool ELRS::is_connected() const noexcept {
  if (!has_valid_frame_) {return false;}
  return (std::chrono::steady_clock::now() - last_valid_frame_time_) < CONNECTION_TIMEOUT;
}

const std::string& ELRS::last_error() const noexcept {return last_error_;}
std::uint64_t ELRS::valid_frames() const noexcept {return valid_frames_;}
std::uint64_t ELRS::crc_errors() const noexcept {return crc_errors_;}
std::uint64_t ELRS::length_errors() const noexcept {return length_errors_;}
std::uint64_t ELRS::dropped_bytes() const noexcept {return dropped_bytes_;}

bool ELRS::configure_uart() {
  termios tty{};

  if (::tcgetattr(fd_, &tty) != 0) {
    last_error_ = std::string("tcgetattr failed: ") + std::strerror(errno);
    return false;
  }

  ::cfmakeraw(&tty);

  tty.c_cflag &= static_cast<tcflag_t>(~(CSIZE | PARENB | CSTOPB | CRTSCTS));
  tty.c_cflag |= static_cast<tcflag_t>(CS8 | CLOCAL | CREAD);
  tty.c_iflag &= static_cast<tcflag_t>(~(IXON | IXOFF | IXANY));
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 0;

  // A standard speed must be installed before IOSSIOSPEED is applied.
  if (::cfsetispeed(&tty, B9600) != 0 || ::cfsetospeed(&tty, B9600) != 0) {
    last_error_ = std::string("cfsetispeed/cfsetospeed failed: ") + std::strerror(errno);
    return false;
  }

  if (::tcsetattr(fd_, TCSANOW, &tty) != 0) {
    last_error_ = std::string("tcsetattr failed: ") + std::strerror(errno);
    return false;
  }

  speed_t speed = static_cast<speed_t>(BAUD_RATE);
  if (::ioctl(fd_, IOSSIOSPEED, &speed) != 0) {
    last_error_ = std::string("IOSSIOSPEED failed: ") + std::strerror(errno);
    return false;
  }

  return true;
}

bool ELRS::parse_stream() {
  bool rc_frame_updated = false;

  for (;;) {
    std::size_t sync_position = 0U;
    while (sync_position < stream_size_ && stream_[sync_position] != CRSF_ADDRESS_FLIGHT_CONTROLLER) {++sync_position;}

    if (sync_position == stream_size_) {
      dropped_bytes_ += stream_size_;
      stream_size_ = 0U;
      return rc_frame_updated;
    }

    if (sync_position > 0U) {
      discard_prefix(sync_position);
      dropped_bytes_ += sync_position;
    }

    if (stream_size_ < 2U) {
      return rc_frame_updated;
    }

    const std::uint8_t length = stream_[1];
    if (length < CRSF_MIN_LENGTH || length > CRSF_MAX_LENGTH) {
      ++length_errors_;
      ++dropped_bytes_;
      discard_prefix(1U);
      continue;
    }

    const std::size_t frame_size = static_cast<std::size_t>(length) + 2U;
    if (stream_size_ < frame_size) {
      return rc_frame_updated;
    }

    const std::uint8_t received_crc = stream_[frame_size - 1U];
    const std::uint8_t computed_crc = crc8_d5(&stream_[2], static_cast<std::size_t>(length) - 1U);

    if (computed_crc != received_crc) {
      ++crc_errors_;
      ++dropped_bytes_;

      // Discard only the current sync candidate. This allows recovery
      // from a valid frame that begins inside corrupted input.
      discard_prefix(1U);
      continue;
    }

    const std::uint8_t frame_type = stream_[2];
    const std::size_t payload_size = static_cast<std::size_t>(length) - 2U;

    if (frame_type == CRSF_FRAMETYPE_RC_CHANNELS_PACKED && payload_size == CRSF_RC_PAYLOAD_SIZE) {
      unpack_rc_channels(&stream_[3], channels_);
      ++valid_frames_;
      has_valid_frame_ = true;
      last_valid_frame_time_ = std::chrono::steady_clock::now();
      rc_frame_updated = true;
    }

    // A CRC-valid frame is consumed even if its type is not used.
    discard_prefix(frame_size);
  }
}

void ELRS::discard_prefix(const std::size_t count) {
  if (count >= stream_size_) {
    stream_size_ = 0U;
    return;
  }

  const std::size_t remaining = stream_size_ - count;
  std::memmove(stream_.data(), stream_.data() + count, remaining);
  stream_size_ = remaining;
}

std::uint8_t ELRS::crc8_d5(const std::uint8_t* data, std::size_t length) {
  std::uint8_t crc = 0U;

  while (length-- > 0U) {
    crc ^= *data++;
    for (std::uint8_t bit=0U; bit<8U; ++bit) {crc = (crc & 0x80U) != 0U ? static_cast<std::uint8_t>((crc << 1U) ^ 0xD5U) : static_cast<std::uint8_t>(crc << 1U);}
  }

  return crc;
}

void ELRS::unpack_rc_channels(const std::uint8_t* payload, Channels& channels) {
  std::uint32_t bit_buffer = 0U;
  std::uint8_t bits_available = 0U;
  std::size_t byte_index = 0U;

  for (std::size_t channel=0U; channel<channels.size(); ++channel) {
    while (bits_available < 11U) {
      bit_buffer |= static_cast<std::uint32_t>(payload[byte_index++]) << bits_available;
      bits_available = static_cast<std::uint8_t>(bits_available + 8U);
    }

    channels[channel] = static_cast<std::uint16_t>(bit_buffer & 0x07FFU);
    bit_buffer >>= 11U;
    bits_available = static_cast<std::uint8_t>(bits_available - 11U);
  }
}