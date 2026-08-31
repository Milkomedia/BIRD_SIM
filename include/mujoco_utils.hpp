#pragma once

#include "params.hpp"
#include <Eigen/Core>
#include <mujoco/mujoco.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>

namespace mj_utils {

  inline mjModel* g_model = nullptr;
  inline mjData* g_data = nullptr;

  inline mjvCamera g_camera;
  inline std::array<mjtNum, 3> g_camera_pan_offset{};
  inline mjvOption g_option;
  inline mjvPerturb g_perturb;
  inline mjvScene g_scene;
  inline mjrContext g_context;
  inline mjUI g_ui;
  inline mjuiState g_ui_state;

  inline std::array<int, param::NUM_JOINTS> g_actuator_ids{};
  inline std::array<int, param::NUM_JOINTS> g_joint_qpos_adrs{};
  inline std::array<mjtNum, param::NUM_JOINTS> g_command_theta{};
  inline mjtNum g_command_theta_t = param::INITIAL_THETA_T;
  inline mjtNum g_sim_speed = 1.0;
  inline mjtNum g_resume_sim_speed = 1.0;

  inline bool g_left_pressed = false;
  inline bool g_middle_pressed = false;
  inline bool g_right_pressed = false;
  inline bool g_paused = false;
  inline std::uint64_t g_reset_epoch = 0;
  inline double g_last_x = 0.0;
  inline double g_last_y = 0.0;

  inline constexpr double FRAME_AXIS_LENGTH = 0.04;    // [m]
  inline constexpr double FRAME_ARROW_WIDTH = 0.0025;  // [m]
  inline constexpr double AERO_FORCE_ARROW_SCALE = 0.2000; // [m/N]
  inline constexpr double AERO_FORCE_ARROW_WIDTH = 0.0035; // [m]
  inline constexpr double STRIP_FRAME_SCALE = 0.5;
  inline constexpr double STRIP_PLATE_THICKNESS = 0.001; // [m]
  inline constexpr double COM_MARKER_RADIUS = 0.005; // [m]

  enum ArrowQuantity : int {
    ARROW_V = 0,
    ARROW_A,
    ARROW_W,
    ARROW_WDOT
  };

  inline int g_arrow_quantity = ARROW_V;

  enum ArrowView : int {
    ARROW_VIEW_NONE = 0,
    ARROW_VIEW_WING,
    ARROW_VIEW_TAIL,
    ARROW_VIEW_BOTH
  };

  inline int g_arrow_view = ARROW_VIEW_NONE;

  enum TrajectoryView : int {
    TRAJECTORY_VIEW_OFF = 0,
    TRAJECTORY_VIEW_ON
  };

  inline int g_trajectory_view = TRAJECTORY_VIEW_ON;

  inline constexpr double V_ARROW_SCALE = 0.25;        // [s]
  inline constexpr double A_ARROW_SCALE = 0.01;        // [s^2]
  inline constexpr double W_ARROW_SCALE = 0.02;        // [m.s/rad]
  inline constexpr double WDOT_ARROW_SCALE = 0.001;    // [m.s^2/rad]
  inline constexpr double STATE_ARROW_WIDTH = 0.0015;  // [m]
  inline constexpr double MANUS_TRAIL_WIDTH = 0.0015;  // [m]
  inline constexpr std::size_t MANUS_TRAIL_SAMPLE_MULTIPLIER = 3;

  inline constexpr std::array<float, 4> V_ARROW_COLOR    = {0.35f, 0.80f, 1.00f, 0.50f};
  inline constexpr std::array<float, 4> A_ARROW_COLOR    = {1.00f, 0.40f, 0.10f, 0.50f};
  inline constexpr std::array<float, 4> W_ARROW_COLOR    = {0.85f, 0.20f, 0.95f, 0.50f};
  inline constexpr std::array<float, 4> WDOT_ARROW_COLOR = {1.00f, 0.85f, 0.10f, 0.50f};
  inline constexpr std::array<float, 4> AERO_FORCE_ARROW_COLOR = {1.00f, 0.00f, 0.00f, 0.90f};
  inline constexpr std::array<float, 4> MANUS_FIRST_TRAIL_COLOR = {1.00f, 0.65f, 0.30f, 0.35f};
  inline constexpr std::array<float, 4> MANUS_LAST_TRAIL_COLOR = {0.45f, 0.85f, 1.00f, 0.35f};
  inline constexpr std::array<float, 4> COM_MARKER_COLOR = {1.00f, 1.00f, 1.00f, 1.00f};

  class ManusTrajectory {
  public:
    void set_enabled(const bool enabled) {
      enabled_.store(enabled, std::memory_order_release);
      if (!enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& trail : trails_) {trail.clear();}
      }
    }

    void reset() {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto& trail : trails_) {trail.clear();}
      next_sample_time_ = -1.0;
      previous_time_ = -1.0;
    }

    void sample_if_due(const double sim_time, const std::array<Eigen::Vector3d, 2*param::NM>& body_manus_positions, const Eigen::Matrix3d& world_R_body, const Eigen::Vector3d& world_p_body) {
      if (!std::isfinite(sim_time)) {
        reset();
        return;
      }
      if (!enabled_.load(std::memory_order_acquire)) {
        next_sample_time_ = -1.0;
        previous_time_ = sim_time;
        return;
      }
      if (previous_time_ >= 0.0 && sim_time < previous_time_) {reset();}
      previous_time_ = sim_time;
      if (next_sample_time_ >= 0.0 && sim_time+1.e-12 < next_sample_time_) {return;}
      if (!world_R_body.allFinite() || !world_p_body.allFinite()) {return;}

      std::array<Eigen::Vector3d, NUM_TRAILS> sampled_position{};
      std::array<bool, NUM_TRAILS> sampled_position_valid{};
      for (std::size_t wing=0; wing<2; ++wing) {
        for (std::size_t point=0; point<MANUS_FRAME_INDICES.size(); ++point) {
          const std::size_t trail_idx = wing*TRAILS_PER_WING+point;
          const Eigen::Vector3d& body_position = body_manus_positions[wing*param::NM+MANUS_FRAME_INDICES[point]];
          sampled_position_valid[trail_idx] = body_position.allFinite() && body_position.squaredNorm() > 1.e-12;
          if (sampled_position_valid[trail_idx]) {sampled_position[trail_idx] = world_p_body+world_R_body*body_position;}
        }
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enabled_.load(std::memory_order_relaxed)) {return;}
        bool sample_added = false;
        const double oldest_stored_time = sim_time - STORED_DURATION;
        for (std::size_t trail_idx=0; trail_idx<NUM_TRAILS; ++trail_idx) {
          while (!trails_[trail_idx].empty() && trails_[trail_idx].front().time < oldest_stored_time) {trails_[trail_idx].pop_front();}
          if (sampled_position_valid[trail_idx]) {
            trails_[trail_idx].push_back({sim_time, sampled_position[trail_idx]});
            sample_added = true;
          }
        }
        if (!sample_added) {return;}
      }

      if (next_sample_time_ < 0.0) {next_sample_time_ = sim_time+SAMPLE_DT;}
      else {do {next_sample_time_ += SAMPLE_DT;} while (next_sample_time_ <= sim_time+1.e-12);}
    }

    void render(const double sim_time, const std::array<Eigen::Vector3d, 2*param::NM>& body_manus_positions, const Eigen::Matrix4d& world_T_body) {
      if (!enabled_.load(std::memory_order_acquire) || !std::isfinite(sim_time) || !world_T_body.allFinite()) {return;}

      std::array<std::deque<Sample>, NUM_TRAILS> visible_trails{};
      {
        std::lock_guard<std::mutex> lock(mutex_);
        visible_trails = trails_;
      }

      const double oldest_visible_time = sim_time-TRAIL_DURATION;
      for (auto& trail : visible_trails) {
        while (!trail.empty() && trail.front().time < oldest_visible_time) {trail.pop_front();}
        while (!trail.empty() && trail.back().time > sim_time+1.e-12) {trail.pop_back();}
      }

      const Eigen::Matrix3d world_R_body = world_T_body.block<3, 3>(0, 0);
      const Eigen::Vector3d world_p_body = world_T_body.block<3, 1>(0, 3);
      for (std::size_t wing=0; wing<2; ++wing) {
        for (std::size_t point=0; point<MANUS_FRAME_INDICES.size(); ++point) {
          const std::size_t trail_idx = wing*TRAILS_PER_WING+point;
          const std::deque<Sample>& trail = visible_trails[trail_idx];
          const std::array<float, 4>& color = point == 0 ? MANUS_FIRST_TRAIL_COLOR : MANUS_LAST_TRAIL_COLOR;
          for (std::size_t i=1; i<trail.size(); ++i) {append_segment(trail[i-1].position, trail[i].position, color);}
          const Eigen::Vector3d& body_position = body_manus_positions[wing*param::NM+MANUS_FRAME_INDICES[point]];
          if (!trail.empty() && body_position.allFinite() && body_position.squaredNorm() > 1.e-12) {append_segment(trail.back().position, world_p_body+world_R_body*body_position, color);}
        }
      }
    }

  private:
    struct Sample {
      double time;
      Eigen::Vector3d position;
    };

    static constexpr double TRAIL_DURATION = 0.5; // [s], simulation time
    static constexpr double RENDER_DT = std::chrono::duration<double>(param::RENDER_DT_US).count();
    static constexpr double SAMPLE_DT = RENDER_DT/static_cast<double>(MANUS_TRAIL_SAMPLE_MULTIPLIER);
    static constexpr double STORED_DURATION = TRAIL_DURATION+RENDER_DT;
    static constexpr std::array<std::size_t, 2> MANUS_FRAME_INDICES = {0, param::NM-1};
    static constexpr std::size_t TRAILS_PER_WING = MANUS_FRAME_INDICES.size();
    static constexpr std::size_t NUM_TRAILS = 2*TRAILS_PER_WING;

    static void append_segment(const Eigen::Vector3d& from_point, const Eigen::Vector3d& to_point, const std::array<float, 4>& color) {
      if (!from_point.allFinite() || !to_point.allFinite() || (to_point-from_point).squaredNorm() < 1.e-12 || g_scene.ngeom >= g_scene.maxgeom) {return;}

      const mjtNum from[3] = {static_cast<mjtNum>(from_point(0)), static_cast<mjtNum>(from_point(1)), static_cast<mjtNum>(from_point(2))};
      const mjtNum to[3] = {static_cast<mjtNum>(to_point(0)), static_cast<mjtNum>(to_point(1)), static_cast<mjtNum>(to_point(2))};

      mjvGeom& geom = g_scene.geoms[g_scene.ngeom];
      mjv_initGeom(&geom, mjGEOM_CAPSULE, nullptr, nullptr, nullptr, color.data());
      mjv_connector(&geom, mjGEOM_CAPSULE, static_cast<mjtNum>(MANUS_TRAIL_WIDTH), from, to);
      geom.category = mjCAT_DECOR;
      ++g_scene.ngeom;
    }

    std::array<std::deque<Sample>, NUM_TRAILS> trails_{};
    std::atomic<bool> enabled_{true};
    std::mutex mutex_;
    double next_sample_time_ = -1.0;
    double previous_time_ = -1.0;
  };

  inline ManusTrajectory g_manus_trajectory{};

  inline void layout_ui(GLFWwindow* window) {
    int framebuffer_width = 0;
    int framebuffer_height = 0;
    glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);

    g_ui_state.nrect = 2;
    g_ui_state.rect[0] = {0, 0, framebuffer_width, framebuffer_height};
    g_ui_state.rect[1] = {std::max(0, framebuffer_width - g_ui.width), 0, g_ui.width, framebuffer_height};
  }

  inline void update_ui_state(GLFWwindow* window) {
    g_ui_state.left = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    g_ui_state.right = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    g_ui_state.middle = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    g_ui_state.control = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
    g_ui_state.shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    g_ui_state.alt = glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;

    int window_width = 0;
    int window_height = 0;
    int framebuffer_width = 0;
    int framebuffer_height = 0;
    glfwGetWindowSize(window, &window_width, &window_height);
    glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
    if (window_width <= 0 || window_height <= 0) {return;}

    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos(window, &x, &y);
    x *= static_cast<double>(framebuffer_width) / window_width;
    y = framebuffer_height - y * static_cast<double>(framebuffer_height) / window_height;

    g_ui_state.dx = x - g_ui_state.x;
    g_ui_state.dy = y - g_ui_state.y;
    g_ui_state.x = x;
    g_ui_state.y = y;
    g_ui_state.mouserect = mjr_findRect(mju_round(x), mju_round(y), g_ui_state.nrect - 1, g_ui_state.rect + 1) + 1;
  }

  inline void sync_pause_from_sim_speed() {
    g_sim_speed = std::clamp(g_sim_speed, static_cast<mjtNum>(0.0), static_cast<mjtNum>(1.0));
    g_paused = g_sim_speed <= static_cast<mjtNum>(0.0);
    if (!g_paused) {g_resume_sim_speed = g_sim_speed;}
  }

  inline void toggle_pause() {
    if (g_paused) {
      g_sim_speed = std::clamp(g_resume_sim_speed, static_cast<mjtNum>(0.0), static_cast<mjtNum>(1.0));
      if (g_sim_speed <= static_cast<mjtNum>(0.0)) {g_sim_speed = static_cast<mjtNum>(1.0);}
      g_paused = false;
    }
    else {
      if (g_sim_speed > static_cast<mjtNum>(0.0)) {g_resume_sim_speed = g_sim_speed;}
      g_sim_speed = static_cast<mjtNum>(0.0);
      g_paused = true;
    }
  }

  inline bool process_ui_event() {
    const bool targets_ui =
      g_ui_state.dragrect == g_ui.rectid || (g_ui_state.dragrect == 0 && g_ui_state.mouserect == g_ui.rectid) || g_ui_state.type == mjEVENT_KEY;
    if (!targets_ui) {return false;}

    mjuiItem* changed = mjui_event(&g_ui, &g_ui_state, &g_context);
    if (changed && changed->pdata == &g_sim_speed) {sync_pause_from_sim_speed();}
    if (changed && changed->pdata == &g_trajectory_view) {g_manus_trajectory.set_enabled(g_trajectory_view == TRAJECTORY_VIEW_ON);}
    return changed != nullptr || g_ui_state.mouserect == g_ui.rectid || g_ui_state.dragrect == g_ui.rectid || (g_ui_state.type == mjEVENT_KEY && g_ui_state.key == 0);
  }

  inline void glfw_error_callback(int error, const char* description) {
    std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
  }

  inline void keyboard_callback(GLFWwindow* window, int key, int, int action, int) {
    if (action != GLFW_PRESS) {return;}

    if (key == GLFW_KEY_ESCAPE) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
      return;
    }
    else if (key == GLFW_KEY_SPACE) {
      toggle_pause();
      mjui_update(-1, -1, &g_ui, &g_ui_state, &g_context);
      return;
    }
    else if (key == GLFW_KEY_BACKSPACE) {
      ++g_reset_epoch;
      g_camera_pan_offset.fill(0.0);
      mjv_defaultPerturb(&g_perturb);
      return;
    }

    update_ui_state(window);
    g_ui_state.type = mjEVENT_KEY;
    g_ui_state.key = key;
    g_ui_state.keytime = glfwGetTime();
    if (process_ui_event()) {return;}
  }

  inline void select_body(GLFWwindow* window, double x, double y) {
    int width = 0;
    int height = 0;
    glfwGetWindowSize(window, &width, &height);
    if (width <= 0 || height <= 0) {return;}

    int geom_id = -1;
    int flex_id = -1;
    int skin_id = -1;
    mjtNum selected_point[3] = {};

    const int body_id = mjv_select(
      g_model,
      g_data,
      &g_option,
      static_cast<mjtNum>(width) / height,
      static_cast<mjtNum>(x) / width,
      static_cast<mjtNum>(height - y) / height,
      &g_scene,
      selected_point,
      &geom_id,
      &flex_id,
      &skin_id
    );

    g_perturb.select = body_id > 0 ? body_id : 0;
    g_perturb.flexselect = flex_id;
    g_perturb.skinselect = skin_id;
    g_perturb.active = 0;

    if (g_perturb.select <= 0) {return;}

    mjtNum relative_point[3] = {};
    mju_sub3(relative_point, selected_point, g_data->xpos + 3 * g_perturb.select);
    mju_mulMatTVec(g_perturb.localpos, g_data->xmat + 9 * g_perturb.select, relative_point, 3, 3);
    mjv_initPerturb(g_model, g_data, &g_scene, &g_perturb);
  }

  inline void highlight_selected_body() {
    if (g_perturb.select <= 0) {return;}

    for (int i = 0; i < g_scene.ngeom; ++i) {
      mjvGeom& geom = g_scene.geoms[i];
      if (geom.objtype != mjOBJ_GEOM || geom.objid < 0) {continue;}
      if (g_model->geom_bodyid[geom.objid] != g_perturb.select) {continue;}

      for (int rgb = 0; rgb < 3; ++rgb) {geom.rgba[rgb] = std::min(1.0f, geom.rgba[rgb] + 0.18f);}
      geom.emission = std::max(geom.emission, 0.15f);
    }
  }

  inline void append_frame(const Eigen::Matrix4d& world_T_frame, double axis_length = FRAME_AXIS_LENGTH, double arrow_width = FRAME_ARROW_WIDTH) {
    if (!world_T_frame.allFinite() || g_scene.maxgeom - g_scene.ngeom < 3) {return;}

    constexpr std::array<std::array<float, 4>, 3> kAxisColors = {{
      {0.95f, 0.10f, 0.10f, 1.0f},
      {0.10f, 0.85f, 0.10f, 1.0f},
      {0.10f, 0.35f, 1.00f, 1.0f}
    }};
    const Eigen::Vector3d origin = world_T_frame.block<3, 1>(0, 3);

    for (int axis = 0; axis < 3; ++axis) {
      const Eigen::Vector3d endpoint =
        origin + axis_length * world_T_frame.block<3, 1>(0, axis).normalized();
      mjtNum from[3] = {
        static_cast<mjtNum>(origin.x()),
        static_cast<mjtNum>(origin.y()),
        static_cast<mjtNum>(origin.z())
      };
      mjtNum to[3] = {
        static_cast<mjtNum>(endpoint.x()),
        static_cast<mjtNum>(endpoint.y()),
        static_cast<mjtNum>(endpoint.z())
      };

      mjvGeom& geom = g_scene.geoms[g_scene.ngeom];
      mjv_initGeom(&geom, mjGEOM_ARROW, nullptr, nullptr, nullptr, kAxisColors[axis].data());
      mjv_connector(&geom, mjGEOM_ARROW, static_cast<mjtNum>(arrow_width), from, to);
      geom.category = mjCAT_DECOR;
      ++g_scene.ngeom;
    }
  }

  inline void append_com_marker(const Eigen::Vector3d& world_com) {
    if (!world_com.allFinite() || g_scene.ngeom >= g_scene.maxgeom) {return;}

    const mjtNum size[3] = {static_cast<mjtNum>(COM_MARKER_RADIUS), static_cast<mjtNum>(COM_MARKER_RADIUS), static_cast<mjtNum>(COM_MARKER_RADIUS)};
    const mjtNum pos[3] = {static_cast<mjtNum>(world_com(0)), static_cast<mjtNum>(world_com(1)), static_cast<mjtNum>(world_com(2))};
    constexpr mjtNum mat[9] = {
      1, 0, 0,
      0, 1, 0,
      0, 0, 1
    };

    mjvGeom& geom = g_scene.geoms[g_scene.ngeom];
    mjv_initGeom(&geom, mjGEOM_SPHERE, size, pos, mat, COM_MARKER_COLOR.data());
    geom.category = mjCAT_DECOR;
    ++g_scene.ngeom;
  }

  inline void append_arrow(const Eigen::Vector3d& origin, const Eigen::Vector3d& vector, const double arrow_scale, const double arrow_width, const std::array<float, 4>& color) {
    if (!origin.allFinite() || !vector.allFinite() || vector.squaredNorm() < 1.e-12 || g_scene.ngeom >= g_scene.maxgeom) {return;}
    const Eigen::Vector3d endpoint = origin + arrow_scale*vector;

    mjtNum from[3] = {static_cast<mjtNum>(origin(0)), static_cast<mjtNum>(origin(1)), static_cast<mjtNum>(origin(2))};
    mjtNum to[3] = {static_cast<mjtNum>(endpoint(0)), static_cast<mjtNum>(endpoint(1)), static_cast<mjtNum>(endpoint(2))};

    mjvGeom& geom = g_scene.geoms[g_scene.ngeom];
    mjv_initGeom(&geom, mjGEOM_ARROW, nullptr, nullptr, nullptr, color.data());
    mjv_connector(&geom, mjGEOM_ARROW, static_cast<mjtNum>(arrow_width), from, to);
    geom.category = mjCAT_DECOR;
    ++g_scene.ngeom;
  }

  template <std::size_t N>
  inline void append_segment_vector(const std::array<Eigen::Vector3d, 2*N>& p, const std::array<Eigen::Vector3d, 2*N>& v, const std::size_t idx0, const Eigen::Matrix3d& world_R_body, const Eigen::Vector3d& world_p_body, const Eigen::Matrix3d& Qsi, const double arrow_scale, const double arrow_width, const std::array<float, 4>& color) {
    const Eigen::Matrix3d world_R_strip = world_R_body * Qsi;
    for (std::size_t i=0; i<N; ++i) {
      const std::size_t idx = idx0+i;
      const Eigen::Vector3d origin = world_p_body + world_R_body * p[idx];
      const Eigen::Vector3d vector_world = world_R_strip * v[idx];
      append_arrow(origin, vector_world, arrow_scale, arrow_width, color);
    }
  }

  template <std::size_t N>
  inline void append_segment_vector(const std::array<Eigen::Vector3d, 2*N>& p, const std::array<Eigen::Vector3d, 2*N>& v, const std::size_t idx0, const Eigen::Matrix3d& world_R_body, const Eigen::Vector3d& world_p_body, const std::array<Eigen::Matrix3d, N>& Qsi, const double arrow_scale, const double arrow_width, const std::array<float, 4>& color) {
    for (std::size_t i=0; i<N; ++i) {
      const std::size_t idx = idx0+i;
      const Eigen::Vector3d origin = world_p_body + world_R_body * p[idx];
      const Eigen::Vector3d vector_world = world_R_body * (Qsi[i] * v[idx]);
      append_arrow(origin, vector_world, arrow_scale, arrow_width, color);
    }
  }

  template <std::size_t N>
  inline void append_segment_vector(const std::array<Eigen::Vector3d, 2*N>& p, const Eigen::Vector3d& v, const std::size_t idx0, const Eigen::Matrix3d& world_R_body, const Eigen::Vector3d& world_p_body, const Eigen::Matrix3d& Qsi, const double arrow_scale, const double arrow_width, const std::array<float, 4>& color) {
    const Eigen::Vector3d vector_world = world_R_body * (Qsi * v);
    for (std::size_t i=0; i<N; ++i) {
      const std::size_t idx = idx0+i;
      const Eigen::Vector3d origin = world_p_body + world_R_body * p[idx];
      append_arrow(origin, vector_world, arrow_scale, arrow_width, color);
    }
  }

  inline void append_strip_frame_and_plate(const Eigen::Matrix4d& world_T_strip, const double l, const double width) {
    const bool draw_plate = l > 0.0 && width > 0.0;
    const int required_geoms = draw_plate ? 4 : 3;
    if (g_scene.maxgeom-g_scene.ngeom < required_geoms) {return;}

    append_frame(world_T_strip, STRIP_FRAME_SCALE*FRAME_AXIS_LENGTH, STRIP_FRAME_SCALE*FRAME_ARROW_WIDTH);
    if (!draw_plate) {return;}

    constexpr float gray[4] = {0.70f, 0.70f, 0.70f, 0.30f};
    const Eigen::Vector3d origin = world_T_strip.block<3, 1>(0, 3);
    const Eigen::Matrix3d world_R_strip = world_T_strip.block<3, 3>(0, 0);
    const Eigen::Vector3d center = origin + 0.5*l*world_R_strip.col(0).normalized();

    const mjtNum size[3] = {static_cast<mjtNum>(0.5*l), static_cast<mjtNum>(0.5*width), static_cast<mjtNum>(0.5*STRIP_PLATE_THICKNESS)};
    const mjtNum pos[3] = {static_cast<mjtNum>(center(0)), static_cast<mjtNum>(center(1)), static_cast<mjtNum>(center(2))};
    const mjtNum mat[9] = {
      static_cast<mjtNum>(world_R_strip(0, 0)), static_cast<mjtNum>(world_R_strip(0, 1)), static_cast<mjtNum>(world_R_strip(0, 2)),
      static_cast<mjtNum>(world_R_strip(1, 0)), static_cast<mjtNum>(world_R_strip(1, 1)), static_cast<mjtNum>(world_R_strip(1, 2)),
      static_cast<mjtNum>(world_R_strip(2, 0)), static_cast<mjtNum>(world_R_strip(2, 1)), static_cast<mjtNum>(world_R_strip(2, 2))
    };

    mjvGeom& geom = g_scene.geoms[g_scene.ngeom];
    mjv_initGeom(&geom, mjGEOM_BOX, size, pos, mat, gray);
    geom.category = mjCAT_DECOR;
    ++g_scene.ngeom;
  }

  inline void append_radius_strip_frames(const std::array<Eigen::Vector3d, 2*param::NR>& p, const std::size_t idx0, const Eigen::Matrix4d& wTb, const std::array<Eigen::Matrix3d, param::NR>& Qri, const std::array<double, param::NR>& cos_psi) {
    const Eigen::Matrix3d world_R_body = wTb.block<3, 3>(0, 0);
    const Eigen::Vector3d world_p_body = wTb.block<3, 1>(0, 3);
    Eigen::Matrix4d world_T_strip = Eigen::Matrix4d::Identity();
    for (std::size_t i=0; i<param::NR; ++i) {
      world_T_strip.block<3, 3>(0, 0) = world_R_body * Qri[i];
      world_T_strip.block<3, 1>(0, 3) = world_p_body + world_R_body * p[idx0+i];
      const double l = param::C_R0 + param::DL_R*static_cast<double>(i);
      const double width = param::DY_R*std::abs(cos_psi[i]);
      append_strip_frame_and_plate(world_T_strip, l, width);
    }
  }

  inline void append_humerus_strip_frames(const std::array<Eigen::Vector3d, 2*param::NH>& p, const std::size_t idx0, const Eigen::Matrix4d& wTb, const Eigen::Matrix3d& Qsi, const double cos_psi) {
    const Eigen::Matrix3d world_R_body = wTb.block<3, 3>(0, 0);
    const Eigen::Vector3d world_p_body = wTb.block<3, 1>(0, 3);
    const double width = param::DY_H * std::abs(cos_psi);

    Eigen::Matrix4d world_T_strip = Eigen::Matrix4d::Identity();
    world_T_strip.block<3, 3>(0, 0) = world_R_body * Qsi;
    for (std::size_t i=0; i<param::NH; ++i) {
      world_T_strip.block<3, 1>(0, 3) = world_p_body + world_R_body * p[idx0+i];
      const double l = param::C_H0 + param::DL_H * static_cast<double>(i);
      append_strip_frame_and_plate(world_T_strip, l, width);
    }
  }

  inline void append_manus_strip_frames(const std::array<Eigen::Vector3d, 2*param::NM>& p, const std::size_t idx0, const Eigen::Matrix4d& wTb, const Eigen::Matrix3d& Qsi, const double cos_psi) {
    const Eigen::Matrix3d world_R_body = wTb.block<3, 3>(0, 0);
    const Eigen::Vector3d world_p_body = wTb.block<3, 1>(0, 3);
    const double width = param::DY_M * std::abs(cos_psi);

    Eigen::Matrix4d world_T_strip = Eigen::Matrix4d::Identity();
    world_T_strip.block<3, 3>(0, 0) = world_R_body * Qsi;
    for (std::size_t i=0; i<param::DECLINE_IDX_K; ++i) {
      world_T_strip.block<3, 1>(0, 3) = world_p_body + world_R_body * p[idx0+i];
      const double l = param::C_M0 + param::DL_M1 * static_cast<double>(i);
      append_strip_frame_and_plate(world_T_strip, l, width);
    }
    for (std::size_t i=param::DECLINE_IDX_K; i<param::NM; ++i) {
      world_T_strip.block<3, 1>(0, 3) = world_p_body + world_R_body * p[idx0+i];
      const double l = param::C_MK + param::DL_M2 * static_cast<double>(i-param::DECLINE_IDX_K);
      append_strip_frame_and_plate(world_T_strip, l, width);
    }
  }

  inline void append_tail_strip_frames(const std::array<Eigen::Vector3d, 2*param::NT>& p, const std::array<Eigen::Matrix3d, 2>& bR_t, const double theta_t, const Eigen::Matrix4d& wTb) {
    const Eigen::Matrix3d world_R_body = wTb.block<3, 3>(0, 0);
    const Eigen::Vector3d world_p_body = wTb.block<3, 1>(0, 3);
    const double sin_theta_t = std::sin(theta_t);
    const double cos_theta_t = std::cos(theta_t);
    const double side_width = param::C_T*std::abs(sin_theta_t)/static_cast<double>(param::NT_S);
    const double sin_theta_t2 = sin_theta_t*sin_theta_t;
    Eigen::Matrix4d world_T_strip = Eigen::Matrix4d::Identity();

    for (std::size_t section=0; section<2; ++section) {
      const std::size_t idx0 = section*param::NT;
      world_T_strip.block<3, 3>(0, 0) = world_R_body*bR_t[section];

      for (std::size_t i=0; i<param::NT_R; ++i) {
        world_T_strip.block<3, 1>(0, 3) = world_p_body + world_R_body*p[idx0+i];
        append_strip_frame_and_plate(world_T_strip, param::C_T, param::DY_T);
      }

      if (side_width <= 0.0) {continue;}
      for (std::size_t i=0; i<param::NT_S; ++i) {
        const double n = (static_cast<double>(i)+0.5)/static_cast<double>(param::NT_S);
        const double radicand = std::max(0.0, 1.0-n*n*sin_theta_t2);
        const double chord = param::C_T*std::max(0.0, std::sqrt(radicand)-n*cos_theta_t);
        world_T_strip.block<3, 1>(0, 3) = world_p_body + world_R_body*p[idx0+param::NT_R+i];
        append_strip_frame_and_plate(world_T_strip, chord, side_width);
      }
    }
  }

  inline void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    g_left_pressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    g_middle_pressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    g_right_pressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    glfwGetCursorPos(window, &g_last_x, &g_last_y);

    layout_ui(window);
    update_ui_state(window);
    g_ui_state.button =
      button == GLFW_MOUSE_BUTTON_LEFT ? mjBUTTON_LEFT :
      button == GLFW_MOUSE_BUTTON_RIGHT ? mjBUTTON_RIGHT :
      button == GLFW_MOUSE_BUTTON_MIDDLE ? mjBUTTON_MIDDLE : mjBUTTON_NONE;
    g_ui_state.type = action == GLFW_PRESS ? mjEVENT_PRESS : mjEVENT_RELEASE;
    if (action == GLFW_PRESS) {
      g_ui_state.buttontime = glfwGetTime();
      if (g_ui_state.mouserect) {
        g_ui_state.dragbutton = g_ui_state.button;
        g_ui_state.dragrect = g_ui_state.mouserect;
      }
    }
    const bool ui_handled = process_ui_event();
    if (action == GLFW_RELEASE) {
      g_ui_state.dragbutton = 0;
      g_ui_state.dragrect = 0;
    }
    if (ui_handled) {return;}

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {select_body(window, g_last_x, g_last_y);}

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS && (mods & GLFW_MOD_SHIFT)) {
      select_body(window, g_last_x, g_last_y);
      if (g_perturb.select > 0) {g_perturb.active = mjPERT_TRANSLATE;}
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {g_perturb.active = 0;}
  }

  inline void mouse_move_callback(GLFWwindow* window, double x, double y) {
    if (!g_left_pressed && !g_middle_pressed && !g_right_pressed) {return;}

    const double dx = x - g_last_x;
    const double dy = y - g_last_y;
    g_last_x = x;
    g_last_y = y;

    layout_ui(window);
    update_ui_state(window);
    g_ui_state.type = mjEVENT_MOVE;
    if (process_ui_event()) {return;}

    int width = 0;
    int height = 0;
    glfwGetWindowSize(window, &width, &height);
    if (height <= 0) {return;}

    if (g_perturb.active) {
      mjv_movePerturb(g_model, g_data, mjMOUSE_MOVE_V, dx / height, dy / height, &g_scene, &g_perturb);
      return;
    }

    const bool shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;

    mjtMouse action;
    if (g_right_pressed) {action = shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;}
    else if (g_left_pressed) {action = shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;}
    else {action = mjMOUSE_ZOOM;}

    const bool is_pan = action == mjMOUSE_MOVE_H || action == mjMOUSE_MOVE_V;
    std::array<mjtNum, 3> lookat_before{};
    if (is_pan) {std::copy_n(g_camera.lookat, lookat_before.size(), lookat_before.begin());}

    #if mjVERSION_HEADER >= 3011000
      mjv_moveCamera(g_model, action, dx / height, dy / height, &g_camera);
    #else
      mjv_moveCamera(g_model, action, dx / height, dy / height, &g_scene, &g_camera);
    #endif

    if (is_pan) {
      for (std::size_t axis = 0; axis < g_camera_pan_offset.size(); ++axis) {
        g_camera_pan_offset[axis] += g_camera.lookat[axis] - lookat_before[axis];
      }
    }
  }

  inline void scroll_callback(GLFWwindow* window, double x_offset, double y_offset) {
    layout_ui(window);
    update_ui_state(window);
    g_ui_state.type = mjEVENT_SCROLL;
    g_ui_state.sx = x_offset;
    g_ui_state.sy = y_offset;
    if (process_ui_event()) {return;}
    #if mjVERSION_HEADER >= 3011000
      mjv_moveCamera(g_model, mjMOUSE_ZOOM, 0.0, -0.05 * y_offset, &g_camera);
    #else
      mjv_moveCamera(g_model, mjMOUSE_ZOOM, 0.0, -0.05 * y_offset, &g_scene, &g_camera);
    #endif
  }

  inline void set_rgb(float color[3], float red, float green, float blue) {
    color[0] = red;
    color[1] = green;
    color[2] = blue;
  }

  inline void apply_flat_ui_theme(mjuiThemeColor& color) {
    set_rgb(color.master,       0.07f, 0.08f, 0.10f);
    set_rgb(color.thumb,        0.30f, 0.33f, 0.39f);
    set_rgb(color.secttitle,    0.13f, 0.15f, 0.18f);
  #if mjVERSION_HEADER >= 320
    set_rgb(color.secttitle2,   0.13f, 0.15f, 0.18f);
    set_rgb(color.separator,    0.13f, 0.15f, 0.18f);
    set_rgb(color.separator2,   0.13f, 0.15f, 0.18f);
  #endif
    set_rgb(color.sectfont,     0.92f, 0.93f, 0.95f);
    set_rgb(color.sectsymbol,   0.65f, 0.68f, 0.73f);
    set_rgb(color.sectpane,     0.09f, 0.10f, 0.12f);
    set_rgb(color.shortcut,     0.18f, 0.20f, 0.24f);
    set_rgb(color.fontactive,   0.88f, 0.90f, 0.93f);
    set_rgb(color.fontinactive, 0.45f, 0.48f, 0.54f);
    set_rgb(color.decorinactive,  0.24f, 0.27f, 0.32f);
    set_rgb(color.decorinactive2, 0.24f, 0.27f, 0.32f);
    set_rgb(color.button,       0.18f, 0.20f, 0.24f);
    set_rgb(color.check,        0.25f, 0.55f, 0.95f);
    set_rgb(color.radio,        0.25f, 0.55f, 0.95f);
    set_rgb(color.select,       0.18f, 0.20f, 0.24f);
    set_rgb(color.select2,      0.18f, 0.20f, 0.24f);
    set_rgb(color.slider,       0.16f, 0.18f, 0.22f);
    set_rgb(color.slider2,      0.25f, 0.55f, 0.95f);
    set_rgb(color.edit,         0.14f, 0.16f, 0.19f);
    set_rgb(color.edit2,        0.70f, 0.25f, 0.25f);
    set_rgb(color.cursor,       0.92f, 0.93f, 0.95f);
  }

  inline void initialize_ui(GLFWwindow* window) {
    g_ui.spacing = mjui_themeSpacing(0);
    g_ui.spacing.total = 180;
    g_ui.spacing.label = 50;
    g_ui.color = mjui_themeColor(0);
    apply_flat_ui_theme(g_ui.color);
    g_ui.rectid = 1;
    g_ui.auxid = 0;

    const mjuiDef definitions[] = {
      {mjITEM_SECTION, "simulation", 1, nullptr, ""},
      {mjITEM_SLIDERNUM, "spd", 2, &g_sim_speed, "0 1"},
      {mjITEM_SECTION, "joint control", 1, nullptr, ""},
      {mjITEM_SLIDERNUM, "J1",  2, &g_command_theta[0], "-1.5707963268 1.5707963268"},
      {mjITEM_SLIDERNUM, "J2",  2, &g_command_theta[1], "-1.5707963268 1.5707963268"},
      {mjITEM_SLIDERNUM, "J3",  2, &g_command_theta[2], "-1.5707963268 3.1415926536"},
      {mjITEM_SLIDERNUM, "J4",  2, &g_command_theta[3], "-1.5707963268 2.6179938780"},
      {mjITEM_SLIDERNUM, "J5",  2, &g_command_theta[4], "-1.5707963268 1.5707963268"},
      {mjITEM_SLIDERNUM, "J6",  2, &g_command_theta[5], "-2.6179938780 1.5707963268"},
      {mjITEM_STATIC, "", 1, nullptr, " "},
      {mjITEM_SLIDERNUM, "J7",  2, &g_command_theta[6], "-1.5707963268 1.5707963268"},
      {mjITEM_SLIDERNUM, "J8",  2, &g_command_theta[7], "-1.5707963268 1.5707963268"},
      {mjITEM_SLIDERNUM, "J9",  2, &g_command_theta[8], "-1.5707963268 3.1415926536"},
      {mjITEM_SLIDERNUM, "J10", 2, &g_command_theta[9], "-1.5707963268 2.6179938780"},
      {mjITEM_SLIDERNUM, "J11", 2, &g_command_theta[10], "-1.5707963268 1.5707963268"},
      {mjITEM_SLIDERNUM, "J12", 2, &g_command_theta[11], "-2.6179938780 1.5707963268"},
      {mjITEM_STATIC, "", 1, nullptr, " "},
      {mjITEM_SLIDERNUM, "J13", 2, &g_command_theta[param::NUM_WING_JOINTS], "-1.5707963268 1.5707963268"},
      {mjITEM_SLIDERNUM, "J14", 2, &g_command_theta[param::NUM_WING_JOINTS+1], "-1.5707963268 1.5707963268"},
      {mjITEM_SLIDERNUM, "theta_t", 2, &g_command_theta_t, "0 1.5707963268"},
      {mjITEM_SECTION, "view control", 1, nullptr, ""},
      {mjITEM_SELECT, "tgt", 2, &g_arrow_view, "none\nwing\ntail\nboth"},
      {mjITEM_SELECT, "type", 2, &g_arrow_quantity, "v [m/s]\na [m/s^2]\nw [rad/s]\nwdot [rad/s^2]"},
      {mjITEM_SELECT, "trj", 2, &g_trajectory_view, "off\non"},
      {mjITEM_END}
    };
    mjui_add(&g_ui, definitions);
    mjui_resize(&g_ui, &g_context);
    mjr_addAux(g_ui.auxid, g_ui.width, g_ui.maxheight, g_ui.spacing.samples, &g_context);
    layout_ui(window);
    update_ui_state(window);
    mjui_update(-1, -1, &g_ui, &g_ui_state, &g_context);
  }

  inline void update_sim_speed_slider_color() {
    const std::array<float, 3> original_slider2 = {
      g_ui.color.slider2[0],
      g_ui.color.slider2[1],
      g_ui.color.slider2[2]
    };

    set_rgb(g_ui.color.slider2, 0.20f, 0.75f, 0.38f);
    mjui_update(0, 0, &g_ui, &g_ui_state, &g_context);
    std::copy(original_slider2.begin(), original_slider2.end(), g_ui.color.slider2);
  }

  inline void render_joint_current_overlay() {
    if (!g_model || !g_data || g_ui.nsect < 2) {return;}
    constexpr std::array<int, param::NUM_JOINTS> kSliderItemIndices = []() {
      std::array<int, param::NUM_JOINTS> result{};
      for (std::size_t i=0; i<param::NUM_JOINTS; ++i) {result[i] = static_cast<int>(i + i/param::NUM_WING_JOINTS_PER_WING);}
      return result;
    }();

    constexpr int marker_width = 4;
    constexpr float marker_r = 1.00f;
    constexpr float marker_g = 0.55f;
    constexpr float marker_b = 0.10f;
    constexpr float marker_a = 0.45f;

    const mjrRect ui_viewport = g_ui_state.rect[g_ui.rectid];
    const mjuiSection& section = g_ui.sect[1];

    // Convert UI-local y coordinates to framebuffer coordinates.
    const int ui_bottom = ui_viewport.bottom + ui_viewport.height - g_ui.height + (g_ui.height > ui_viewport.height ? g_ui.scroll : 0);

    for (std::size_t i = 0; i < kSliderItemIndices.size(); ++i) {
      const int qpos_adr = g_joint_qpos_adrs[i];
      const int item_idx = kSliderItemIndices[i];

      if (qpos_adr < 0 || qpos_adr >= g_model->nq) {continue;}
      if (item_idx < 0 || item_idx >= section.nitem) {continue;}

      const mjuiItem& item = section.item[item_idx];
      if (item.type != mjITEM_SLIDERNUM || item.skip) {continue;}

      const double lower = item.slider.range[0];
      const double upper = item.slider.range[1];
      if (!(upper > lower)) {continue;}

      const double current = static_cast<double>(g_data->qpos[qpos_adr]);
      const double ratio = std::clamp((current - lower) / (upper - lower), 0.0, 1.0);

      const int marker_center = ui_viewport.left + item.rect.left + mju_round(ratio * static_cast<double>(item.rect.width));

      mjrRect marker = {marker_center - marker_width / 2, ui_bottom + item.rect.bottom + 2, marker_width, std::max(1, item.rect.height - 4)};

      // Clip the marker to the visible UI viewport.
      const int left = std::max(marker.left, ui_viewport.left);
      const int right = std::min(marker.left + marker.width, ui_viewport.left + ui_viewport.width);
      const int bottom = std::max(marker.bottom, ui_viewport.bottom);
      const int top = std::min(marker.bottom + marker.height, ui_viewport.bottom + ui_viewport.height);
      if (right <= left || top <= bottom) {continue;}

      marker.left = left;
      marker.bottom = bottom;
      marker.width = right - left;
      marker.height = top - bottom;

      mjr_rectangle(marker, marker_r, marker_g, marker_b, marker_a);
    }
  }

  inline void render_ui(GLFWwindow* window) {
    layout_ui(window);
    update_sim_speed_slider_color();
    mjui_render(&g_ui, &g_ui_state, &g_context);
    render_joint_current_overlay();
  }

  inline GLFWwindow* initialize(int argc, char** argv) {
    const std::string scene_path = argc > 1 ? argv[1] : std::string(BIRD_PROJECT_DIR) + "/mujoco/scene.xml";

    char error[1024] = {};
    g_model = mj_loadXML(scene_path.c_str(), nullptr, error, static_cast<int>(sizeof(error)));
    if (!g_model) {
      std::fprintf(stderr, "Failed to load MuJoCo model:\n%s\n", error);
      std::exit(EXIT_FAILURE);
    }

    g_data = mj_makeData(g_model);
    if (!g_data) {
      std::fprintf(stderr, "Failed to allocate mjData.\n");
      mj_deleteModel(g_model);
      std::exit(EXIT_FAILURE);
    }

    g_actuator_ids.fill(-1);
    g_joint_qpos_adrs.fill(-1);

    for (std::size_t i = 0; i < param::NUM_JOINTS; ++i) {
      const int actuator_id = mj_name2id(g_model, mjOBJ_ACTUATOR, param::ACTUATOR_NAMES[i]);
      g_actuator_ids[i] = actuator_id;

      if (actuator_id < 0) {continue;}
      if (g_model->actuator_trntype[actuator_id] != mjTRN_JOINT) {continue;}

      const int joint_id = g_model->actuator_trnid[2 * actuator_id];
      if (joint_id < 0 || joint_id >= g_model->njnt) {continue;}

      g_joint_qpos_adrs[i] = g_model->jnt_qposadr[joint_id];
    }

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
      std::fprintf(stderr, "Could not initialize GLFW.\n");
      mj_deleteData(g_data);
      mj_deleteModel(g_model);
      std::exit(EXIT_FAILURE);
    }

    GLFWwindow* window = glfwCreateWindow(1200, 900, "Bird sim", nullptr, nullptr);
    if (!window) {
      std::fprintf(stderr, "Could not create GLFW window.\n");
      glfwTerminate();
      mj_deleteData(g_data);
      mj_deleteModel(g_model);
      std::exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    glfwSetKeyCallback(window, keyboard_callback);
    glfwSetCursorPosCallback(window, mouse_move_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    mjv_defaultCamera(&g_camera);
    mjv_defaultOption(&g_option);
    mjv_defaultPerturb(&g_perturb);
    mjv_defaultScene(&g_scene);
    mjr_defaultContext(&g_context);

    g_option.flags[mjVIS_PERTFORCE] = 1;
    g_option.flags[mjVIS_PERTOBJ] = 0;
    g_option.flags[mjVIS_SELECT] = 1;

    g_camera.type = mjCAMERA_FREE;
    g_camera.trackbodyid = -1;

    g_camera.distance = 20.0;
    g_camera.azimuth = 120.0;
    g_camera.elevation = -25.0;

    mjv_makeScene(g_model, &g_scene, 2000);
    mjr_makeContext(g_model, &g_context, mjFONTSCALE_100);
    initialize_ui(window);

    return window;
  }

  inline void shutdown(GLFWwindow* window) {
    mjr_freeContext(&g_context);
    mjv_freeScene(&g_scene);
    glfwDestroyWindow(window);
    glfwTerminate();
    mj_deleteData(g_data);
    mj_deleteModel(g_model);
  }

}  // namespace mj_utils
