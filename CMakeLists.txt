cmake_minimum_required(VERSION 3.16)
project(bird_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(mujoco CONFIG REQUIRED PATHS "$ENV{HOME}/.local/mujoco")
find_package(glfw3 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_executable(
  run
  main.cpp
  src/Servo.cpp
  src/ELRS.cpp
  src/MST.cpp
)

target_include_directories(
  run
  PRIVATE
    include
    src
)

target_link_libraries(
  run
  PRIVATE
    mujoco::mujoco
    glfw
    Eigen3::Eigen
    Threads::Threads
)

target_compile_definitions(
  run
  PRIVATE BIRD_PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)