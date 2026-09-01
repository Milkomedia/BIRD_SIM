(GPT generated Readme)

# BIRD_SIM

Real-time MuJoCo-based simulation framework for bird-like flapping-wing aerial vehicles.

## Current features

* MuJoCo model with six joints per wing and articulated left/right tail
* Real-time C++17 simulation at 5 kHz with a separate approximately 60 Hz rendering path
* Modified strip-theory aerodynamic model for both wings and tail

  * humerus: 7 strips per wing
  * radius: 6 strips per wing
  * manus: 25 strips per wing
  * tail: independently discretized right/left sections
* Bilinear aerodynamic-coefficient lookup in angle of attack and Reynolds number
* Goman--Khrabrov-style dynamic-stall correction with separate upper- and lower-surface separation states
* Two-state Wagner/Jones wake-memory correction
* Translational and rotational added-mass effects

  * acceleration-dependent inertia is added to MuJoCo's generalized mass matrix
  * velocity-dependent bias loads are applied as external wrenches
* Ellipsoidal body-fluid model including drag, lift, Magnus, angular drag, and virtual-inertia terms
* DC motor, gearbox, ESC lag, PD servo, torque saturation, rotor inertia, and joint damping models
* Optional CRSF/ExpressLRS input
* In-viewer visualization of wing/tail strip frames, aerodynamic forces, velocity, acceleration, angular velocity, and angular acceleration
* Shared-memory telemetry with live plotting and replay

## Requirements

The current development target is **macOS**. The CRSF serial implementation uses Apple's `IOKit/serial/ioss.h`. Most other parts of the simulator are platform-independent.

Required C++ dependencies:

* CMake 3.16 or newer
* C++17 compiler
* MuJoCo 3.11
* GLFW 3
* Eigen 3
* POSIX threads and memory mapping

Required Python packages:

```bash
python3 -m pip install numpy PyQt5 pyqtgraph
```

## Build

```bash
git clone https://github.com/Milkomedia/BIRD_SIM.git
cd BIRD_SIM

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"

cmake --build build --parallel
```

`CMAKE_PREFIX_PATH` may be omitted when MuJoCo, GLFW, and Eigen are already in CMake's default search paths.

## Run

```bash
./build/run
```

A different MuJoCo XML file can be supplied as the first argument:

```bash
./build/run /absolute/path/to/scene.xml
```

When an ELRS receiver is not detected, the current fallback test case uses:

* synchronized right/left flapping at 3 Hz
* flapping amplitude of ±90 degrees
* steady wind `(-10, 0, 0)` m/s in the internal NED convention
* selected joint commands from MuJoCo UI sliders

The default `mujoco/bird.xml` currently keeps the vehicle fixed in space. Free-flight simulation requires enabling the root free joint and using an appropriate controller.

### Viewer controls

| Input                | Action                                         |
| -------------------- | ---------------------------------------------- |
| `Space`              | Pause or resume simulation                     |
| `Backspace`          | Reset simulation and aerodynamic memory states |
| `Esc`                | Exit                                           |
| Left drag            | Rotate camera                                  |
| Middle drag / scroll | Zoom                                           |
| Shift + right drag   | Apply a translational perturbation             |

The right-side UI selects the displayed strip quantity (`none`, `v`, `a`, `w`, or `wdot`) and provides joint-command sliders.

## Telemetry monitor

The simulator publishes schema-described telemetry at 250 Hz to:

```text
/tmp/bird_sim.mmap
```

Start the live monitor with:

```bash
python3 viewer.py
```

The monitor provides:

* Flight
* Joint
* Strip Flow
* Strip Loads
* Strip Segment
* Channel Browser

Both wing and tail aerodynamic quantities are available in the strip views.

Unless `--no-record` is specified, the monitor records the session as a compressed `.npz` file and saves the final `.mmap` snapshot under `log/`.

Replay:

```bash
python3 viewer.py log/npz/<recording>.npz
python3 viewer.py log/mmap/<recording>.mmap
```

Useful options:

```bash
python3 viewer.py --mmap /tmp/bird_sim.mmap \
  --log-dir log \
  --update-ms 40

python3 viewer.py --no-record
```

## ELRS/CRSF input

The receiver interface expects 16 packed CRSF channels at 420000 baud.

The serial device is configured in `include/ELRS.hpp`:

```cpp
static constexpr const char* DEVICE = "/dev/cu.usbserial-DU0E4O7H";
```

Change this value to match the local receiver.

A valid CRSF frame must be received during startup for ELRS control to be enabled.

## Model conventions and configuration

* Vehicle states and aerodynamic quantities use NED/FRD conventions internally.
* MuJoCo rendering uses its world convention, with conversions performed at the simulator boundary.
* Main geometry, motor, timing, wing, and tail parameters are defined in `include/params.hpp`.
* Aerodynamic lookup tables are stored in `include/coeff/coeff.hpp`.
* Wing LUTs currently cover:

  * angle of attack: `[-10, 25] deg`
  * Reynolds number: `[28753, 215645]`
* Out-of-range wing LUT queries are clamped.
* The current tail LUT is a provisional flat-plate approximation and is intended to be replaced with experimental data.

## Default rates

| Quantity                |      Current value |
| ----------------------- | -----------------: |
| Simulation step         |     200 us (5 kHz) |
| Render period           | 16.667 ms (~60 Hz) |
| Telemetry rate          |             250 Hz |
| Telemetry ring duration |                3 s |

## Repository layout

```text
.
|-- main.cpp
|-- include/
|   |-- MST.hpp               # Wing/tail strip aerodynamics
|   |-- Servo.hpp             # Servo/motor model
|   |-- ELRS.hpp              # CRSF receiver interface
|   |-- mmap_manager.hpp      # Shared-memory telemetry
|   |-- mujoco_utils.hpp      # MuJoCo viewer utilities
|   |-- params.hpp            # Simulation/model parameters
|   `-- coeff/                # Aerodynamic lookup tables
|-- src/                      # C++ implementations
|-- mujoco/                   # MuJoCo XML models and STL meshes
`-- viewer.py                 # Live telemetry monitor and replay viewer
```
