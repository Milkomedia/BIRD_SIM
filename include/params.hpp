#pragma once

#include <Eigen/Core>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace param {
  inline constexpr std::chrono::steady_clock::duration SIM_DT_US = std::chrono::microseconds(200);
  inline constexpr std::chrono::steady_clock::duration RENDER_DT_US = std::chrono::microseconds(16667);
  inline constexpr std::chrono::steady_clock::duration SPIN_MARGIN_US = std::chrono::microseconds(20);
  inline constexpr double SIM_DT_SEC = std::chrono::duration<double>(SIM_DT_US).count();

  inline constexpr bool DISABLE_WAGNER = true;

  // ----- Flapping Motion Parameters -----
  inline constexpr double R1 = 0.45;
  inline constexpr double R2 = 0.35;
  inline constexpr double FLAPPING_DELTA_0  = -3.00 * M_PI / 180.0; // [rad]
  inline constexpr double PITCHING_DELTA_0  =  6.75 * M_PI / 180.0; // [rad]
  inline constexpr double FOLDING_DELTA_0   =  5.00 * M_PI / 180.0; // [rad]
  inline constexpr double SWEEP_AMPLITUDE   =  10.0 * M_PI / 180.0; // [rad]
  inline constexpr double FOLDING_AMPLITUDE =  22.5 * M_PI / 180.0; // [rad]

  inline constexpr double MIN_FREQ = 1.5; // [Hz]
  inline constexpr double MAX_FREQ = 4.0;
  inline constexpr double MIN_FLAPPING_AMPLITUDE  = 0.8 * M_PI / 8.0; // mean flapping amplitude [rad]
  inline constexpr double MAX_FLAPPING_AMPLITUDE  = 3.0 * M_PI / 8.0;
  inline constexpr double MIN_FLAPPING_DIFFERENCE = 0.0;              // flapping amplitude difference [rad]
  inline constexpr double MAX_FLAPPING_DIFFERENCE = M_PI / 4.0;
  inline constexpr double MIN_PITCHING_AMPLITUDE  = 0.0;              // mean pitching amplitude [rad]
  inline constexpr double MAX_PITCHING_AMPLITUDE  = M_PI / 4.0;
  inline constexpr double MIN_PITCHING_DIFFERENCE = 0.0;              // pitching amplitude difference [rad]
  inline constexpr double MAX_PITCHING_DIFFERENCE = M_PI / 8.0;
  inline constexpr double MIN_SWEEP_BIAS = 0.0;                       // sweep bias [rad]
  inline constexpr double MAX_SWEEP_BIAS = M_PI / 8.0;
  inline constexpr double MIXER_B_FD_FRACTION = 1e-3; // fraction of each input range

  // ----- model topology -----
  inline constexpr std::size_t NUM_WING_JOINTS_PER_WING = 6;
  inline constexpr std::size_t NUM_WING_JOINTS = 2 * NUM_WING_JOINTS_PER_WING;
  inline constexpr std::size_t NUM_TAIL_JOINTS = 2;
  inline constexpr std::size_t NUM_JOINTS = NUM_WING_JOINTS + NUM_TAIL_JOINTS;
  inline constexpr std::size_t NUM_MOTOR_MODELS = NUM_WING_JOINTS_PER_WING + NUM_TAIL_JOINTS;

  // ----- body elipsoid -----
  inline const Eigen::Vector3d ELLIPSOID_CENTER_POS = Eigen::Vector3d(-0.084, 0.0, 0.0); // [m], body FRD
  inline const Eigen::Vector3d ELLIPSOID_SIZE = Eigen::Vector3d(0.156, 0.06, 0.06);       // [m], semi-axes x,y,z

  // ----- wing wake to tail -----
  inline constexpr std::size_t WAKE_UPDATE_DECIMATION = 10; // 500 Hz at the current simulation rate
  inline constexpr std::size_t WAKE_SPAN_PANELS = 12;       // 2 humerus + 2 radius + 8 manus
  inline constexpr std::size_t WAKE_AGE_CELLS = 32;         // 64 ms prescribed-wake history
  inline constexpr double WAKE_CORE_RADIUS = 0.010;         // [m], initial Scully vortex-core radius

  // ----- MST -----
  inline constexpr double AIR_DENSITY = 1.225;               // [kg/m^3]
  inline constexpr double AIR_KINEMATIC_VISCOSITY = 1.5e-5;  // [m^2/s]

  inline constexpr std::size_t NH   = 7;  // number of humerus strip frame
  inline constexpr std::size_t NR   = 6;  // number of radius strip frame
  inline constexpr std::size_t NM   = 25; // number of manus strip frame
  inline constexpr std::size_t NT_R = 3;  // number of tail root strip frames per section
  inline constexpr std::size_t NT_S = 8;  // number of tail side strip frames per section
  inline constexpr std::size_t NT   = NT_R + NT_S; // number of tail strip frames per section

  inline constexpr double L_H = 0.100; // humerus length [m]
  inline constexpr double L_R = 0.086; // radius length [m]
  inline constexpr double L_M = 0.420; // manus length [m]
  inline constexpr double L_T = 0.060; // tail half-root length [m]
  inline constexpr double L_P = 0.189; // primary feather length [m]

  inline constexpr double DY_H = L_H / static_cast<double>(NH-1);   // initial humerus strip width [m]
  inline constexpr double DY_R = L_R / static_cast<double>(NR-1);   // initial radius strip width [m]
  inline constexpr double DY_M = L_M / static_cast<double>(NM-1);   // initial manus strip width [m]
  inline constexpr double DY_T = L_T / static_cast<double>(NT_R);   // tail root strip width [m]

  inline constexpr double C_H0  = 0.217; // first humerus chord length [m]
  inline constexpr double C_R0  = 0.200; // first radius chord length [m]
  inline constexpr double C_M0  = 0.215; // first manus chord length [m]
  inline constexpr double C_MK  = 0.172; // manus chord length at DECLINE_IDX_K [m]
  inline constexpr double C_MNM = 0.049; // last manus chord length [m]
  inline constexpr double C_T   = 0.192; // tail root chord length [m]
  inline constexpr double D_P   = 0.017; // primary bending length [m]

  constexpr std::size_t DECLINE_IDX_K = static_cast<std::size_t>((L_M - L_P) / DY_M);
  inline constexpr double DL_H  = (C_R0  - C_H0) / static_cast<double>(NH-1); // change in chord length at humerus strip [m]
  inline constexpr double DL_R  = (C_M0  - C_R0) / static_cast<double>(NR-1); // change in chord length at radius strip [m]
  inline constexpr double DL_M1 = (C_MK  - C_M0) / static_cast<double>(DECLINE_IDX_K-1); // change in chord length at manus strip [m]
  inline constexpr double DL_M2 = (C_MNM - C_MK) / static_cast<double>(NM-DECLINE_IDX_K-1); // change in chord length at manus strip [m]

  // ----- Kinematics Parameters -----
  inline constexpr std::array<double, NUM_JOINTS> INITIAL_DES_THETA { // [rad]
    0.125, -0.28, 0.1963495408, -0.7072074129, 0.19, 0.5173155903,
    0.125, -0.28, 0.1963495408, -0.7072074129, 0.19, 0.5173155903,
    0.0, -0.05
  };

  inline constexpr double INITIAL_THETA_T = 0.35; // [rad]

  inline const std::array<Eigen::Matrix4d, 8> J_T_S0 = {
    // Right Wing
    (Eigen::Matrix4d() << // Humerus
      -0.28106457057544920,  0.14947796388954818,  0.94797628951291457, -0.00514105438769492,
       0.14556749017354492, -0.96972573058582978,  0.19606660410267107, -0.01106972353457700,
       0.94858463669396065,  0.19310190509472863,  0.25079641400341668,  0.00017544415292337,
       0.0,                  0.0,                  0.0,                  1.0
    ).finished(),
    (Eigen::Matrix4d() << // Radius
      -0.49809752637334626, -0.08715575919243604,  0.86272981162142748, -0.00403445426854931,
      -0.35916051221787704, -0.88484043330021867, -0.29675096303209120, -0.01277304827290989,
       0.78924177581026822, -0.45766940168277742,  0.40943392394766553, -0.00075658216533172,
       0.0,                  0.0,                  0.0,                  1.0
    ).finished(),
    (Eigen::Matrix4d() << // Manus
       0.25986791443213653, -0.18168016523594308,  0.94839916944740765, -0.00218016198283132,
       0.02524885748749609, -0.98052725973145005, -0.19475314662184390, -0.01176632711677740,
       0.96531402260832611,  0.07455608951235160, -0.25022035742992282,  0.00089467307414822,
       0.0,                  0.0,                  0.0,                  1.0
    ).finished(),

    // Left Wing
    (Eigen::Matrix4d() << // Humerus
       0.28106457057544920,  0.14947796388954818, -0.94797628951291457,  0.00514105438769492,
      -0.14556749017354492, -0.96972573058582978, -0.19606660410267107,  0.01106972353457700,
      -0.94858463669396065,  0.19310190509472863, -0.25079641400341668, -0.00017544415292337,
       0.0,                  0.0,                  0.0,                  1.0
    ).finished(),
    (Eigen::Matrix4d() << // Radius
       0.49809752637334626, -0.08715575919243604, -0.86272981162142748,  0.00403445426854931,
       0.35916051221787704, -0.88484043330021867,  0.29675096303209120,  0.01277304827290989,
      -0.78924177581026822, -0.45766940168277742, -0.40943392394766553,  0.00075658216533172,
       0.0,                  0.0,                  0.0,                  1.0
    ).finished(),
    (Eigen::Matrix4d() << // Manus
      -0.25986791443213653, -0.18168016523594308, -0.94839916944740765,  0.00218016198283132,
      -0.02524885748749609, -0.98052725973145005,  0.19475314662184390,  0.01176632711677740,
      -0.96531402260832611,  0.07455608951235160,  0.25022035742992282, -0.00089467307414822,
       0.0,                  0.0,                  0.0,                  1.0
    ).finished(),

    // Tail
    (Eigen::Matrix4d() << // Right section
      0.0, -0.9659258262890683, -0.2588190451025207, -0.9659258262890683*DY_T/2.0,
      1.0,  0.0,                 0.0,                 0.008,
      0.0, -0.2588190451025207,  0.9659258262890683, -0.2588190451025207*DY_T/2.0,
      0.0,  0.0,                 0.0,                 1.0
    ).finished(),
    (Eigen::Matrix4d() << // Left section
      0.0, -0.9659258262890683,  0.2588190451025207,  0.9659258262890683*DY_T/2.0,
      1.0,  0.0,                 0.0,                 0.008,
      0.0,  0.2588190451025207,  0.9659258262890683, -0.2588190451025207*DY_T/2.0,
      0.0,  0.0,                 0.0,                 1.0
    ).finished()
  };

  inline const std::array<Eigen::Matrix4d, NUM_JOINTS> JOINT_FIXED_TRANSFORM = {
    // Right Wing
    (Eigen::Matrix4d() <<
      1.0,  0.0,  0.0, -0.054,
      0.0, -1.0,  0.0,  0.066,
      0.0,  0.0, -1.0, -0.012,
      0.0,  0.0,  0.0,  1.0
    ).finished(),
    (Eigen::Matrix4d() <<
      0.0, 0.0, -1.0,  0.022,
      -1.0, 0.0,  0.0, -0.006,
       0.0, 1.0,  0.0,  0.0,
       0.0, 0.0,  0.0,  1.0
    ).finished(),
    (Eigen::Matrix4d() <<
      0.0, -1.0, 0.0,  0.006,
      1.0,  0.0, 0.0, -0.006,
      0.0,  0.0, 1.0, -0.020,
      0.0, 0.0,  0.0,  1.0
    ).finished(),
    (Eigen::Matrix4d() <<
       0.94481709128728697, -0.17404583775373195,  0.27754046618832384,  0.010924,
       0.18116301750379044,  0.98345308026815459,  0.0,                 -0.112374,
      -0.27294802637196669,  0.050280068334085472, 0.96071394786792175,  0.017385,
       0.0,                  0.0,                   0.0,                   1.0
    ).finished(),
    (Eigen::Matrix4d() <<
      -0.087155560770988738,  0.22415907121562181,  0.97064752563350420, -0.010737317504428925,
     -0.88484047557676260,   0.43021672398697414, -0.17880409162825162, -0.082573126049754225,
     -0.45766935773305067,  -0.87445198907296517,  0.16084923934713383, -0.038775274662863249,
      0.0,                   0.0,                  0.0,                   1.0
    ).finished(),
    (Eigen::Matrix4d() <<
      -0.087155313452979541, -0.89862859790082117, -0.42996580604940760,  0.013530156304094348,
      -0.37507188516431561,   0.42944876871384535, -0.82151983299823550, -0.00030070200063790631,
       0.92288950167189565,   0.089668266878345199,-0.37447906432647815,  0.00096619961373791731,
       0.0,                   0.0,                  0.0,                   1.0
    ).finished(),

    // Left Wing
    (Eigen::Matrix4d() <<
      -1.0,  0.0,  0.0, -0.054,
       0.0, -1.0,  0.0, -0.066,
       0.0,  0.0,  1.0, -0.012,
       0.0,  0.0,  0.0,  1.0
    ).finished(),
    (Eigen::Matrix4d() <<
       0.0,  0.0,  1.0, -0.022,
      -1.0,  0.0,  0.0,  0.006,
       0.0, -1.0,  0.0,  0.0,
       0.0,  0.0,  0.0,  1.0
    ).finished(),
    (Eigen::Matrix4d() <<
       0.0, -1.0,  0.0, -0.006,
      -1.0,  0.0,  0.0, -0.006,
       0.0,  0.0, -1.0, -0.020,
       0.0,  0.0,  0.0,  1.0
    ).finished(),
    (Eigen::Matrix4d() <<
       0.94481709128728697, -0.17404583775373195,  0.27754046618832384, -0.010924,
       0.18116301750379044,  0.98345308026815459,  0.0,                  0.112374,
      -0.27294802637196669,  0.050280068334085472, 0.96071394786792175, -0.017385,
       0.0,                  0.0,                   0.0,                   1.0
    ).finished(),
    (Eigen::Matrix4d() <<
      -0.087155560770988752,  0.22415907121562181,  0.97064752563350432, 0.010737317504428929,
      -0.88484047557676260,   0.43021672398697414, -0.17880409162825164, 0.082573126049754225,
      -0.45766935773305067,  -0.87445198907296517,  0.16084923934713383, 0.038775274662863249,
      0.0,                   0.0,                  0.0,                  1.0
    ).finished(),
    (Eigen::Matrix4d() <<
      -0.08715531345297961, -0.89862859790082128, -0.42996580604940782, -0.013530156304094388,
      -0.37507188516431550,   0.42944876871384524, -0.82151983299823539, 0.00030070200063790241,
       0.92288950167189554,   0.089668266878345199, -0.37447906432647804, -0.00096619961373792110,
       0.0,                   0.0,                  0.0,                  1.0
    ).finished(),

    // Tail
    (Eigen::Matrix4d() <<
      -1.0,  0.0,  0.0, -0.240,
       0.0,  1.0,  0.0,  0.000,
       0.0,  0.0, -1.0,  0.000,
       0.0,  0.0,  0.0,  1.000
    ).finished(),
    (Eigen::Matrix4d() <<
       0.0,  1.0,  0.0,  0.010,
      -1.0,  0.0,  0.0,  0.000,
       0.0,  0.0,  1.0,  0.000,
       0.0,  0.0,  0.0,  1.000
    ).finished()
  };

  // ----- Servo Motor Parameters -----
  inline constexpr std::array<std::size_t, NUM_JOINTS> MOTOR_MODEL_INDEX = []() {
    std::array<std::size_t, NUM_JOINTS> result{};
    for (std::size_t i=0; i<NUM_WING_JOINTS; ++i) {result[i] = i % NUM_WING_JOINTS_PER_WING;}
    for (std::size_t i=0; i<NUM_TAIL_JOINTS; ++i) {result[NUM_WING_JOINTS+i] = NUM_WING_JOINTS_PER_WING+i;}
    return result;
  }();

  inline constexpr std::array<double, NUM_MOTOR_MODELS> MOTOR_OHM =              {0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500};  // [ohm]
  inline constexpr std::array<double, NUM_MOTOR_MODELS> MOTOR_H =                {0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001};  // [H]
  inline constexpr std::array<double, NUM_MOTOR_MODELS> MOTOR_KT =               {0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050};  // [Nm/A]
  inline constexpr std::array<double, NUM_MOTOR_MODELS> MOTOR_KE =               {0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050};  // [V/(rad/s)]
  inline constexpr std::array<double, NUM_MOTOR_MODELS> MOTOR_REDUCTION_RATIO =  {10.00, 5.000, 5.000, 5.000, 5.000, 5.000, 1.000, 1.000};
  inline constexpr std::array<double, NUM_MOTOR_MODELS> MOTOR_EFFICIENCY =       {0.950, 0.950, 0.950, 0.950, 0.950, 0.950, 0.950, 0.950};
  inline constexpr std::array<double, NUM_MOTOR_MODELS> MOTOR_MAX_TORQUE =       {20.00, 15.00, 15.00, 15.00, 15.00, 15.00, 15.00, 15.00};  // [Nm]
  inline constexpr std::array<double, NUM_MOTOR_MODELS> MOTOR_VISCOUS_FRICTION = {0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002};  // [Nm/(rad/s)]
  inline constexpr std::array<double, NUM_MOTOR_MODELS> MOTOR_ESC_TIME_CONSTANT ={.0003, .0003, .0003, .0003, .0003, .0003, .0003, .0003};  // [s]
  inline constexpr std::array<double, NUM_MOTOR_MODELS> MOTOR_KP =               {100.0, 30.00, 50.00, 30.00, 30.00, 30.00, 1.000, 5.000};  // [Nm/rad]
  inline constexpr std::array<double, NUM_MOTOR_MODELS> MOTOR_KD =               {1.500, 0.500, 1.000, 0.250, 0.200, 0.200, 0.050, 0.300};  // [Nm/(rad/s)]
  inline constexpr std::array<double, NUM_MOTOR_MODELS> MOTOR_ROTOR_INERTIA =    {1.e-6, 1.e-6, 1.e-6, 1.e-6, 1.e-6, 1.e-6, 1.e-6, 1.e-6};  // [kg m^2], motor side

  inline constexpr std::array<const char*, NUM_JOINTS> ACTUATOR_NAMES = {
    "motor_J1", "motor_J2", "motor_J3", "motor_J4", "motor_J5", "motor_J6",
    "motor_J7", "motor_J8", "motor_J9", "motor_J10", "motor_J11", "motor_J12",
    "motor_J13", "motor_J14"
  };

  // world NED -> MuJoCo world FLU
  inline const Eigen::DiagonalMatrix<double, 3> NED_TO_FLU(1.0, -1.0, -1.0);
  inline constexpr std::array<double, 2> STRIP_SPAN_SIGN = {1.0, -1.0};
} // namespace param
