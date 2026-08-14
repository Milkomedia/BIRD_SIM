#pragma once

#include <Eigen/Core>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace param {
  inline constexpr std::chrono::steady_clock::duration SIM_DT_US = std::chrono::microseconds(500);
  inline constexpr std::chrono::steady_clock::duration RENDER_DT_US = std::chrono::microseconds(16667);
  inline constexpr std::chrono::steady_clock::duration SPIN_MARGIN_US = std::chrono::microseconds(150);
  inline constexpr double SIM_DT_SEC = std::chrono::duration<double>(SIM_DT_US).count();

  // ----- body elipsoid -----
  inline const Eigen::Vector3d ELIPSOID_CENTER_POS = Eigen::Vector3d(-0.084, 0.0, 0.0); // [m], body FRD
  inline const Eigen::Vector3d ELIPSOID_SIZE = Eigen::Vector3d(0.156, 0.06, 0.06);       // [m], semi-axes x,y,z

  // ----- MST -----
  inline constexpr double AIR_DENSITY = 1.225;               // [kg/m^3]
  inline constexpr double AIR_KINEMATIC_VISCOSITY = 1.5e-5;  // [m^2/s]

  // ----- wing kinematics -----
  inline constexpr double KIN_GAIN = 2.2;

  const std::array<double, 12> INITIAL_DES_THETA{
    0.12, -0.1, 0.1963495408, -0.7072074129, 0.19, 0.5173155903,
    0.12, -0.1, 0.1963495408, -0.7072074129, 0.19, 0.5173155903
  };

  inline constexpr std::size_t NH = 7;  // number of humerus strip frame
  inline constexpr std::size_t NR = 6;  // number of radius strip frame
  inline constexpr std::size_t NM = 25; // number of manus strip frame

  inline constexpr double HUMERUS_LENGTH = 0.100; // [m]
  inline constexpr double RADIUS_LENGTH  = 0.086; // [m]
  inline constexpr double MANUS_LENGTH   = 0.420; // [m]

  inline constexpr double L_ROOT = 0.217; // [m]
  inline constexpr double L_TRI  = 0.200; // [m]
  inline constexpr double L_SEC  = 0.215; // [m]
  inline constexpr double L_MPRI = 0.172; // [m]
  inline constexpr double L_LPRI = 0.049; // [m]
  inline constexpr double D_LPRI = 0.017; // [m]
  inline constexpr double S_MRPI = 0.189; // [m]

  inline constexpr double DY_H = HUMERUS_LENGTH / static_cast<double>(NH-1); // initial humerus strip width [m]
  inline constexpr double DY_R = RADIUS_LENGTH / static_cast<double>(NR-1); // initial radius strip width [m]
  inline constexpr double DY_M = MANUS_LENGTH / static_cast<double>(NM-1); // initial manus strip width [m]

  constexpr std::size_t DECLINE_IDX = static_cast<std::size_t>(param::S_MRPI / param::DY_M);
  inline constexpr double DL_H = (L_TRI - L_ROOT) / static_cast<double>(NH-1);
  inline constexpr double DL_R = (L_SEC - L_TRI) / static_cast<double>(NR-1);
  inline constexpr double DL_M1 = (L_MPRI - L_SEC) / static_cast<double>(DECLINE_IDX-1);
  inline constexpr double DL_M2 = (L_LPRI - L_MPRI) / static_cast<double>(NM-DECLINE_IDX-1);

  inline const std::array<Eigen::Matrix4d, 6> J_T_S0 = {
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
    ).finished()
  };

  inline const std::array<Eigen::Matrix4d, 12> WING_FIXED_TRANSFORM = {
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
    ).finished()
  };

  // ----- Servo Motor Parameters -----
  inline constexpr std::array<double, 6> MOTOR_OHM =              {0.500, 0.500, 0.500, 0.500, 0.500, 0.500};  // [Ω]
  inline constexpr std::array<double, 6> MOTOR_H =                {0.001, 0.001, 0.001, 0.001, 0.001, 0.001};  // [H]
  inline constexpr std::array<double, 6> MOTOR_KT =               {0.050, 0.050, 0.050, 0.050, 0.050, 0.050};  // [Nm/A]
  inline constexpr std::array<double, 6> MOTOR_KE =               {0.050, 0.050, 0.050, 0.050, 0.050, 0.050};  // [V/(rad/s)]
  inline constexpr std::array<double, 6> MOTOR_REDUCTION_RATIO =  {9.000, 5.000, 5.000, 5.000, 5.000, 5.000};
  inline constexpr std::array<double, 6> MOTOR_EFFICIENCY =       {0.950, 0.950, 0.950, 0.950, 0.950, 0.950};
  inline constexpr std::array<double, 6> MOTOR_MAX_VOLTAGE =      {24.00, 24.00, 24.00, 24.00, 24.00, 24.00};  // [V]
  inline constexpr std::array<double, 6> MOTOR_MAX_CURRENT =      {100.0, 100.0, 100.0, 100.0, 100.0, 100.0};  // [A]
  inline constexpr std::array<double, 6> MOTOR_MAX_TORQUE =       {50.00, 50.00, 50.00, 50.00, 50.00, 100.00};  // [Nm]
  inline constexpr std::array<double, 6> MOTOR_VISCOUS_FRICTION = {0.002, 0.002, 0.002, 0.002, 0.002, 0.002};  // [Nm/(rad/s)]
  inline constexpr std::array<double, 6> MOTOR_KP =               {500.0, 100.0, 100.0, 100.0, 100.0, 50.00};  // [Nm/rad]
  inline constexpr std::array<double, 6> MOTOR_KD =               {10.00, 0.500, 0.500, 0.500, 0.500, 0.300};  // [Nm/(rad/s)]
  inline constexpr std::array<double, 6> MOTOR_DT = {SIM_DT_SEC, SIM_DT_SEC, SIM_DT_SEC, SIM_DT_SEC, SIM_DT_SEC, SIM_DT_SEC}; // [sec]

  // world NED -> MuJoCo world FLU
  inline const Eigen::DiagonalMatrix<double, 3> NED_TO_FLU(1.0, -1.0, -1.0);
  inline constexpr std::array<double, 2> STRIP_SPAN_SIGN = {1.0, -1.0};
} // namespace param
