#include "MST.hpp"

#include "coeff/coeff.hpp"
#include "utils.hpp" // State

#include <algorithm>
#include <cmath>

MST::MST() {reset();}

void MST::reset() {
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  strip_state_.reset();
  aero_telemetry_.reset();
  for (DynamicStallState& state : dynamic_stall_state_) {state = {};}
  for (std::array<double, 2>& state : wagner_state_) {state = {};}
  aero_pos_.fill(zero);
  aero_force_.fill(zero);
  aero_torque_.fill(zero);
  added_mass_pos_.fill(zero);
  for (Eigen::Matrix<double, 6, 6>& matrix : added_mass_matrix_) {matrix.setZero();}
  tail_chord_.fill(0.0);
  tail_width_.fill(0.0);
  wing_circulation_.fill(0.0);
  for (std::array<Eigen::Vector3d, WAKE_SPAN_NODES>& row : bound_wake_nodes_) {row.fill(zero);}
  for (std::array<Eigen::Vector3d, WAKE_SPAN_NODES>& row : trailing_edge_wake_nodes_) {row.fill(zero);}
  for (std::array<double, param::WAKE_SPAN_PANELS>& row : bound_wake_gamma_) {row.fill(0.0);}
  for (std::array<double, param::WAKE_SPAN_PANELS>& row : wake_gamma_sum_) {row.fill(0.0);}
  for (auto& wing : wake_nodes_) {
    for (std::array<Eigen::Vector3d, WAKE_SPAN_NODES>& row : wing) {row.fill(zero);}
  }
  for (auto& wing : wake_gamma_) {
    for (std::array<double, param::WAKE_SPAN_PANELS>& row : wing) {row.fill(0.0);}
  }
  tail_wake_velocity_world_.fill(zero);
  wake_valid_cells_ = 0;
  wake_sample_count_ = 0;
  wake_initialized_ = false;
  aero_pos_.back() = param::ELLIPSOID_CENTER_POS;
}

void MST::update_body_elipsoid(const State& s) {
  constexpr double PI = 3.14159265358979323846;
  constexpr double EPS = 1e-14;
  constexpr double MIN = 1e-15;
  constexpr double RHO = param::AIR_DENSITY;
  constexpr double VISCOSITY = param::AIR_DENSITY * param::AIR_KINEMATIC_VISCOSITY;
  constexpr double BLUNT_DRAG_COEF = 0.5;
  constexpr double SLENDER_DRAG_COEF = 0.25;
  constexpr double ANGULAR_DRAG_COEF = 1.5;
  constexpr double KUTTA_LIFT_COEF = 1.0;
  constexpr double MAGNUS_LIFT_COEF = 1.0;

  // Static geometry terms are evaluated once. Layout:
  // [0] V, [1] Amax, [2] linear viscous force coef, [3] linear viscous torque coef,
  // [4:6] projected-area factors, [7:9] rho*virtual mass,
  // [10:12] rho*virtual inertia, [13:15] angular quadratic-drag factors.
  static const std::array<double, 16> fluid = []() {
    constexpr double PI = 3.14159265358979323846;
    constexpr double EPS = 1e-14;
    constexpr double RHO = param::AIR_DENSITY;
    constexpr double VISCOSITY = param::AIR_DENSITY * param::AIR_KINEMATIC_VISCOSITY;
    constexpr double SLENDER_DRAG_COEF = 0.25;
    constexpr double ANGULAR_DRAG_COEF = 1.5;

    const auto kappa = [](const double dx, const double dy, const double dz) {
      static constexpr double W[15] = {
        0.01146766, 0.03154605, 0.05239501, 0.07032663, 0.08450236,
        0.09517529, 0.10221647, 0.10474107, 0.10221647, 0.09517529,
        0.08450236, 0.07032663, 0.05239501, 0.03154605, 0.01146766
      };
      static constexpr double L[15] = {
        7.865151709349917e-08, 1.7347976913907274e-05, 0.0003548008144506193,
        0.002846636252924549, 0.014094260903596077, 0.053063261727396636,
        0.17041978741317773, 0.5, 1.4036301548686991, 3.9353484827022642,
        11.644841677041734, 39.53187807410903, 177.5711362220801,
        1429.4772912937397, 54087.416549217705
      };
      static constexpr double D[15] = {
        5.538677720489877e-05, 0.002080868285293228, 0.016514126520723166,
        0.07261900344370877, 0.23985243401862602, 0.6868318249020725,
        1.8551129519182894, 5.0, 14.060031152313941, 43.28941239611009,
        156.58546376397112, 747.9826085305024, 5827.4042950027115,
        116754.0197944512, 25482945.327264845
      };

      const double inv_dx2 = 1.0/(dx*dx);
      const double inv_dy2 = 1.0/(dy*dy);
      const double inv_dz2 = 1.0/(dz*dz);
      const double scale = std::pow(dx*dx*dx*dy*dz, 0.4);
      double result = 0.0;
      for (std::size_t i=0; i<15; ++i) {
        const double lambda = scale*L[i];
        const double denom = (1.0 + lambda*inv_dx2)*std::sqrt((1.0 + lambda*inv_dx2)*(1.0 + lambda*inv_dy2)*(1.0 + lambda*inv_dz2));
        result += scale*D[i]/denom*W[i];
      }
      return result*inv_dx2;
    };

    const double dx = param::ELLIPSOID_SIZE.x();
    const double dy = param::ELLIPSOID_SIZE.y();
    const double dz = param::ELLIPSOID_SIZE.z();
    const double dx2 = dx*dx;
    const double dy2 = dy*dy;
    const double dz2 = dz*dz;
    const double volume = (4.0/3.0)*PI*dx*dy*dz;
    const double kx = kappa(dx, dy, dz);
    const double ky = kappa(dy, dz, dx);
    const double kz = kappa(dz, dx, dy);

    const double Ixfac = (dy2-dz2)*(dy2-dz2)*std::abs(kz-ky) / std::max(EPS, std::abs(2.0*(dy2-dz2) + (dy2+dz2)*(ky-kz)));
    const double Iyfac = (dz2-dx2)*(dz2-dx2)*std::abs(kx-kz) / std::max(EPS, std::abs(2.0*(dz2-dx2) + (dz2+dx2)*(kz-kx)));
    const double Izfac = (dx2-dy2)*(dx2-dy2)*std::abs(ky-kx) / std::max(EPS, std::abs(2.0*(dx2-dy2) + (dx2+dy2)*(kx-ky)));

    const double d_max = std::max({dx, dy, dz});
    const double d_min = std::min({dx, dy, dz});
    const double d_mid = dx + dy + dz - d_max - d_min;
    const double eq_sphere_D = (2.0/3.0)*(dx + dy + dz);
    const double d_max2 = d_max*d_max;
    const double I_max = (8.0/15.0)*PI*d_mid*d_max2*d_max2;

    const double max_yz = std::max(dy, dz);
    const double max_zx = std::max(dz, dx);
    const double max_xy = std::max(dx, dy);
    const double max_yz2 = max_yz*max_yz;
    const double max_zx2 = max_zx*max_zx;
    const double max_xy2 = max_xy*max_xy;
    const double IIx = (8.0/15.0)*PI*dx*max_yz2*max_yz2;
    const double IIy = (8.0/15.0)*PI*dy*max_zx2*max_zx2;
    const double IIz = (8.0/15.0)*PI*dz*max_xy2*max_xy2;

    const double pyz = dy*dz;
    const double pzx = dz*dx;
    const double pxy = dx*dy;

    std::array<double, 16> out{};
    out[0] = volume;
    out[1] = PI*d_max*d_mid;
    out[2] = VISCOSITY*3.0*PI*eq_sphere_D;
    out[3] = VISCOSITY*PI*eq_sphere_D*eq_sphere_D*eq_sphere_D;
    out[4] = pyz*pyz;
    out[5] = pzx*pzx;
    out[6] = pxy*pxy;
    out[7] = RHO*volume*kx/std::max(EPS, 2.0-kx);
    out[8] = RHO*volume*ky/std::max(EPS, 2.0-ky);
    out[9] = RHO*volume*kz/std::max(EPS, 2.0-kz);
    out[10] = RHO*volume*Ixfac/5.0;
    out[11] = RHO*volume*Iyfac/5.0;
    out[12] = RHO*volume*Izfac/5.0;
    out[13] = ANGULAR_DRAG_COEF*IIx + SLENDER_DRAG_COEF*(I_max-IIx);
    out[14] = ANGULAR_DRAG_COEF*IIy + SLENDER_DRAG_COEF*(I_max-IIy);
    out[15] = ANGULAR_DRAG_COEF*IIz + SLENDER_DRAG_COEF*(I_max-IIz);
    return out;
  }();

  const Eigen::Vector3d lin_vel = s.R.transpose()*(s.vel-s.vel_f) + s.w.cross(param::ELLIPSOID_CENTER_POS);
  const Eigen::Vector3d& ang_vel = s.w;
  const double lin_speed = lin_vel.norm();

  // MuJoCo's stateless ellipsoid model uses only the velocity-dependent added-mass bias.
  const Eigen::Vector3d virtual_lin_mom(fluid[7]*lin_vel.x(), fluid[8]*lin_vel.y(), fluid[9]*lin_vel.z());
  const Eigen::Vector3d virtual_ang_mom(fluid[10]*ang_vel.x(), fluid[11]*ang_vel.y(), fluid[12]*ang_vel.z());
  Eigen::Vector3d force = virtual_lin_mom.cross(ang_vel);
  Eigen::Vector3d torque = virtual_lin_mom.cross(lin_vel) + virtual_ang_mom.cross(ang_vel);

  force += MAGNUS_LIFT_COEF*RHO*fluid[0]*ang_vel.cross(lin_vel);

  const double vx2 = lin_vel.x()*lin_vel.x();
  const double vy2 = lin_vel.y()*lin_vel.y();
  const double vz2 = lin_vel.z()*lin_vel.z();
  const double proj_denom = fluid[4]*fluid[4]*vx2 + fluid[5]*fluid[5]*vy2 + fluid[6]*fluid[6]*vz2;
  const double proj_num = fluid[4]*vx2 + fluid[5]*vy2 + fluid[6]*vz2;
  const double A_proj = PI*std::sqrt(proj_denom/std::max(MIN, proj_num));

  const Eigen::Vector3d normal(fluid[4]*lin_vel.x(), fluid[5]*lin_vel.y(), fluid[6]*lin_vel.z());
  const double cos_alpha = proj_num/std::max(MIN, lin_speed*proj_denom);
  const Eigen::Vector3d kutta_circ = KUTTA_LIFT_COEF*RHO*cos_alpha*A_proj*normal.cross(lin_vel);
  force += kutta_circ.cross(lin_vel);

  const Eigen::Vector3d mom_visc(fluid[13]*ang_vel.x(), fluid[14]*ang_vel.y(), fluid[15]*ang_vel.z());
  const double drag_lin_coef = fluid[2] + RHO*lin_speed*(A_proj*BLUNT_DRAG_COEF + SLENDER_DRAG_COEF*(fluid[1]-A_proj));
  const double drag_ang_coef = fluid[3] + RHO*mom_visc.norm();
  force -= drag_lin_coef*lin_vel;
  torque -= drag_ang_coef*ang_vel;

  aero_pos_.back() = param::ELLIPSOID_CENTER_POS;
  aero_force_.back() = force;
  aero_torque_.back() = torque;
}

void MST::update_atan2_dot_ddot(double& angle_dot, double& angle_ddot, const double y, const double x, const double y_dot, const double x_dot, const double y_ddot, const double x_ddot) {
  constexpr double EPS2 = 1e-16;

  const double r2 = x*x + y*y;
  if (r2 <= EPS2) {angle_dot = 0.0; angle_ddot = 0.0; return;}

  const double q = x*y_dot - y*x_dot;
  const double inv_r2 = 1.0 / r2;
  angle_dot = q * inv_r2;
  angle_ddot = (x*y_ddot - y*x_ddot)*inv_r2 - 2.0*q*(x*x_dot + y*y_dot)*inv_r2*inv_r2;
}

void MST::update_relative_vector_dot_ddot(Eigen::Vector3d& x_dot_0, Eigen::Vector3d& x_ddot_0, const Eigen::Vector3d& x_0, const Eigen::Matrix3d& bR0, const Eigen::Vector3d& b_omega_0, const Eigen::Vector3d& b_omega_dot_0, const Eigen::Vector3d& b_omega_x, const Eigen::Vector3d& b_omega_dot_x) {
  const Eigen::Vector3d b_omega_rel = b_omega_x - b_omega_0;
  const Eigen::Vector3d omega_rel_0 = bR0.transpose() * b_omega_rel;
  const Eigen::Vector3d omega_dot_rel_0 = bR0.transpose() * (b_omega_dot_x - b_omega_dot_0 - b_omega_0.cross(b_omega_rel));

  x_dot_0 = omega_rel_0.cross(x_0);
  x_ddot_0 = omega_dot_rel_0.cross(x_0) + omega_rel_0.cross(x_dot_0);
}

void MST::update_Rz_psi(Eigen::Matrix3d& bRsi, double& sin_psi, double& cos_psi, double& psi_dot, double& psi_ddot, const Eigen::Vector3d& n, const Eigen::Matrix3d& bRs0, const Eigen::Vector3d& b_omega_n, const Eigen::Vector3d& b_omega_dot_n, const Eigen::Vector3d& b_omega_s0, const Eigen::Vector3d& b_omega_dot_s0) {
  constexpr double EPS2 = 1e-16;

  const Eigen::Vector3d n_s0 = bRs0.transpose() * n;
  const double a = n_s0(0);
  const double b = n_s0(1);
  const double r2 = a * a + b * b;

  if (r2 <= EPS2) {
    cos_psi = 1.0;
    sin_psi = 0.0;
    psi_dot = 0.0;
    psi_ddot = 0.0;
  }
  else {
    const double inv_r = 1.0 / std::sqrt(r2);
    cos_psi =  b * inv_r;
    sin_psi = -a * inv_r;

    Eigen::Vector3d n_dot_s0;
    Eigen::Vector3d n_ddot_s0;
    update_relative_vector_dot_ddot(n_dot_s0, n_ddot_s0, n_s0, bRs0, b_omega_s0, b_omega_dot_s0, b_omega_n, b_omega_dot_n);
    update_atan2_dot_ddot(psi_dot, psi_ddot, -a, b, -n_dot_s0(0), n_dot_s0(1), -n_ddot_s0(0), n_ddot_s0(1));
  }

  bRsi.col(0) =  cos_psi*bRs0.col(0) + sin_psi*bRs0.col(1);
  bRsi.col(1) = -sin_psi*bRs0.col(0) + cos_psi*bRs0.col(1);
  bRsi.col(2) = bRs0.col(2);
}

void MST::update_Ryphi_Rzpsi(Eigen::Matrix3d& bRri, double& sin_psi, double& cos_psi, double& sin_phi, double& cos_phi, double& psi_dot, double& psi_ddot, double& phi_dot, double& phi_ddot, const Eigen::Vector3d& xi, const Eigen::Matrix3d& bRr0, const Eigen::Vector3d& b_omega_xi, const Eigen::Vector3d& b_omega_dot_xi, const Eigen::Vector3d& b_omega_r0, const Eigen::Vector3d& b_omega_dot_r0, const bool initialized) {
  constexpr double EPS2 = 1e-16;

  const double x = bRr0.col(0).dot(xi);
  const double y = bRr0.col(1).dot(xi);
  const double z = bRr0.col(2).dot(xi);
  const double rho2 = x*x + z*z;

  if (rho2 <= EPS2) {
    if (!initialized) {sin_phi = 0.0; cos_phi = 1.0;}
    sin_psi = y >= 0.0 ? 1.0 : -1.0;
    cos_psi = 0.0;

    psi_dot = 0.0;
    psi_ddot = 0.0;
    phi_dot = 0.0;
    phi_ddot = 0.0;
  }
  else {
    const double rho = std::sqrt(rho2);
    const double inv_rho = 1.0 / rho;
    const double candidate_sin_phi = -z * inv_rho;
    const double candidate_cos_phi =  x * inv_rho;
    const double branch = initialized && candidate_sin_phi*sin_phi + candidate_cos_phi*cos_phi < 0.0 ? -1.0 : 1.0;

    sin_psi = y;
    cos_psi = branch * rho;
    sin_phi = branch * candidate_sin_phi;
    cos_phi = branch * candidate_cos_phi;
    const Eigen::Vector3d xi_r0(x, y, z);
    Eigen::Vector3d xi_dot_r0;
    Eigen::Vector3d xi_ddot_r0;
    update_relative_vector_dot_ddot(xi_dot_r0, xi_ddot_r0, xi_r0, bRr0, b_omega_r0, b_omega_dot_r0, b_omega_xi, b_omega_dot_xi);

    update_atan2_dot_ddot(phi_dot, phi_ddot, -z, x, -xi_dot_r0(2), xi_dot_r0(0), -xi_ddot_r0(2), xi_ddot_r0(0));

    const double rho_dot = (x*xi_dot_r0(0) + z*xi_dot_r0(2)) * inv_rho;
    const double rho_ddot = (xi_dot_r0(0)*xi_dot_r0(0) + x*xi_ddot_r0(0) + xi_dot_r0(2)*xi_dot_r0(2) + z*xi_ddot_r0(2) - rho_dot*rho_dot) * inv_rho;
    update_atan2_dot_ddot(psi_dot, psi_ddot, y, branch*rho, xi_dot_r0(1), branch*rho_dot, xi_ddot_r0(1), branch*rho_ddot);
  }

  bRri.col(0) = cos_psi*(cos_phi*bRr0.col(0) - sin_phi*bRr0.col(2)) + sin_psi*bRr0.col(1);
  bRri.col(2) = sin_phi*bRr0.col(0) + cos_phi*bRr0.col(2);
  bRri.col(1) = bRri.col(2).cross(bRri.col(0));
}

void MST::update_radius_strip_rotation(StripRotation<param::NR>& rotation, const Eigen::Vector3d& n, const Eigen::Matrix3d& bRhi, const Eigen::Matrix3d& bRmi, const Eigen::Matrix3d& bRr0, const Eigen::Vector3d& b_omega_n, const Eigen::Vector3d& b_omega_dot_n, const Eigen::Vector3d& b_omega_hi, const Eigen::Vector3d& b_omega_dot_hi, const Eigen::Vector3d& b_omega_mi, const Eigen::Vector3d& b_omega_dot_mi, const Eigen::Vector3d& b_omega_r0, const Eigen::Vector3d& b_omega_dot_r0) {
  constexpr std::size_t k = param::NR-1;
  constexpr double inv_k = 1.0 / static_cast<double>(k);
  constexpr double EPS2 = 1e-16;

  // Radius root strip
  update_Ryphi_Rzpsi(rotation.bRri[0], rotation.sin_psi[0], rotation.cos_psi[0], rotation.sin_phi[0], rotation.cos_phi[0], rotation.psi_dot[0], rotation.psi_ddot[0], rotation.phi_dot[0], rotation.phi_ddot[0], bRhi.col(0), bRr0, b_omega_hi, b_omega_dot_hi, b_omega_r0, b_omega_dot_r0, rotation.initialized);
 
  // Radius tip strip
  update_Ryphi_Rzpsi(rotation.bRri[k], rotation.sin_psi[k], rotation.cos_psi[k], rotation.sin_phi[k], rotation.cos_phi[k], rotation.psi_dot[k], rotation.psi_ddot[k], rotation.phi_dot[k], rotation.phi_ddot[k], bRmi.col(0), bRr0, b_omega_mi, b_omega_dot_mi, b_omega_r0, b_omega_dot_r0, rotation.initialized);

  // Interpolate phi linearly in angle space.
  const double phi0 = std::atan2(rotation.sin_phi[0], rotation.cos_phi[0]);
  const double phik = std::atan2(rotation.sin_phi[k], rotation.cos_phi[k]);

  double delta_phi = phik-phi0;
  if (delta_phi > M_PI) {delta_phi -= 2.0*M_PI;}
  else if (delta_phi < -M_PI) {delta_phi += 2.0*M_PI;}

  const double dphi = delta_phi * inv_k;
  const double dphi_dot = (rotation.phi_dot[k] - rotation.phi_dot[0]) * inv_k;
  const double dphi_ddot = (rotation.phi_ddot[k] - rotation.phi_ddot[0]) * inv_k;
  const double sin_dphi = std::sin(dphi);
  const double cos_dphi = std::cos(dphi);
  Eigen::Vector3d chord_ref = bRhi.col(0);
  const Eigen::Vector3d dchord_ref = (bRmi.col(0) - bRhi.col(0)) * inv_k;

  const Eigen::Vector3d n_r0 = bRr0.transpose() * n;
  Eigen::Vector3d n_dot_r0;
  Eigen::Vector3d n_ddot_r0;
  update_relative_vector_dot_ddot(n_dot_r0, n_ddot_r0, n_r0, bRr0, b_omega_r0, b_omega_dot_r0, b_omega_n, b_omega_dot_n); 

  double sin_phi = rotation.sin_phi[0];
  double cos_phi = rotation.cos_phi[0];
  double phi_dot = rotation.phi_dot[0];
  double phi_ddot = rotation.phi_ddot[0];

  for (std::size_t i=1; i<k; ++i) {
    const double next_cos_phi = cos_phi*cos_dphi - sin_phi*sin_dphi;
    sin_phi = sin_phi*cos_dphi + cos_phi*sin_dphi;
    cos_phi = next_cos_phi;

    rotation.sin_phi[i] = sin_phi;
    rotation.cos_phi[i] = cos_phi;
    phi_dot += dphi_dot;
    phi_ddot += dphi_ddot;
    rotation.phi_dot[i] = phi_dot;
    rotation.phi_ddot[i] = phi_ddot;

    chord_ref += dchord_ref;

    const Eigen::Vector3d u = cos_phi*bRr0.col(0) - sin_phi*bRr0.col(2);
    const double n_phi_z = sin_phi*n_r0(0) + cos_phi*n_r0(2);
    const double a = cos_phi*n_r0(0) - sin_phi*n_r0(2);
    const double b = n_r0(1);
    const double r2 = a*a + b*b;
    const double chord_ref2 = chord_ref.squaredNorm();
    
    const double n_dot_phi_z = sin_phi*n_dot_r0(0) + cos_phi*n_dot_r0(2);
    const double a_dot = cos_phi*n_dot_r0(0) - sin_phi*n_dot_r0(2) - phi_dot*n_phi_z;
    const double a_ddot = cos_phi*n_ddot_r0(0) - sin_phi*n_ddot_r0(2) - 2.0*phi_dot*n_dot_phi_z - phi_ddot*n_phi_z - phi_dot*phi_dot*a;
    const double b_dot = n_dot_r0(1);
    const double b_ddot = n_ddot_r0(1);

    double cos_psi;
    double sin_psi;
    if (r2 <= EPS2) {
      const double du = chord_ref.dot(u);
      const double dv = chord_ref.dot(bRr0.col(1));
      const double q2 = du*du + dv*dv;

      if (chord_ref2 <= EPS2 || q2 <= EPS2*chord_ref2) {
        cos_psi = rotation.initialized ? rotation.cos_psi[i] : 1.0;
        sin_psi = rotation.initialized ? rotation.sin_psi[i] : 0.0;
        rotation.psi_dot[i] = 0.0;
        rotation.psi_ddot[i] = 0.0;
      }
      else {
        const double inv_q = 1.0 / std::sqrt(q2);
        cos_psi = du * inv_q;
        sin_psi = dv * inv_q;

        const Eigen::Vector3d bRr0_y = bRr0.col(1);
        const Eigen::Vector3d b_omega_phi = b_omega_r0 + phi_dot*bRr0_y;
        const Eigen::Vector3d b_omega_dot_phi = b_omega_dot_r0 + phi_ddot*bRr0_y + phi_dot*b_omega_r0.cross(bRr0_y);
        const Eigen::Vector3d u_dot = b_omega_phi.cross(u);
        const Eigen::Vector3d v_dot = b_omega_r0.cross(bRr0_y);
        const Eigen::Vector3d u_ddot = b_omega_dot_phi.cross(u) + b_omega_phi.cross(u_dot);
        const Eigen::Vector3d v_ddot = b_omega_dot_r0.cross(bRr0_y) + b_omega_r0.cross(v_dot);
        const double lambda = static_cast<double>(i) * inv_k;
        const Eigen::Vector3d chord_ref_dot_h = b_omega_hi.cross(bRhi.col(0));
        const Eigen::Vector3d chord_ref_dot_m = b_omega_mi.cross(bRmi.col(0));
        const Eigen::Vector3d chord_ref_ddot_h = b_omega_dot_hi.cross(bRhi.col(0)) + b_omega_hi.cross(chord_ref_dot_h);
        const Eigen::Vector3d chord_ref_ddot_m = b_omega_dot_mi.cross(bRmi.col(0)) + b_omega_mi.cross(chord_ref_dot_m);
        const Eigen::Vector3d chord_ref_dot = chord_ref_dot_h + lambda*(chord_ref_dot_m - chord_ref_dot_h);
        const Eigen::Vector3d chord_ref_ddot = chord_ref_ddot_h + lambda*(chord_ref_ddot_m - chord_ref_ddot_h);
        const double du_dot = chord_ref_dot.dot(u) + chord_ref.dot(u_dot);
        const double dv_dot = chord_ref_dot.dot(bRr0_y) + chord_ref.dot(v_dot);
        const double du_ddot = chord_ref_ddot.dot(u) + 2.0*chord_ref_dot.dot(u_dot) + chord_ref.dot(u_ddot);
        const double dv_ddot = chord_ref_ddot.dot(bRr0_y) + 2.0*chord_ref_dot.dot(v_dot) + chord_ref.dot(v_ddot);
        update_atan2_dot_ddot(rotation.psi_dot[i], rotation.psi_ddot[i], dv, du, dv_dot, du_dot, dv_ddot, du_ddot);      
      }
    }
    else {
      const double inv_r = 1.0 / std::sqrt(r2);
      cos_psi =  b * inv_r;
      sin_psi = -a * inv_r;
      update_atan2_dot_ddot(rotation.psi_dot[i], rotation.psi_ddot[i], -a, b, -a_dot, b_dot, -a_ddot, b_ddot);

      // Select the solution pointing toward the interpolated H-M chord.
      const Eigen::Vector3d candidate_x = cos_psi*u + sin_psi*bRr0.col(1);
      double score = chord_ref.dot(candidate_x);
      if (rotation.initialized && (chord_ref2 <= EPS2 || score*score <= EPS2*chord_ref2)) {score = rotation.bRri[i].col(0).dot(candidate_x);}

      if (score < 0.0) {
        cos_psi = -cos_psi;
        sin_psi = -sin_psi;
      }
    }

    rotation.sin_psi[i] = sin_psi;
    rotation.cos_psi[i] = cos_psi;
    rotation.bRri[i].col(0) =  cos_psi*u + sin_psi*bRr0.col(1);
    rotation.bRri[i].col(1) = -sin_psi*u + cos_psi*bRr0.col(1);
    rotation.bRri[i].col(2) =  sin_phi*bRr0.col(0) + cos_phi*bRr0.col(2);
  }
  rotation.initialized = true;
}


void MST::update_humerus_stream_p_v_a(const std::size_t idx0, const Eigen::Matrix3d& bRh0, const Eigen::Vector3d& bph0, const Eigen::Matrix3d& bRhi, const Eigen::Vector3d& RtVrel, const Eigen::Vector3d& RtArel, const Eigen::Vector3d& bvh0, const Eigen::Vector3d& bah0, const Eigen::Vector3d& omega, const Eigen::Vector3d& omega_dot, const double omega2, const double dy) {
  std::array<Eigen::Vector3d, 2*param::NH>& p = strip_state_.p_h;
  std::array<Eigen::Vector3d, 2*param::NH>& v = strip_state_.v_h;
  std::array<Eigen::Vector3d, 2*param::NH>& a = strip_state_.a_h;
  const Eigen::Vector3d drho = dy * bRh0.col(1);
  const Eigen::Vector3d dv_local = bRhi.transpose() * (-omega.cross(drho));
  const Eigen::Vector3d da_local = bRhi.transpose() * (-omega_dot.cross(drho) - omega*omega.dot(drho) + omega2*drho);

  p[idx0] = bph0;
  v[idx0] = bRhi.transpose() * (RtVrel - bvh0);
  a[idx0] = bRhi.transpose() * (RtArel - bah0);
  for (std::size_t i=1; i<param::NH; ++i) {
    p[idx0+i] = p[idx0+i-1] + drho;
    v[idx0+i] = v[idx0+i-1] + dv_local;
    a[idx0+i] = a[idx0+i-1] + da_local;
  }
}

void MST::update_radius_stream_p_v_a(const std::size_t idx0, const Eigen::Matrix3d& bRr0, const Eigen::Vector3d& bpr0, const std::array<Eigen::Matrix3d, param::NR>& bRri, const Eigen::Vector3d& RtVrel, const Eigen::Vector3d& RtArel, const Eigen::Vector3d& bvr0, const Eigen::Vector3d& bar0, const Eigen::Vector3d& omega, const Eigen::Vector3d& omega_dot, const double omega2, const double dy) {
  std::array<Eigen::Vector3d, 2*param::NR>& p = strip_state_.p_r;
  std::array<Eigen::Vector3d, 2*param::NR>& v = strip_state_.v_r;
  std::array<Eigen::Vector3d, 2*param::NR>& a = strip_state_.a_r;
  const Eigen::Vector3d drho = dy * bRr0.col(1);
  const Eigen::Vector3d dv_body = -omega.cross(drho);
  const Eigen::Vector3d da_body = -omega_dot.cross(drho) - omega*omega.dot(drho) + omega2*drho;
  Eigen::Vector3d v_body = RtVrel - bvr0;
  Eigen::Vector3d a_body = RtArel - bar0;

  p[idx0] = bpr0;
  v[idx0] = bRri[0].transpose() * v_body;
  a[idx0] = bRri[0].transpose() * a_body;

  for (std::size_t i=1; i<param::NR; ++i) {
    p[idx0+i] = p[idx0+i-1] + drho;
    v_body += dv_body;
    a_body += da_body;
    v[idx0+i] = bRri[i].transpose() * v_body;
    a[idx0+i] = bRri[i].transpose() * a_body;
  }
}

void MST::update_manus_stream_p_v_a(const std::size_t idx0, const Eigen::Matrix3d& bRm0, const Eigen::Vector3d& bpm0, const Eigen::Matrix3d& bRmi, const Eigen::Vector3d& RtVrel, const Eigen::Vector3d& RtArel, const Eigen::Vector3d& bvm0, const Eigen::Vector3d& bam0, const Eigen::Vector3d& omega, const Eigen::Vector3d& omega_dot, const double omega2, const double dy) {
  std::array<Eigen::Vector3d, 2*param::NM>& p = strip_state_.p_m;
  std::array<Eigen::Vector3d, 2*param::NM>& v = strip_state_.v_m;
  std::array<Eigen::Vector3d, 2*param::NM>& a = strip_state_.a_m;
  constexpr double dx = param::D_P / (static_cast<double>(param::NM - param::DECLINE_IDX_K));

  const Eigen::Vector3d drho_y  = dy * bRm0.col(1);
  const Eigen::Vector3d drho_xy = drho_y + dx * bRm0.col(0);

  const Eigen::Vector3d dv_y = bRmi.transpose() * (-omega.cross(drho_y));
  const Eigen::Vector3d dv_xy = bRmi.transpose() * (-omega.cross(drho_xy));
  const Eigen::Vector3d da_y = bRmi.transpose() * (-omega_dot.cross(drho_y) - omega*omega.dot(drho_y) + omega2*drho_y);
  const Eigen::Vector3d da_xy = bRmi.transpose() * (-omega_dot.cross(drho_xy) - omega*omega.dot(drho_xy) + omega2*drho_xy);

  p[idx0] = bpm0;
  v[idx0] = bRmi.transpose() * (RtVrel - bvm0);
  a[idx0] = bRmi.transpose() * (RtArel - bam0);
  for (std::size_t i=1; i<param::DECLINE_IDX_K; ++i) {
    p[idx0+i] = p[idx0+i-1] + drho_y;
    v[idx0+i] = v[idx0+i-1] + dv_y;
    a[idx0+i] = a[idx0+i-1] + da_y;
  }
  for (std::size_t i=param::DECLINE_IDX_K; i<param::NM; ++i) {
    p[idx0+i] = p[idx0+i-1] + drho_xy;
    v[idx0+i] = v[idx0+i-1] + dv_xy;
    a[idx0+i] = a[idx0+i-1] + da_xy;
  }
}

void MST::update_tail_section_p_v_a(const std::size_t section, const Eigen::Matrix3d& bRt, const Eigen::Vector3d& bpt0, const Eigen::Vector3d& bpj, const Eigen::Vector3d& bvj, const Eigen::Vector3d& baj, const Eigen::Vector3d& RtVrel, const Eigen::Vector3d& RtArel, const Eigen::Vector3d& omega, const Eigen::Vector3d& omega_dot, const double sin_theta_t, const double cos_theta_t) {
  std::array<Eigen::Vector3d, 2*param::NT>& p = strip_state_.p_t;
  std::array<Eigen::Vector3d, 2*param::NT>& v = strip_state_.v_t;
  std::array<Eigen::Vector3d, 2*param::NT>& a = strip_state_.a_t;
  const std::size_t idx0 = section*param::NT;
  const Eigen::Matrix3d bRt_transpose = bRt.transpose();
  const double span_sign = param::STRIP_SPAN_SIGN[section];
  const double omega2 = omega.squaredNorm();

  const auto update_at_position = [&](const std::size_t idx, const Eigen::Vector3d& position) {
    const Eigen::Vector3d drho = position - bpj;
    const Eigen::Vector3d point_velocity = bvj + omega.cross(drho);
    const Eigen::Vector3d point_acceleration = baj + omega_dot.cross(drho) + omega*omega.dot(drho) - omega2*drho;
    p[idx] = position;
    v[idx] = bRt_transpose * (RtVrel - point_velocity);
    a[idx] = bRt_transpose * (RtArel - point_acceleration);
  };

  for (std::size_t i=0; i<param::NT_R; ++i) {
    const Eigen::Vector3d position = bpt0 + span_sign*param::DY_T*static_cast<double>(i)*bRt.col(1);
    update_at_position(idx0+i, position);
  }

  const Eigen::Vector3d root_edge = bpt0 + span_sign*param::DY_T*(static_cast<double>(param::NT_R)-0.5)*bRt.col(1);
  for (std::size_t i=0; i<param::NT_S; ++i) {
    const double n = (static_cast<double>(i)+0.5)/static_cast<double>(param::NT_S);
    const Eigen::Vector3d position = root_edge + n*param::C_T*(cos_theta_t*bRt.col(0) + span_sign*sin_theta_t*bRt.col(1));
    update_at_position(idx0+param::NT_R+i, position);
  }
}

void MST::update_strip_w_wdot(Eigen::Vector3d& omega_i, Eigen::Vector3d& omega_dot_i, const Eigen::Matrix3d& bRsi, const Eigen::Vector3d& b_omega_b_theta, const Eigen::Vector3d& b_omega_dot_b_theta, const Eigen::Vector3d& omega_phi_psi, const Eigen::Vector3d& omega_dot_phi_psi) {
  const Eigen::Vector3d omega_b_theta = bRsi.transpose() * b_omega_b_theta;
  omega_i = omega_b_theta + omega_phi_psi;
  omega_dot_i = bRsi.transpose() * b_omega_dot_b_theta + omega_b_theta.cross(omega_phi_psi) + omega_dot_phi_psi;
}

void MST::update_wake_source(const State& s) {
  static_assert(param::NH == 7 && param::NR == 6 && param::NM == 25, "Wake source map must follow the wing strip layout.");
  static_assert(param::WAKE_SPAN_PANELS == 12, "Wake source map must contain 12 span panels.");

  constexpr std::array<std::size_t, 8> MANUS_NODE_INDEX{3, 6, 9, 12, 15, 18, 21, 24};
  const auto humerus_chord = [](const std::size_t i) {return param::C_H0 + param::DL_H*static_cast<double>(i);};
  const auto radius_chord = [](const std::size_t i) {return param::C_R0 + param::DL_R*static_cast<double>(i);};
  const auto manus_chord = [](const std::size_t i) {return i < param::DECLINE_IDX_K ? param::C_M0 + param::DL_M1*static_cast<double>(i) : param::C_MK + param::DL_M2*static_cast<double>(i-param::DECLINE_IDX_K);};

  for (std::size_t wing=0; wing<2; ++wing) {
    const std::size_t h_idx0 = wing*param::NH;
    const std::size_t r_idx0 = wing*param::NR;
    const std::size_t m_idx0 = wing*param::NM;
    const std::size_t h_state_idx0 = wing*param::NH;
    const std::size_t r_state_idx0 = 2*param::NH + wing*param::NR;
    const std::size_t m_state_idx0 = 2*(param::NH+param::NR) + wing*param::NM;
    const Eigen::Matrix3d& bRh = strip_state_.humerus_rotation[wing].bRri[0];
    const std::array<Eigen::Matrix3d, param::NR>& bRr = strip_state_.radius_rotation[wing].bRri;
    const Eigen::Matrix3d& bRm = strip_state_.manus_rotation[wing].bRri[0];
    std::array<Eigen::Vector3d, WAKE_SPAN_NODES>& quarter_chord_nodes = bound_wake_nodes_[wing];
    std::array<Eigen::Vector3d, WAKE_SPAN_NODES>& trailing_edge_nodes = trailing_edge_wake_nodes_[wing];
    std::array<double, WAKE_SPAN_NODES> node_gamma{};

    const auto set_node = [&s, &quarter_chord_nodes, &trailing_edge_nodes, &node_gamma](const std::size_t node, const Eigen::Vector3d& leading_edge, const Eigen::Matrix3d& rotation, const double chord, const double gamma) {
      const Eigen::Vector3d chord_vector = chord*rotation.col(0);
      quarter_chord_nodes[node] = s.pos + s.R*(leading_edge + 0.25*chord_vector);
      trailing_edge_nodes[node] = s.pos + s.R*(leading_edge + chord_vector);
      node_gamma[node] = gamma;
    };
    const auto set_joint_node = [&s, &quarter_chord_nodes, &trailing_edge_nodes, &node_gamma](const std::size_t node, const Eigen::Vector3d& leading_edge0, const Eigen::Matrix3d& rotation0, const double chord0, const double gamma0, const Eigen::Vector3d& leading_edge1, const Eigen::Matrix3d& rotation1, const double chord1, const double gamma1) {
      const Eigen::Vector3d chord_vector0 = chord0*rotation0.col(0);
      const Eigen::Vector3d chord_vector1 = chord1*rotation1.col(0);
      const Eigen::Vector3d quarter_chord = 0.5*(leading_edge0 + 0.25*chord_vector0 + leading_edge1 + 0.25*chord_vector1);
      const Eigen::Vector3d trailing_edge = 0.5*(leading_edge0 + chord_vector0 + leading_edge1 + chord_vector1);
      quarter_chord_nodes[node] = s.pos + s.R*quarter_chord;
      trailing_edge_nodes[node] = s.pos + s.R*trailing_edge;
      node_gamma[node] = 0.5*(gamma0+gamma1);
    };

    set_node(0, strip_state_.p_h[h_idx0], bRh, humerus_chord(0), wing_circulation_[h_state_idx0]);
    set_node(1, strip_state_.p_h[h_idx0+3], bRh, humerus_chord(3), wing_circulation_[h_state_idx0+3]);
    set_joint_node(2, strip_state_.p_h[h_idx0+6], bRh, humerus_chord(6), wing_circulation_[h_state_idx0+6], strip_state_.p_r[r_idx0], bRr[0], radius_chord(0), wing_circulation_[r_state_idx0]);
    set_node(3, strip_state_.p_r[r_idx0+3], bRr[3], radius_chord(3), wing_circulation_[r_state_idx0+3]);
    set_joint_node(4, strip_state_.p_r[r_idx0+5], bRr[5], radius_chord(5), wing_circulation_[r_state_idx0+5], strip_state_.p_m[m_idx0], bRm, manus_chord(0), wing_circulation_[m_state_idx0]);
    for (std::size_t i=0; i<MANUS_NODE_INDEX.size(); ++i) {
      const std::size_t strip = MANUS_NODE_INDEX[i];
      set_node(5+i, strip_state_.p_m[m_idx0+strip], bRm, manus_chord(strip), wing_circulation_[m_state_idx0+strip]);
    }

    const double orientation = param::STRIP_SPAN_SIGN[wing];
    for (std::size_t panel=0; panel<param::WAKE_SPAN_PANELS; ++panel) {
      bound_wake_gamma_[wing][panel] = 0.5*orientation*(node_gamma[panel]+node_gamma[panel+1]);
    }
  }
}

bool MST::update_wake(const State& s) {
  static_assert(param::WAKE_UPDATE_DECIMATION > 0, "Wake update decimation must be positive.");
  static_assert(param::WAKE_AGE_CELLS > 0, "Wake history must contain at least one age cell.");

  update_wake_source(s);
  if (!wake_initialized_) {
    for (std::size_t wing=0; wing<2; ++wing) {wake_nodes_[wing][0] = trailing_edge_wake_nodes_[wing];}
    wake_initialized_ = true;
    return true;
  }

  for (std::size_t wing=0; wing<2; ++wing) {
    for (std::size_t panel=0; panel<param::WAKE_SPAN_PANELS; ++panel) {
      wake_gamma_sum_[wing][panel] += bound_wake_gamma_[wing][panel];
    }
  }
  ++wake_sample_count_;
  if (wake_sample_count_ < param::WAKE_UPDATE_DECIMATION) {return false;}

  constexpr double WAKE_DT = static_cast<double>(param::WAKE_UPDATE_DECIMATION)*param::SIM_DT_SEC;
  const Eigen::Vector3d convection = WAKE_DT*s.vel_f;
  const std::size_t new_valid_cells = std::min(wake_valid_cells_+1, param::WAKE_AGE_CELLS);
  const double inv_sample_count = 1.0/static_cast<double>(wake_sample_count_);

  for (std::size_t wing=0; wing<2; ++wing) {
    for (std::size_t age=new_valid_cells; age>0; --age) {
      for (std::size_t node=0; node<WAKE_SPAN_NODES; ++node) {
        wake_nodes_[wing][age][node] = wake_nodes_[wing][age-1][node] + convection;
      }
    }
    wake_nodes_[wing][0] = trailing_edge_wake_nodes_[wing];

    for (std::size_t age=new_valid_cells-1; age>0; --age) {wake_gamma_[wing][age] = wake_gamma_[wing][age-1];}
    for (std::size_t panel=0; panel<param::WAKE_SPAN_PANELS; ++panel) {
      wake_gamma_[wing][0][panel] = inv_sample_count*wake_gamma_sum_[wing][panel];
      wake_gamma_sum_[wing][panel] = 0.0;
    }
  }

  wake_valid_cells_ = new_valid_cells;
  wake_sample_count_ = 0;
  return true;
}

void MST::update_tail_wake_velocity(const State& s) {
  constexpr double INV_FOUR_PI = 0.07957747154594767;
  constexpr double EPS2 = 1.0e-16;
  constexpr double MIN_ACTIVE_AREA = 1.0e-12;
  constexpr double CORE_RADIUS2 = param::WAKE_CORE_RADIUS*param::WAKE_CORE_RADIUS;
  constexpr double WAKE_DT = static_cast<double>(param::WAKE_UPDATE_DECIMATION)*param::SIM_DT_SEC;
  constexpr double CORE_GROWTH_PER_CELL = 4.0*param::AIR_KINEMATIC_VISCOSITY*WAKE_DT;
  std::array<Eigen::Vector3d, 2*param::NT> query_world{};
  std::array<std::size_t, 2*param::NT> active_tail_index{};
  std::size_t num_active_tail_strips = 0;

  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  tail_wake_velocity_world_.fill(zero);

  for (std::size_t section=0; section<2; ++section) {
    const Eigen::Matrix3d& bRt = strip_state_.bR_t[section];
    for (std::size_t i=0; i<param::NT; ++i) {
      const std::size_t tail_idx = section*param::NT+i;
      if (tail_chord_[i]*tail_width_[i] <= MIN_ACTIVE_AREA) {continue;}
      const Eigen::Vector3d query_body = strip_state_.p_t[tail_idx] + 0.25*tail_chord_[i]*bRt.col(0);
      query_world[tail_idx] = s.pos + s.R*query_body;
      active_tail_index[num_active_tail_strips++] = tail_idx;
    }
  }

  // Segment geometry is shared by every tail query and evaluated only once.
  const auto add_segment = [&query_world, &active_tail_index, &num_active_tail_strips, this](const Eigen::Vector3d& node_a, const Eigen::Vector3d& node_b, const double gamma, const double core_radius2) {
    if (std::abs(gamma) <= 1.0e-12) {return;}
    const Eigen::Vector3d segment = node_b-node_a;
    const double segment2 = segment.squaredNorm();
    if (segment2 <= EPS2) {return;}
    const double scale = gamma*INV_FOUR_PI;

    for (std::size_t i=0; i<num_active_tail_strips; ++i) {
      const std::size_t tail_idx = active_tail_index[i];
      const Eigen::Vector3d r1 = query_world[tail_idx]-node_a;
      const Eigen::Vector3d r2 = query_world[tail_idx]-node_b;
      const double r1_squared = r1.squaredNorm();
      const double r2_squared = r2.squaredNorm();
      if (r1_squared <= EPS2 || r2_squared <= EPS2) {continue;}

      const Eigen::Vector3d cross = r1.cross(r2);
      const double denominator = cross.squaredNorm() + core_radius2*segment2;
      if (denominator <= EPS2) {continue;}

      const double finite_segment = segment.dot(r1/std::sqrt(r1_squared) - r2/std::sqrt(r2_squared));
      tail_wake_velocity_world_[tail_idx] += scale*finite_segment/denominator*cross;
    }
  };

  for (std::size_t wing=0; wing<2; ++wing) {
    const std::array<Eigen::Vector3d, WAKE_SPAN_NODES>& bound_nodes = bound_wake_nodes_[wing];
    const std::array<Eigen::Vector3d, WAKE_SPAN_NODES>& trailing_edge_nodes = trailing_edge_wake_nodes_[wing];
    const std::array<double, param::WAKE_SPAN_PANELS>& bound_gamma = bound_wake_gamma_[wing];

    // Current bound lattice: quarter-chord, chordwise closure, and trailing edge.
    for (std::size_t panel=0; panel<param::WAKE_SPAN_PANELS; ++panel) {
      add_segment(bound_nodes[panel], bound_nodes[panel+1], bound_gamma[panel], CORE_RADIUS2);
      add_segment(trailing_edge_nodes[panel], trailing_edge_nodes[panel+1], -bound_gamma[panel], CORE_RADIUS2);
    }
    for (std::size_t node=0; node<WAKE_SPAN_NODES; ++node) {
      double gamma;
      if (node == 0) {gamma = -bound_gamma[0];}
      else if (node == param::WAKE_SPAN_PANELS) {gamma = bound_gamma.back();}
      else {gamma = bound_gamma[node-1]-bound_gamma[node];}
      add_segment(bound_nodes[node], trailing_edge_nodes[node], gamma, CORE_RADIUS2);
    }

    if (wake_valid_cells_ == 0) {continue;}
    const auto& wing_nodes = wake_nodes_[wing];
    const auto& wing_gamma = wake_gamma_[wing];

    // Spanwise edges shared by adjacent wake-age cells are evaluated once.
    for (std::size_t age=0; age<=wake_valid_cells_; ++age) {
      const double core_radius2 = CORE_RADIUS2 + CORE_GROWTH_PER_CELL*static_cast<double>(age);
      for (std::size_t panel=0; panel<param::WAKE_SPAN_PANELS; ++panel) {
        double gamma;
        if (age == 0) {gamma = wing_gamma[0][panel];}
        else if (age == wake_valid_cells_) {gamma = -wing_gamma[age-1][panel];}
        else {gamma = wing_gamma[age][panel]-wing_gamma[age-1][panel];}
        add_segment(wing_nodes[age][panel], wing_nodes[age][panel+1], gamma, core_radius2);
      }
    }

    // Streamwise edges retain only the net circulation of adjacent span panels.
    for (std::size_t age=0; age<wake_valid_cells_; ++age) {
      const double core_radius2 = CORE_RADIUS2 + CORE_GROWTH_PER_CELL*(static_cast<double>(age)+0.5);
      for (std::size_t node=0; node<WAKE_SPAN_NODES; ++node) {
        double gamma;
        if (node == 0) {gamma = -wing_gamma[age][0];}
        else if (node == param::WAKE_SPAN_PANELS) {gamma = wing_gamma[age].back();}
        else {gamma = wing_gamma[age][node-1]-wing_gamma[age][node];}
        add_segment(wing_nodes[age][node], wing_nodes[age+1][node], gamma, core_radius2);
      }
    }
  }
}


template <const double (&CD)[176][14], const double (&CL)[176][14], const double (&CM)[176][14], const double (&X0)[176][14], const double (&ALPHA_STALL_POS)[14], const double (&ALPHA_STALL_NEG)[14], std::size_t N, typename RotationAt, typename OmegaAt, typename OmegaDotYAt, typename ChordAt, typename WidthAt>
void MST::update_segment_aerodynamics(const std::array<Eigen::Vector3d, 2*N>& p, const std::array<Eigen::Vector3d, 2*N>& v, const std::array<Eigen::Vector3d, 2*N>& a, const std::size_t idx0, const std::size_t state_idx0, const std::size_t load_idx, RotationAt&& rotation_at, OmegaAt&& omega_at, OmegaDotYAt&& omega_dot_y_at, ChordAt&& chord_at, WidthAt&& width_at, const bool update_telemetry, const double helmbold_aspect_ratio) {
  constexpr std::size_t UPPER_SURFACE = 0;
  constexpr std::size_t LOWER_SURFACE = 1;
  constexpr double RAD_TO_DEG = 57.29577951308232;
  constexpr double DEG_TO_RAD = 0.017453292519943295;
  constexpr double TWO_PI = 6.283185307179586;
  constexpr double GK_D1 = 4.24;
  constexpr double GK_DSTALL_MAX = 20.0;
  constexpr double GK_MIN_SPEED = 0.1;
  constexpr double GK_SURFACE_BLEND_ANGLE = DEG_TO_RAD;
  constexpr double GK_REATTACH_TOLERANCE = 0.02;
  constexpr double WAGNER_A1 = 0.165;
  constexpr double WAGNER_A2 = 0.335;
  constexpr double WAGNER_EPS1 = 0.0455;
  constexpr double WAGNER_EPS2 = 0.3;
  constexpr double WAGNER_PHI0 = 1.0-WAGNER_A1-WAGNER_A2;
  constexpr double WAGNER_MIN_LIFT_SLOPE = 1.0e-6;
  constexpr double MIN_ACTIVE_AREA = 1.0e-12;
  constexpr std::size_t WAGNER_SLOPE_OFFSET = 5; // +/-1 deg around alpha=0
  constexpr double INV_DT = 1.0 / param::SIM_DT_SEC;
  constexpr double HALF_RHO = 0.5 * param::AIR_DENSITY;
  constexpr double QUARTER_PI_RHO = 0.7853981633974483 * param::AIR_DENSITY;
  constexpr double INV_KINEMATIC_VISCOSITY = 1.0 / param::AIR_KINEMATIC_VISCOSITY;
  constexpr std::size_t ZERO_ALPHA_IDX = 50;

  static const std::array<std::array<double, 14>, 2> wagner_airfoil = []() {
    std::array<std::array<double, 14>, 2> result{};
    for (std::size_t j=0; j<14; ++j) {
      const double lift_slope = (CL[ZERO_ALPHA_IDX+WAGNER_SLOPE_OFFSET][j]-CL[ZERO_ALPHA_IDX-WAGNER_SLOPE_OFFSET][j])/(2.0*DEG_TO_RAD);
      if (lift_slope > WAGNER_MIN_LIFT_SLOPE) {
        result[0][j] = lift_slope;
        result[1][j] = -CL[ZERO_ALPHA_IDX][j]/lift_slope;
      }
    }
    return result;
  }();

  Eigen::Vector3d force_accum = Eigen::Vector3d::Zero(); // body FRD
  Eigen::Vector3d moment_accum = Eigen::Vector3d::Zero(); // body FRD
  Eigen::Vector3d weighted_pos = Eigen::Vector3d::Zero(); // body FRD
  Eigen::Vector3d& added_mass_pos = added_mass_pos_[load_idx];
  Eigen::Matrix<double, 6, 6>& added_mass_matrix = added_mass_matrix_[load_idx];
  added_mass_pos = p[idx0];
  added_mass_matrix.setZero();
  Eigen::Matrix<double, 6, 1> added_direction;
  double weight = 0.0;

  // calculation for each strip
  for (std::size_t i=0; i<N; ++i) {
    const std::size_t idx = idx0+i;
    const std::size_t state_idx = state_idx0+i;
    DynamicStallState& dynamic_stall = dynamic_stall_state_[state_idx];
    std::array<double, 2>& wagner_state = wagner_state_[state_idx];
    const double c = chord_at(i);
    const double dy = width_at(i);
    const double area = c * dy;
    const Eigen::Vector3d& omega = omega_at(i);
    const double vx = v[idx].x();
    const double vy = v[idx].y();
    const double omega_y = omega.y();
    const double vz = v[idx].z() + 0.25*c*omega_y;
    const double U2 = vx*vx + vz*vz;
    
    double U = 0.0;
    double alpha = 0.0;
    double alpha_dot = 0.0;
    double Re = 0.0;
    double Cd = 0.0;
    double Cl_lut = 0.0;
    double Cl_dynamic = 0.0;
    double Cl_wagner = 0.0;
    double Cm = 0.0;
    double wagner_input = 0.0;
    const double wagner_z1 = wagner_state[0];
    const double wagner_z2 = wagner_state[1];
    double wagner_output = WAGNER_A1*wagner_z1 + WAGNER_A2*wagner_z2;
    double wagner_delta_downwash = 0.0;
    double wagner_lift_slope = 0.0;
    double X_eq = 1.0;
    double X = 1.0;
    double X_target = 1.0;
    double qS = 0.0;
    double Fx_lut = 0.0;
    double Fz_lut = 0.0;
    double My_lut = 0.0;
    double Fx_dynamic = 0.0;
    double Fz_dynamic = 0.0;
    double Fx_wagner = 0.0;
    double Fz_wagner = 0.0;

    const auto upper_surface_weight = [](const double angle) {
      const double coordinate = std::clamp(0.5*(angle/GK_SURFACE_BLEND_ANGLE + 1.0), 0.0, 1.0);
      return coordinate*coordinate*(3.0 - 2.0*coordinate);
    };
    const auto blend_surface = [](const std::array<double, 2>& value, const double upper_weight) {
      return upper_weight*value[UPPER_SURFACE] + (1.0-upper_weight)*value[LOWER_SURFACE];
    };

    if (U2 > 1e-12 && c > 0.0 && area > MIN_ACTIVE_AREA) {
      U = std::sqrt(U2);
      alpha = std::atan2(vz, vx);

      const double alpha_deg = alpha * RAD_TO_DEG;
      std::size_t alpha_idx;
      double k_alpha;
      param::coeff::get_idx_alpha(alpha_idx, k_alpha, alpha_deg);

      Re = U * c * INV_KINEMATIC_VISCOSITY;
      std::size_t Re_idx;
      double k_Re;
      param::coeff::get_idx_Re(Re_idx, k_Re, Re);

      Cd     = param::coeff::bilinear_interpolate(CD, alpha_idx, Re_idx, k_alpha, k_Re);
      Cl_lut = param::coeff::bilinear_interpolate(CL, alpha_idx, Re_idx, k_alpha, k_Re);
      Cm     = param::coeff::bilinear_interpolate(CM, alpha_idx, Re_idx, k_alpha, k_Re);

      const double alpha_zero_lift = wagner_airfoil[1][Re_idx] + k_Re*(wagner_airfoil[1][Re_idx+1]-wagner_airfoil[1][Re_idx]);
      wagner_lift_slope = wagner_airfoil[0][Re_idx] + k_Re*(wagner_airfoil[0][Re_idx+1]-wagner_airfoil[0][Re_idx]);
      if (helmbold_aspect_ratio > 0.0) {
        // Scale only circulatory lift; drag, pitching moment, and added mass stay unchanged.
        const double slope_ratio = wagner_lift_slope/(0.5*TWO_PI*helmbold_aspect_ratio);
        const double helmbold_scale = 1.0/(std::sqrt(1.0+slope_ratio*slope_ratio)+slope_ratio);
        Cl_lut *= helmbold_scale;
        wagner_lift_slope *= helmbold_scale;
      }

      const double wagner_alpha_downwash = vz-vx*alpha_zero_lift;
      wagner_input = wagner_alpha_downwash + 0.5*c*omega_y;
      wagner_output = WAGNER_PHI0*wagner_input + WAGNER_A1*wagner_z1 + WAGNER_A2*wagner_z2;
      wagner_delta_downwash = wagner_output-wagner_alpha_downwash;

      const double delta_tau = 2.0*U*param::SIM_DT_SEC/c;
      const double gain1 = -std::expm1(-WAGNER_EPS1*delta_tau);
      const double gain2 = -std::expm1(-WAGNER_EPS2*delta_tau);
      wagner_state[0] += gain1*(wagner_input-wagner_state[0]);
      wagner_state[1] += gain2*(wagner_input-wagner_state[1]);

      const auto lookup_X0 = [&Re_idx, &k_Re](const double lookup_alpha) {
        std::size_t lookup_idx;
        double lookup_fraction;
        param::coeff::get_idx_alpha(lookup_idx, lookup_fraction, lookup_alpha*RAD_TO_DEG);
        return param::coeff::bilinear_interpolate(X0, lookup_idx, Re_idx, lookup_fraction, k_Re);
      };

      const bool valid_gk_sample = U >= GK_MIN_SPEED;
      const double state_alpha = valid_gk_sample || !dynamic_stall.state_initialized ? alpha : dynamic_stall.alpha;
      const double neutral_X = X0[ZERO_ALPHA_IDX][Re_idx] + k_Re*(X0[ZERO_ALPHA_IDX][Re_idx+1]-X0[ZERO_ALPHA_IDX][Re_idx]);
      std::array<double, 2> X_eq_surface = dynamic_stall.X_eq;
      if (valid_gk_sample || !dynamic_stall.state_initialized) {
        // Only the suction-side branch needs an alpha lookup; the pressure side relaxes to X0(0).
        if (state_alpha >= 0.0) {
          X_eq_surface[UPPER_SURFACE] = lookup_X0(state_alpha);
          X_eq_surface[LOWER_SURFACE] = neutral_X;
        }
        else {
          X_eq_surface[UPPER_SURFACE] = neutral_X;
          X_eq_surface[LOWER_SURFACE] = lookup_X0(state_alpha);
        }
      }

      const bool advance_gk = valid_gk_sample && dynamic_stall.state_initialized && dynamic_stall.alpha_initialized;
      if (valid_gk_sample && !dynamic_stall.state_initialized) {
        dynamic_stall.X = X_eq_surface;
        dynamic_stall.X_eq = X_eq_surface;
        dynamic_stall.X_target = X_eq_surface;
        dynamic_stall.q_ss.fill(0.0);
        dynamic_stall.D2.fill(0.0);
        dynamic_stall.active.fill(false);
        dynamic_stall.state_initialized = true;
      }

      if (advance_gk) {
        // Use a causal finite difference because the current load pass excludes qddot-dependent acceleration terms.
        double delta_alpha = alpha-dynamic_stall.alpha;
        if (delta_alpha > M_PI) {delta_alpha -= TWO_PI;}
        else if (delta_alpha < -M_PI) {delta_alpha += TWO_PI;}
        alpha_dot = delta_alpha * INV_DT;

        const double relaxation_gain = -std::expm1(-U*param::SIM_DT_SEC/(GK_D1*c));
        for (std::size_t surface=0; surface<2; ++surface) {
          const double surface_sign = surface == UPPER_SURFACE ? 1.0 : -1.0;
          const double beta = surface_sign*alpha;
          const double beta_previous = surface_sign*dynamic_stall.alpha;
          const double beta_dot = surface_sign*alpha_dot;
          const double q = beta_dot*c/U;
          const double* alpha_stall_table = surface == UPPER_SURFACE ? ALPHA_STALL_POS : ALPHA_STALL_NEG;
          const double beta_stall = (alpha_stall_table[Re_idx] + k_Re*(alpha_stall_table[Re_idx+1]-alpha_stall_table[Re_idx])) * DEG_TO_RAD;

          if (!dynamic_stall.active[surface] && beta_dot > 0.0 && beta_previous < beta_stall && beta >= beta_stall) {
            // Latch the crossing rate and convective delay; later speed changes must not erase the event history.
            dynamic_stall.q_ss[surface] = std::max(q, 2.0e-6);
            const double Kss = 0.5*dynamic_stall.q_ss[surface];
            dynamic_stall.D2[surface] = std::min(0.0815*std::pow(Kss, -7.0/9.0) + GK_D1, GK_DSTALL_MAX);
            dynamic_stall.active[surface] = true;
          }

          double target = X_eq_surface[surface];
          if (dynamic_stall.active[surface]) {
            // Split reaction and vortex-formation delays; this recovers alpha-D2*q for a constant ramp.
            const double beta_effective = beta - ((dynamic_stall.D2[surface]-GK_D1)*q + GK_D1*dynamic_stall.q_ss[surface]);
            target = lookup_X0(surface_sign*std::max(beta_effective, 0.0));
          }
          dynamic_stall.X_target[surface] = target;
          dynamic_stall.X[surface] += relaxation_gain*(target-dynamic_stall.X[surface]);
          dynamic_stall.X[surface] = std::clamp(dynamic_stall.X[surface], 0.0, 1.0);

          if (dynamic_stall.active[surface] && beta_dot < 0.0 && target >= std::max(0.0, neutral_X-GK_REATTACH_TOLERANCE)) {
            dynamic_stall.active[surface] = false;
            dynamic_stall.q_ss[surface] = 0.0;
            dynamic_stall.D2[surface] = 0.0;
          }
        }
        dynamic_stall.X_eq = X_eq_surface;
      }
      else if (valid_gk_sample) {
        dynamic_stall.X_eq = X_eq_surface;
        for (std::size_t surface=0; surface<2; ++surface) {
          if (!dynamic_stall.active[surface]) {dynamic_stall.X_target[surface] = X_eq_surface[surface];}
        }
      }

      if (valid_gk_sample) {
        dynamic_stall.alpha = alpha;
        dynamic_stall.alpha_initialized = true;
      }
      else {dynamic_stall.alpha_initialized = false;}

      // Blend only the observable state near zero incidence; the two memories remain independent.
      const double upper_weight = upper_surface_weight(state_alpha);
      const std::array<double, 2>& effective_X_eq = dynamic_stall.state_initialized ? dynamic_stall.X_eq : X_eq_surface;
      X_eq = blend_surface(effective_X_eq, upper_weight);
      X = dynamic_stall.state_initialized ? blend_surface(dynamic_stall.X, upper_weight) : X_eq;
      if (update_telemetry) {X_target = dynamic_stall.state_initialized ? blend_surface(dynamic_stall.X_target, upper_weight) : X_eq;}

      const double one_plus_sqrt_X    = 1.0 + std::sqrt(X);
      const double one_plus_sqrt_X_eq = 1.0 + std::sqrt(X_eq);
      const double kirchhoff_X = 0.25*one_plus_sqrt_X*one_plus_sqrt_X;
      const double kirchhoff_X_eq = 0.25*one_plus_sqrt_X_eq*one_plus_sqrt_X_eq;
      const double kirchhoff_ratio = kirchhoff_X/kirchhoff_X_eq;
      Cl_dynamic = Cl_lut * kirchhoff_ratio;
      const double delta_Cl = Cl_dynamic - Cl_lut;
      Cl_wagner = kirchhoff_ratio*wagner_lift_slope*wagner_delta_downwash/U;

      const double k_f = HALF_RHO * U * area;
      qS = k_f * U;
      Fx_lut = k_f * (Cd*vx - Cl_lut*vz);
      Fz_lut = k_f * (Cd*vz + Cl_lut*vx);
      My_lut = qS * c * Cm;
      Fx_dynamic = -k_f * delta_Cl*vz;
      Fz_dynamic =  k_f * delta_Cl*vx;
      const double k_wagner = HALF_RHO*area*kirchhoff_ratio*wagner_lift_slope*wagner_delta_downwash;
      Fx_wagner = -k_wagner*vz;
      Fz_wagner =  k_wagner*vx;
    }
    else {
      // Zero-speed intervals carry no convective time, so preserve separation memory across stroke reversal.
      dynamic_stall.alpha_initialized = false;
      if (dynamic_stall.state_initialized) {
        const double upper_weight = upper_surface_weight(dynamic_stall.alpha);
        X_eq = blend_surface(dynamic_stall.X_eq, upper_weight);
        X = blend_surface(dynamic_stall.X, upper_weight);
        if (update_telemetry) {X_target = blend_surface(dynamic_stall.X_target, upper_weight);}
      }
    }

    if (state_idx < NUM_WING_STRIPS) {
      // Kutta-Joukowski-equivalent circulation from the circulatory lift only.
      wing_circulation_[state_idx] = 0.5*U*c*(Cl_dynamic+Cl_wagner);
    }

    // a and omega_dot contain only qdot-dependent bias in update_dynamics().
    // The qddot-dependent part is represented by MuJoCo's generalized mass matrix.
    const double omega_dot_y = omega_dot_y_at(i);
    const double added_mass = QUARTER_PI_RHO * c * area;
    const double normal_acceleration = a[idx].z() + omega_y*vx - omega.x()*vy + 0.5*c*omega_dot_y;
    const double added_force = added_mass * normal_acceleration;
    const double added_mass_c_c_inv32 = 0.03125*added_mass*c*c;
    const double added_moment = -0.25*c*added_force - added_mass_c_c_inv32*omega_dot_y;

    // Add each strip effect
    const Eigen::Matrix3d& bRsi = rotation_at(i);
    const Eigen::Vector3d quarter_chord = 0.25*c*bRsi.col(0);
    const Eigen::Vector3d aerodynamic_center = p[idx] + quarter_chord;
    const Eigen::Vector3d bF_lut = Fx_lut*bRsi.col(0) + Fz_lut*bRsi.col(2);
    const Eigen::Vector3d bF_dynamic = Fx_dynamic*bRsi.col(0) + Fz_dynamic*bRsi.col(2);
    const Eigen::Vector3d bF_wagner = Fx_wagner*bRsi.col(0) + Fz_wagner*bRsi.col(2);
    const Eigen::Vector3d bF_added = added_force*bRsi.col(2);
    const Eigen::Vector3d bF = bF_lut + bF_dynamic + bF_wagner + bF_added;
    const Eigen::Vector3d bM_lut = My_lut*bRsi.col(1);
    const Eigen::Vector3d bM_added = added_moment*bRsi.col(1);

    // Aggregate strip inertia at one reference point per rigid wing segment.
    const Eigen::Vector3d normal = bRsi.col(2);
    added_direction.head<3>() = normal;
    added_direction.tail<3>() = (p[idx] + 2.0*quarter_chord - added_mass_pos).cross(normal);
    added_mass_matrix.noalias() += added_mass*added_direction*added_direction.transpose();
    added_mass_matrix.bottomRightCorner<3, 3>().noalias() += added_mass_c_c_inv32*bRsi.col(1)*bRsi.col(1).transpose();

    force_accum += bF;
    moment_accum += aerodynamic_center.cross(bF) + bM_lut + bM_added;
    weighted_pos += qS * aerodynamic_center;
    weight += qS;

    if (update_telemetry) {
      aero_telemetry_.alpha[state_idx] = alpha;
      aero_telemetry_.alpha_dot[state_idx] = alpha_dot;
      aero_telemetry_.speed[state_idx] = U;
      aero_telemetry_.Re[state_idx] = Re;
      aero_telemetry_.Cd[state_idx] = Cd;
      aero_telemetry_.Cl_lut[state_idx] = Cl_lut;
      aero_telemetry_.Cl_dynamic[state_idx] = Cl_dynamic;
      aero_telemetry_.Cl_wagner[state_idx] = Cl_wagner;
      aero_telemetry_.Cm[state_idx] = Cm;
      aero_telemetry_.wagner_input[state_idx] = wagner_input;
      aero_telemetry_.wagner_z1[state_idx] = wagner_z1;
      aero_telemetry_.wagner_z2[state_idx] = wagner_z2;
      aero_telemetry_.wagner_output[state_idx] = wagner_output;
      aero_telemetry_.X_eq[state_idx] = X_eq;
      aero_telemetry_.X[state_idx] = X;
      aero_telemetry_.X_target[state_idx] = X_target;
      if (U > 0.0 && c > 0.0) {
        aero_telemetry_.tau1[state_idx] = GK_D1*c/U;
        // D2 is latched per surface; tau2 is only its instantaneous dimensional telemetry equivalent.
        aero_telemetry_.tau2[state_idx] = c/U*std::max(dynamic_stall.D2[UPPER_SURFACE], dynamic_stall.D2[LOWER_SURFACE]);
      }
      else {
        aero_telemetry_.tau1[state_idx] = 0.0;
        aero_telemetry_.tau2[state_idx] = 0.0;
      }
      aero_telemetry_.stall_active[state_idx] = dynamic_stall.active[UPPER_SURFACE] || dynamic_stall.active[LOWER_SURFACE] ? 1.0 : 0.0;
      aero_telemetry_.lut_force[state_idx] = bF_lut;
      aero_telemetry_.dynamic_force[state_idx] = bF_dynamic;
      aero_telemetry_.wagner_force[state_idx] = bF_wagner;
      aero_telemetry_.added_bias_force[state_idx] = bF_added;
      aero_telemetry_.lut_moment[state_idx] = bM_lut;
      aero_telemetry_.added_bias_moment[state_idx] = bM_added;
    }
  }

  Eigen::Vector3d reference_pos = Eigen::Vector3d::Zero();
  if (weight > 0.0) {reference_pos = weighted_pos / weight;}
  
  // Apply the equivalent wrench at the qS-weighted quarter-chord point.
  // This remains bounded when strip forces cancel near stroke reversal.
  aero_pos_[load_idx] = reference_pos;
  aero_force_[load_idx] = force_accum;
  aero_torque_[load_idx] = moment_accum - reference_pos.cross(force_accum);
}

template <std::size_t N, typename RotationAt, typename OmegaAt, typename OmegaDotYAt, typename ChordAt, typename WidthAt>
void MST::update_full_added_mass_segment(const std::array<Eigen::Vector3d, 2*N>& v, const std::array<Eigen::Vector3d, 2*N>& a, const std::size_t idx0, const std::size_t state_idx0, RotationAt&& rotation_at, OmegaAt&& omega_at, OmegaDotYAt&& omega_dot_y_at, ChordAt&& chord_at, WidthAt&& width_at) {
  constexpr double QUARTER_PI_RHO = 0.7853981633974483 * param::AIR_DENSITY;

  for (std::size_t i=0; i<N; ++i) {
    const std::size_t idx = idx0+i;
    const std::size_t state_idx = state_idx0+i;
    const double c = chord_at(i);
    const double area = c * width_at(i);
    const Eigen::Vector3d& omega = omega_at(i);
    const double omega_dot_y = omega_dot_y_at(i);
    const double added_mass = QUARTER_PI_RHO * c * area;
    const double normal_acceleration = a[idx].z() + omega.y()*v[idx].x() - omega.x()*v[idx].y() + 0.5*c*omega_dot_y;
    const double added_force = added_mass * normal_acceleration;
    const double added_moment = -0.25*c*added_force - 0.03125*added_mass*c*c*omega_dot_y;
    const Eigen::Matrix3d& bRsi = rotation_at(i);
    aero_telemetry_.added_full_force[state_idx] = added_force*bRsi.col(2);
    aero_telemetry_.added_full_moment[state_idx] = added_moment*bRsi.col(1);
  }
}

void MST::update_full_added_mass_telemetry() {
  for (std::size_t wing=0; wing<2; ++wing) {
    const StripRotation<1>& humerus_rotation = strip_state_.humerus_rotation[wing];
    const StripRotation<param::NR>& radius_rotation = strip_state_.radius_rotation[wing];
    const StripRotation<1>& manus_rotation = strip_state_.manus_rotation[wing];
    const Eigen::Matrix3d& bRhi = humerus_rotation.bRri[0];
    const Eigen::Matrix3d& bRmi = manus_rotation.bRri[0];

    update_full_added_mass_segment<param::NH>(
      strip_state_.v_h, strip_state_.a_h, wing*param::NH, wing*param::NH,
      [&bRhi](const std::size_t) -> const Eigen::Matrix3d& {return bRhi;},
      [this, wing](const std::size_t) -> const Eigen::Vector3d& {return strip_state_.w_h[wing];},
      [this, wing](const std::size_t) {return strip_state_.wdot_h[wing].y();},
      [](const std::size_t i) {return param::C_H0 + param::DL_H*static_cast<double>(i);},
      [&humerus_rotation](const std::size_t) {return param::DY_H*std::abs(humerus_rotation.cos_psi[0]);}
    );

    update_full_added_mass_segment<param::NR>(
      strip_state_.v_r, strip_state_.a_r, wing*param::NR, 2*param::NH+wing*param::NR,
      [&radius_rotation](const std::size_t i) -> const Eigen::Matrix3d& {return radius_rotation.bRri[i];},
      [this, wing](const std::size_t i) -> const Eigen::Vector3d& {return strip_state_.w_r[wing*param::NR+i];},
      [this, wing](const std::size_t i) {return strip_state_.wdot_r[wing*param::NR+i].y();},
      [](const std::size_t i) {return param::C_R0 + param::DL_R*static_cast<double>(i);},
      [&radius_rotation](const std::size_t i) {return param::DY_R*std::abs(radius_rotation.cos_psi[i]);}
    );

    update_full_added_mass_segment<param::NM>(
      strip_state_.v_m, strip_state_.a_m, wing*param::NM, 2*(param::NH+param::NR)+wing*param::NM,
      [&bRmi](const std::size_t) -> const Eigen::Matrix3d& {return bRmi;},
      [this, wing](const std::size_t) -> const Eigen::Vector3d& {return strip_state_.w_m[wing];},
      [this, wing](const std::size_t) {return strip_state_.wdot_m[wing].y();},
      [](const std::size_t i) {return i < param::DECLINE_IDX_K ? param::C_M0 + param::DL_M1*static_cast<double>(i) : param::C_MK + param::DL_M2*static_cast<double>(i-param::DECLINE_IDX_K);},
      [&manus_rotation](const std::size_t) {return param::DY_M*std::abs(manus_rotation.cos_psi[0]);}
    );
  }

  constexpr std::size_t tail_state_idx0 = 2*(param::NH+param::NR+param::NM);
  for (std::size_t section=0; section<2; ++section) {
    const Eigen::Matrix3d& bRt = strip_state_.bR_t[section];
    update_full_added_mass_segment<param::NT>(
      strip_state_.v_t, strip_state_.a_t, section*param::NT, tail_state_idx0+section*param::NT,
      [&bRt](const std::size_t) -> const Eigen::Matrix3d& {return bRt;},
      [this, section](const std::size_t) -> const Eigen::Vector3d& {return strip_state_.w_t[section];},
      [this, section](const std::size_t) {return strip_state_.wdot_t[section].y();},
      [this](const std::size_t i) {return tail_chord_[i];},
      [this](const std::size_t i) {return tail_width_[i];}
    );
  }
}

void MST::update(const State& s, const double theta_t, const bool acceleration_bias_only, const bool update_loads, const bool update_telemetry) {
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const Eigen::Vector3d body_acc = acceleration_bias_only ? zero : s.acc;
  const Eigen::Vector3d body_w_dot = acceleration_bias_only ? zero : s.w_dot;
  const Eigen::Matrix3d Rt = s.R.transpose();
  const Eigen::Vector3d RtVrel = Rt * (s.vel_f - s.vel);
  const Eigen::Vector3d RtArel = -(Rt * body_acc); // Steady freestream: acc_f = 0.
  bool wake_output_due = false;

  for (std::size_t wing=0; wing<2; ++wing) { // wing=0 : right wing, wing=1 : left wing
    const std::size_t j0 = param::NUM_WING_JOINTS_PER_WING*wing;

    Eigen::Vector3d omega_theta_h;
    Eigen::Vector3d omega_theta_r;
    Eigen::Vector3d omega_theta_m;
    Eigen::Vector3d omega_theta_sec;
    Eigen::Vector3d omega_dot_theta_h;
    Eigen::Vector3d omega_dot_theta_r;
    Eigen::Vector3d omega_dot_theta_m;
    Eigen::Vector3d omega_dot_theta_sec;

    Eigen::Vector3d b_omega_b_theta = s.w;
    Eigen::Vector3d b_omega_dot_b_theta = body_w_dot;
    Eigen::Vector3d b_omega_b_theta_h;
    Eigen::Vector3d b_omega_b_theta_r;
    Eigen::Vector3d b_omega_b_theta_m;
    Eigen::Vector3d b_omega_dot_b_theta_h;
    Eigen::Vector3d b_omega_dot_b_theta_r;
    Eigen::Vector3d b_omega_dot_b_theta_m;

    Eigen::Vector3d bpj_prev = Eigen::Vector3d::Zero();
    Eigen::Vector3d bvj_prev = Eigen::Vector3d::Zero();
    Eigen::Vector3d baj_prev = Eigen::Vector3d::Zero();
    Eigen::Vector3d bvj_h;
    Eigen::Vector3d bvj_r;
    Eigen::Vector3d bvj_m;
    Eigen::Vector3d baj_h;
    Eigen::Vector3d baj_r;
    Eigen::Vector3d baj_m;

    // Forward recursion of joint angular and linear kinematics.
    for (std::size_t local_j=0; local_j<param::NUM_WING_JOINTS_PER_WING; ++local_j) {
      const std::size_t j = j0 + local_j;
      const Eigen::Vector3d bRje1 = s.bTj[j].block<3, 1>(0, 0);
      const Eigen::Vector3d bpj = s.bTj[j].block<3, 1>(0, 3);
      const Eigen::Vector3d drho = bpj - bpj_prev;
      const double omega2 = b_omega_b_theta.squaredNorm();
      const Eigen::Vector3d bvj = bvj_prev + b_omega_b_theta.cross(drho);
      const Eigen::Vector3d baj = baj_prev + b_omega_dot_b_theta.cross(drho) + b_omega_b_theta*b_omega_b_theta.dot(drho) - omega2*drho;
      const Eigen::Vector3d omega_j = s.theta_dot[j] * bRje1;

      const double theta_ddot = acceleration_bias_only ? 0.0 : s.theta_ddot[j];
      b_omega_dot_b_theta += theta_ddot * bRje1 + b_omega_b_theta.cross(omega_j);
      b_omega_b_theta += omega_j;

      if (local_j == 1) {
        omega_theta_sec = b_omega_b_theta - s.w;
        omega_dot_theta_sec = b_omega_dot_b_theta - body_w_dot - s.w.cross(omega_theta_sec);
      }
      else if (local_j == 2) {
        omega_theta_h = b_omega_b_theta - s.w;
        omega_dot_theta_h = b_omega_dot_b_theta - body_w_dot - s.w.cross(omega_theta_h);
        b_omega_b_theta_h = b_omega_b_theta;
        b_omega_dot_b_theta_h = b_omega_dot_b_theta;
        bvj_h = bvj;
        baj_h = baj;
      }
      else if (local_j == 3) {
        omega_theta_r = b_omega_b_theta - s.w;
        omega_dot_theta_r = b_omega_dot_b_theta - body_w_dot - s.w.cross(omega_theta_r);
        b_omega_b_theta_r = b_omega_b_theta;
        b_omega_dot_b_theta_r = b_omega_dot_b_theta;
        bvj_r = bvj;
        baj_r = baj;
      }
      else if (local_j == 5) {
        omega_theta_m = b_omega_b_theta - s.w;
        omega_dot_theta_m = b_omega_dot_b_theta - body_w_dot - s.w.cross(omega_theta_m);
        b_omega_b_theta_m = b_omega_b_theta;
        b_omega_dot_b_theta_m = b_omega_dot_b_theta;
        bvj_m = bvj;
        baj_m = baj;
      }

      bpj_prev = bpj;
      bvj_prev = bvj;
      baj_prev = baj;
    }

    StripRotation<1>& humerus_rotation = strip_state_.humerus_rotation[wing];
    StripRotation<param::NR>& radius_rotation = strip_state_.radius_rotation[wing];
    StripRotation<1>& manus_rotation = strip_state_.manus_rotation[wing];

    Eigen::Matrix3d& bRhi = humerus_rotation.bRri[0];
    Eigen::Matrix3d& bRmi = manus_rotation.bRri[0];

    { // Update lin pos, vel, accel
      const Eigen::Vector3d bRsec_y = s.bTj[j0+1].block<3, 1>(0, 0); // bRsec.col(1)

      { // Humerus
        const Eigen::Matrix3d bRj = s.bTj[j0+2].block<3, 3>(0, 0);
        const Eigen::Vector3d bpj = s.bTj[j0+2].block<3, 1>(0, 3);
        const Eigen::Matrix4d& jTh0 = param::J_T_S0[3*wing];
        const Eigen::Matrix3d bRh0 = bRj * jTh0.block<3, 3>(0, 0);
        const Eigen::Vector3d bph0 = bpj + bRj * jTh0.block<3, 1>(0, 3);
        const Eigen::Vector3d drho0 = bph0 - bpj;
        const double omega2 = b_omega_b_theta_h.squaredNorm();
        const Eigen::Vector3d bvh0 = bvj_h + b_omega_b_theta_h.cross(drho0);
        const Eigen::Vector3d bah0 = baj_h + b_omega_dot_b_theta_h.cross(drho0) + b_omega_b_theta_h*b_omega_b_theta_h.dot(drho0) - omega2*drho0;
        update_Rz_psi(bRhi, humerus_rotation.sin_psi[0], humerus_rotation.cos_psi[0], humerus_rotation.psi_dot[0], humerus_rotation.psi_ddot[0], bRsec_y, bRh0, omega_theta_sec, omega_dot_theta_sec, omega_theta_h, omega_dot_theta_h);
        update_humerus_stream_p_v_a(wing*param::NH, bRh0, bph0, bRhi, RtVrel, RtArel, bvh0, bah0, b_omega_b_theta_h, b_omega_dot_b_theta_h, omega2, param::STRIP_SPAN_SIGN[wing]*param::DY_H);
      }
      
      { // Manus
        const Eigen::Matrix3d bRj = s.bTj[j0+5].block<3, 3>(0, 0);
        const Eigen::Vector3d bpj = s.bTj[j0+5].block<3, 1>(0, 3);
        const Eigen::Matrix4d& jTm0 = param::J_T_S0[3*wing+2];
        const Eigen::Matrix3d bRm0 = bRj * jTm0.block<3, 3>(0, 0);
        const Eigen::Vector3d bpm0 = bpj + bRj * jTm0.block<3, 1>(0, 3);
        const Eigen::Vector3d drho0 = bpm0 - bpj;
        const double omega2 = b_omega_b_theta_m.squaredNorm();
        const Eigen::Vector3d bvm0 = bvj_m + b_omega_b_theta_m.cross(drho0);
        const Eigen::Vector3d bam0 = baj_m + b_omega_dot_b_theta_m.cross(drho0) + b_omega_b_theta_m*b_omega_b_theta_m.dot(drho0) - omega2*drho0;
        update_Rz_psi(bRmi, manus_rotation.sin_psi[0], manus_rotation.cos_psi[0], manus_rotation.psi_dot[0], manus_rotation.psi_ddot[0], bRsec_y, bRm0, omega_theta_sec, omega_dot_theta_sec, omega_theta_m, omega_dot_theta_m);
        update_manus_stream_p_v_a(wing*param::NM, bRm0, bpm0, bRmi, RtVrel, RtArel, bvm0, bam0, b_omega_b_theta_m, b_omega_dot_b_theta_m, omega2, param::STRIP_SPAN_SIGN[wing]*param::DY_M);
      }

      { // Radius
        const Eigen::Matrix3d bRj = s.bTj[j0+3].block<3, 3>(0, 0);
        const Eigen::Vector3d bpj = s.bTj[j0+3].block<3, 1>(0, 3);
        const Eigen::Matrix4d& jTr0 = param::J_T_S0[3*wing+1];
        const Eigen::Matrix3d bRr0 = bRj * jTr0.block<3, 3>(0, 0);
        const Eigen::Vector3d bpr0 = bpj + bRj * jTr0.block<3, 1>(0, 3);
        const Eigen::Vector3d drho0 = bpr0 - bpj;
        const double omega2 = b_omega_b_theta_r.squaredNorm();
        const Eigen::Vector3d bvr0 = bvj_r + b_omega_b_theta_r.cross(drho0);
        const Eigen::Vector3d bar0 = baj_r + b_omega_dot_b_theta_r.cross(drho0) + b_omega_b_theta_r*b_omega_b_theta_r.dot(drho0) - omega2*drho0;
        const Eigen::Vector3d b_omega_hi = omega_theta_h + humerus_rotation.psi_dot[0]*bRhi.col(2);
        const Eigen::Vector3d b_omega_mi = omega_theta_m + manus_rotation.psi_dot[0]*bRmi.col(2);
        const Eigen::Vector3d b_omega_dot_hi = omega_dot_theta_h + humerus_rotation.psi_ddot[0]*bRhi.col(2) + humerus_rotation.psi_dot[0]*omega_theta_h.cross(bRhi.col(2));
        const Eigen::Vector3d b_omega_dot_mi = omega_dot_theta_m + manus_rotation.psi_ddot[0]*bRmi.col(2) + manus_rotation.psi_dot[0]*omega_theta_m.cross(bRmi.col(2));
        update_radius_strip_rotation(radius_rotation, bRsec_y, bRhi, bRmi, bRr0, omega_theta_sec, omega_dot_theta_sec, b_omega_hi, b_omega_dot_hi, b_omega_mi, b_omega_dot_mi, omega_theta_r, omega_dot_theta_r);
        update_radius_stream_p_v_a(wing*param::NR, bRr0, bpr0, radius_rotation.bRri, RtVrel, RtArel, bvr0, bar0, b_omega_b_theta_r, b_omega_dot_b_theta_r, omega2, param::STRIP_SPAN_SIGN[wing]*param::DY_R);
      }
    }

    { // Update ang vel&acc
      // Humerus
      const Eigen::Vector3d omega_phi_psi_h(0.0, 0.0, humerus_rotation.psi_dot[0]);
      const Eigen::Vector3d omega_dot_phi_psi_h(0.0, 0.0, humerus_rotation.psi_ddot[0]);
      update_strip_w_wdot(strip_state_.w_h[wing], strip_state_.wdot_h[wing], bRhi, b_omega_b_theta_h, b_omega_dot_b_theta_h, omega_phi_psi_h, omega_dot_phi_psi_h);

      // Manus
      const Eigen::Vector3d omega_phi_psi_m(0.0, 0.0, manus_rotation.psi_dot[0]);
      const Eigen::Vector3d omega_dot_phi_psi_m(0.0, 0.0, manus_rotation.psi_ddot[0]);
      update_strip_w_wdot(strip_state_.w_m[wing], strip_state_.wdot_m[wing], bRmi, b_omega_b_theta_m, b_omega_dot_b_theta_m, omega_phi_psi_m, omega_dot_phi_psi_m);

      // Radius
      const std::size_t idx0 = wing*param::NR;
      for (std::size_t i=0; i<param::NR; ++i) {
        const double phi_dot = radius_rotation.phi_dot[i];
        const double psi_dot = radius_rotation.psi_dot[i];
        const double phi_dot_psi_dot = phi_dot * psi_dot;

        const double sin_psi = radius_rotation.sin_psi[i];
        const double cos_psi = radius_rotation.cos_psi[i];
        const double phi_ddot = radius_rotation.phi_ddot[i];

        const Eigen::Vector3d omega_phi_psi(phi_dot*sin_psi, phi_dot*cos_psi, psi_dot);
        const Eigen::Vector3d omega_dot_phi_psi(phi_ddot*sin_psi + phi_dot_psi_dot*cos_psi, phi_ddot*cos_psi - phi_dot_psi_dot*sin_psi, radius_rotation.psi_ddot[i]);

        update_strip_w_wdot(strip_state_.w_r[idx0+i], strip_state_.wdot_r[idx0+i], radius_rotation.bRri[i], b_omega_b_theta_r, b_omega_dot_b_theta_r, omega_phi_psi, omega_dot_phi_psi);
      }
    }
        
    if (update_loads) { // Update aerodynamic loads
      const std::size_t load_idx0 = 3*wing;

      update_segment_aerodynamics<param::coeff::NACA_CD, param::coeff::NACA_CL, param::coeff::NACA_CM, param::coeff::NACA_GK_X0, param::coeff::NACA_GK_ALPHA_STALL, param::coeff::NACA_GK_ALPHA_STALL_NEG, param::NH>(
        strip_state_.p_h, strip_state_.v_h, strip_state_.a_h, wing*param::NH, wing*param::NH, load_idx0,
        [&bRhi](const std::size_t) -> const Eigen::Matrix3d& {return bRhi;},
        [this, wing](const std::size_t) -> const Eigen::Vector3d& {return strip_state_.w_h[wing];},
        [this, wing](const std::size_t) {return strip_state_.wdot_h[wing].y();},
        [](const std::size_t i) {return param::C_H0 + param::DL_H*static_cast<double>(i);},
        [&humerus_rotation](const std::size_t) {return param::DY_H*std::abs(humerus_rotation.cos_psi[0]);},
        update_telemetry
      );

      update_segment_aerodynamics<param::coeff::S20_CD, param::coeff::S20_CL, param::coeff::S20_CM, param::coeff::S20_GK_X0, param::coeff::S20_GK_ALPHA_STALL, param::coeff::S20_GK_ALPHA_STALL_NEG, param::NR>(
        strip_state_.p_r, strip_state_.v_r, strip_state_.a_r, wing*param::NR, 2*param::NH+wing*param::NR, load_idx0+1,
        [&radius_rotation](const std::size_t i) -> const Eigen::Matrix3d& {return radius_rotation.bRri[i];},
        [this, wing](const std::size_t i) -> const Eigen::Vector3d& {return strip_state_.w_r[wing*param::NR+i];},
        [this, wing](const std::size_t i) {return strip_state_.wdot_r[wing*param::NR+i].y();},
        [](const std::size_t i) {return param::C_R0 + param::DL_R*static_cast<double>(i);},
        [&radius_rotation](const std::size_t i) {return param::DY_R*std::abs(radius_rotation.cos_psi[i]);},
        update_telemetry
      );

      update_segment_aerodynamics<param::coeff::S40_CD, param::coeff::S40_CL, param::coeff::S40_CM, param::coeff::S40_GK_X0, param::coeff::S40_GK_ALPHA_STALL, param::coeff::S40_GK_ALPHA_STALL_NEG, param::NM>(
        strip_state_.p_m, strip_state_.v_m, strip_state_.a_m, wing*param::NM, 2*(param::NH+param::NR)+wing*param::NM, load_idx0+2,
        [&bRmi](const std::size_t) -> const Eigen::Matrix3d& {return bRmi;},
        [this, wing](const std::size_t) -> const Eigen::Vector3d& {return strip_state_.w_m[wing];},
        [this, wing](const std::size_t) {return strip_state_.wdot_m[wing].y();},
        [](const std::size_t i) {return i < param::DECLINE_IDX_K ? param::C_M0 + param::DL_M1*static_cast<double>(i) : param::C_MK + param::DL_M2*static_cast<double>(i-param::DECLINE_IDX_K);},
        [&manus_rotation](const std::size_t) {return param::DY_M*std::abs(manus_rotation.cos_psi[0]);},
        update_telemetry
      );
    }
  }

  if (update_loads) {wake_output_due = update_wake(s);}

  { // Tail kinematics
    Eigen::Vector3d b_omega_b_theta = s.w;
    Eigen::Vector3d b_omega_dot_b_theta = body_w_dot;
    Eigen::Vector3d bpj_prev = Eigen::Vector3d::Zero();
    Eigen::Vector3d bvj_prev = Eigen::Vector3d::Zero();
    Eigen::Vector3d baj_prev = Eigen::Vector3d::Zero();

    for (std::size_t local_j=0; local_j<param::NUM_TAIL_JOINTS; ++local_j) {
      const std::size_t j = param::NUM_WING_JOINTS + local_j;
      const Eigen::Vector3d bRje1 = s.bTj[j].block<3, 1>(0, 0);
      const Eigen::Vector3d bpj = s.bTj[j].block<3, 1>(0, 3);
      const Eigen::Vector3d drho = bpj - bpj_prev;
      const double omega2 = b_omega_b_theta.squaredNorm();
      const Eigen::Vector3d bvj = bvj_prev + b_omega_b_theta.cross(drho);
      const Eigen::Vector3d baj = baj_prev + b_omega_dot_b_theta.cross(drho) + b_omega_b_theta*b_omega_b_theta.dot(drho) - omega2*drho;
      const Eigen::Vector3d omega_j = s.theta_dot[j] * bRje1;
      const double theta_ddot = acceleration_bias_only ? 0.0 : s.theta_ddot[j];

      b_omega_dot_b_theta += theta_ddot*bRje1 + b_omega_b_theta.cross(omega_j);
      b_omega_b_theta += omega_j;
      bpj_prev = bpj;
      bvj_prev = bvj;
      baj_prev = baj;
    }

    const std::size_t tail_joint_idx = param::NUM_JOINTS-1;
    const Eigen::Matrix3d bRj = s.bTj[tail_joint_idx].block<3, 3>(0, 0);
    const double sin_theta_t = std::sin(theta_t);
    const double cos_theta_t = std::cos(theta_t);
    strip_state_.theta_t = theta_t;

    tail_chord_.fill(param::C_T);
    tail_width_.fill(param::DY_T);
    double tail_section_area = param::L_T*param::C_T;
    // The side-strip midpoint follows the radial leading edge.  Its chord is the
    // circular trailing-edge intercept minus that leading-edge x coordinate.
    const double side_width = param::C_T*std::abs(sin_theta_t)/static_cast<double>(param::NT_S);
    const double sin_theta_t2 = sin_theta_t*sin_theta_t;
    for (std::size_t i=0; i<param::NT_S; ++i) {
      const double n = (static_cast<double>(i)+0.5)/static_cast<double>(param::NT_S);
      const double radicand = std::max(0.0, 1.0-n*n*sin_theta_t2);
      tail_chord_[param::NT_R+i] = param::C_T*std::max(0.0, std::sqrt(radicand)-n*cos_theta_t);
      tail_width_[param::NT_R+i] = side_width;
      tail_section_area += tail_chord_[param::NT_R+i]*side_width;
    }

    // Project both the span and planform area through the fixed 15 deg tail cant.
    constexpr double TAIL_CANT_COS = 0.9659258262890683;
    const double tail_half_span = param::L_T + param::C_T*std::abs(sin_theta_t);
    const double tail_aspect_ratio = 2.0*TAIL_CANT_COS*tail_half_span*tail_half_span/tail_section_area;

    constexpr std::size_t tail_state_idx0 = 2*(param::NH+param::NR+param::NM);
    if (update_loads && side_width <= 1.0e-12) {
      for (std::size_t section=0; section<2; ++section) {
        for (std::size_t i=param::NT_R; i<param::NT; ++i) {
          const std::size_t state_idx = tail_state_idx0+section*param::NT+i;
          dynamic_stall_state_[state_idx] = {};
          wagner_state_[state_idx] = {};
        }
      }
    }

    for (std::size_t section=0; section<2; ++section) {
      const Eigen::Matrix4d& jTt0 = param::J_T_S0[6+section];
      Eigen::Matrix3d& bRt = strip_state_.bR_t[section];
      bRt = bRj * jTt0.block<3, 3>(0, 0);
      const Eigen::Vector3d bpt0 = bpj_prev + bRj*jTt0.block<3, 1>(0, 3);
      update_tail_section_p_v_a(section, bRt, bpt0, bpj_prev, bvj_prev, baj_prev, RtVrel, RtArel, b_omega_b_theta, b_omega_dot_b_theta, sin_theta_t, cos_theta_t);
      update_strip_w_wdot(strip_state_.w_t[section], strip_state_.wdot_t[section], bRt, b_omega_b_theta, b_omega_dot_b_theta, zero, zero);
    }

    if (update_loads && wake_output_due) {update_tail_wake_velocity(s);}
    for (std::size_t section=0; section<2; ++section) {
      const Eigen::Matrix3d& bRt = strip_state_.bR_t[section];
      for (std::size_t i=0; i<param::NT; ++i) {
        const std::size_t idx = section*param::NT+i;
        const Eigen::Vector3d wake_velocity = bRt.transpose()*Rt*tail_wake_velocity_world_[idx];
        if (update_telemetry) {
          const double vx_without_wake = strip_state_.v_t[idx].x();
          const double vz_without_wake = strip_state_.v_t[idx].z()+0.25*tail_chord_[i]*strip_state_.w_t[section].y();
          const double speed_without_wake = std::sqrt(vx_without_wake*vx_without_wake + vz_without_wake*vz_without_wake);
          const double vx_with_wake = vx_without_wake+wake_velocity.x();
          const double vz_with_wake = vz_without_wake+wake_velocity.z();
          const double speed_with_wake = std::sqrt(vx_with_wake*vx_with_wake + vz_with_wake*vz_with_wake);
          aero_telemetry_.tail_wake_delta_speed[idx] = tail_chord_[i]*tail_width_[i] > 1.0e-12 ? speed_with_wake-speed_without_wake : 0.0;
        }
        strip_state_.v_t[idx] += wake_velocity;
      }
    }

    if (update_loads) {
      for (std::size_t section=0; section<2; ++section) {
        const Eigen::Matrix3d& bRt = strip_state_.bR_t[section];
        // Keep the LUT two-dimensional like the wing solver; c*dy carries the
        // changing exposed planform area without a per-strip aspect-ratio term.
        update_segment_aerodynamics<param::coeff::FLAT_CD, param::coeff::FLAT_CL, param::coeff::FLAT_CM, param::coeff::FLAT_GK_X0, param::coeff::FLAT_GK_ALPHA_STALL, param::coeff::FLAT_GK_ALPHA_STALL_NEG, param::NT>(
          strip_state_.p_t, strip_state_.v_t, strip_state_.a_t, section*param::NT, tail_state_idx0+section*param::NT, NUM_WING_LOADS+section,
          [&bRt](const std::size_t) -> const Eigen::Matrix3d& {return bRt;},
          [this, section](const std::size_t) -> const Eigen::Vector3d& {return strip_state_.w_t[section];},
          [this, section](const std::size_t) {return strip_state_.wdot_t[section].y();},
          [this](const std::size_t i) {return tail_chord_[i];},
          [this](const std::size_t i) {return tail_width_[i];},
          update_telemetry,
          tail_aspect_ratio
        );
      }
    }
  }
}
