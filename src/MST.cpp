#include "MST.hpp"

#include "coeff/coeff.hpp"
#include "utils.hpp" // State

#include <algorithm>
#include <cmath>

MST::MST() {reset();}

void MST::reset() {
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  strip_state_.reset();
  for (DynamicStallState& state : dynamic_stall_state_) {state = {};}
  aero_pos_.fill(zero);
  aero_force_.fill(zero);
  added_mass_pos_.fill(zero);
  for (Eigen::Matrix<double, 6, 6>& matrix : added_mass_matrix_) {matrix.setZero();}
  body_elipsoid_[0] = param::ELIPSOID_CENTER_POS;
  body_elipsoid_[1].setZero();
  body_elipsoid_[2].setZero();
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

    const double dx = param::ELIPSOID_SIZE.x();
    const double dy = param::ELIPSOID_SIZE.y();
    const double dz = param::ELIPSOID_SIZE.z();
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

  const Eigen::Vector3d lin_vel = s.R.transpose()*(s.vel-s.vel_f) + s.w.cross(param::ELIPSOID_CENTER_POS);
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

  body_elipsoid_[0] = param::ELIPSOID_CENTER_POS;
  body_elipsoid_[1] = force;
  body_elipsoid_[2] = torque;
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
  constexpr double dx = param::D_LPRI / (static_cast<double>(param::NM - param::DECLINE_IDX));

  const Eigen::Vector3d drho_y  = dy * bRm0.col(1);
  const Eigen::Vector3d drho_xy = drho_y + dx * bRm0.col(0);

  const Eigen::Vector3d dv_y = bRmi.transpose() * (-omega.cross(drho_y));
  const Eigen::Vector3d dv_xy = bRmi.transpose() * (-omega.cross(drho_xy));
  const Eigen::Vector3d da_y = bRmi.transpose() * (-omega_dot.cross(drho_y) - omega*omega.dot(drho_y) + omega2*drho_y);
  const Eigen::Vector3d da_xy = bRmi.transpose() * (-omega_dot.cross(drho_xy) - omega*omega.dot(drho_xy) + omega2*drho_xy);

  p[idx0] = bpm0;
  v[idx0] = bRmi.transpose() * (RtVrel - bvm0);
  a[idx0] = bRmi.transpose() * (RtArel - bam0);
  for (std::size_t i=1; i<param::DECLINE_IDX; ++i) {
    p[idx0+i] = p[idx0+i-1] + drho_y;
    v[idx0+i] = v[idx0+i-1] + dv_y;
    a[idx0+i] = a[idx0+i-1] + da_y;
  }
  for (std::size_t i=param::DECLINE_IDX; i<param::NM; ++i) {
    p[idx0+i] = p[idx0+i-1] + drho_xy;
    v[idx0+i] = v[idx0+i-1] + dv_xy;
    a[idx0+i] = a[idx0+i-1] + da_xy;
  }
}

void MST::update_strip_w_wdot(Eigen::Vector3d& omega_i, Eigen::Vector3d& omega_dot_i, const Eigen::Matrix3d& bRsi, const Eigen::Vector3d& b_omega_b_theta, const Eigen::Vector3d& b_omega_dot_b_theta, const Eigen::Vector3d& omega_phi_psi, const Eigen::Vector3d& omega_dot_phi_psi) {
  const Eigen::Vector3d omega_b_theta = bRsi.transpose() * b_omega_b_theta;
  omega_i = omega_b_theta + omega_phi_psi;
  omega_dot_i = bRsi.transpose() * b_omega_dot_b_theta + omega_b_theta.cross(omega_phi_psi) + omega_dot_phi_psi;
}


template <const double (&CD)[176][14], const double (&CL)[176][14], const double (&CM)[176][14], const double (&X0)[176][14], const double (&ALPHA_STALL)[14], std::size_t N, typename RotationAt, typename OmegaAt, typename OmegaDotYAt, typename ChordAt, typename WidthAt>
void MST::update_segment_aerodynamics(const std::array<Eigen::Vector3d, 2*N>& p, const std::array<Eigen::Vector3d, 2*N>& v, const std::array<Eigen::Vector3d, 2*N>& a, const std::size_t idx0, const std::size_t state_idx0, const std::size_t load_idx, RotationAt&& rotation_at, OmegaAt&& omega_at, OmegaDotYAt&& omega_dot_y_at, ChordAt&& chord_at, WidthAt&& width_at) {
  constexpr double RAD_TO_DEG = 57.29577951308232;
  constexpr double DEG_TO_RAD = 0.017453292519943295;
  constexpr double TWO_PI = 6.283185307179586;
  constexpr double GK_TAU1 = 4.24;
  constexpr double GK_DSTALL_MAX = 20.0;
  constexpr double INV_DT = 1.0 / param::SIM_DT_SEC;
  constexpr double HALF_RHO = 0.5 * param::AIR_DENSITY;
  constexpr double QUARTER_PI_RHO = 0.7853981633974483 * param::AIR_DENSITY;
  constexpr double INV_KINEMATIC_VISCOSITY = 1.0 / param::AIR_KINEMATIC_VISCOSITY;

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
    DynamicStallState& dynamic_stall = dynamic_stall_state_[state_idx0+i];
    const double c = chord_at(i);
    const double dy = width_at(i);
    const double area = c * dy;
    const Eigen::Vector3d& omega = omega_at(i);
    const double vx = v[idx].x();
    const double vy = v[idx].y();
    const double omega_y = omega.y();
    const double vz = v[idx].z() + 0.25*c*omega_y;
    const double U2 = vx*vx + vz*vz;
    
    double qS = 0.0;
    double Fx = 0.0;
    double Fz = 0.0;
    double My = 0.0;

    if (U2 > 1e-12 && c > 0.0 && dy > 0.0) {
      const double U = std::sqrt(U2);
      const double alpha = std::atan2(vz, vx);
      const double alpha_deg = alpha * RAD_TO_DEG;
      const double Re = U * c * INV_KINEMATIC_VISCOSITY;

      std::size_t alpha_idx;
      std::size_t Re_idx;
      double k_alpha;
      double k_Re;
      param::coeff::get_bilinear_lookup(alpha_idx, Re_idx, k_alpha, k_Re, alpha_deg, Re);

      const double Cd   = param::coeff::bilinear_interpolate(CD, alpha_idx, Re_idx, k_alpha, k_Re);
            double Cl   = param::coeff::bilinear_interpolate(CL, alpha_idx, Re_idx, k_alpha, k_Re);
      const double Cm   = param::coeff::bilinear_interpolate(CM, alpha_idx, Re_idx, k_alpha, k_Re);
      const double X_eq = param::coeff::bilinear_interpolate(X0, alpha_idx, Re_idx, k_alpha, k_Re);

      double alpha_dot = 0.0;
      if (dynamic_stall.initialized) {
        double delta_alpha = alpha-dynamic_stall.alpha;
        if (delta_alpha > M_PI) {delta_alpha -= TWO_PI;}
        else if (delta_alpha < -M_PI) {delta_alpha += TWO_PI;}
        alpha_dot = delta_alpha * INV_DT;

        if (!dynamic_stall.active && alpha_dot > 0.0) {
          const double alpha_stall = param::coeff::interpolate_Re(ALPHA_STALL, Re_idx, k_Re) * DEG_TO_RAD;
          if (dynamic_stall.alpha < alpha_stall && alpha >= alpha_stall) {
            // Latch the generalized GK delay at the positive static-stall crossing.
            const double Kss = std::max(0.5*alpha_dot*c/U, 1e-6);
            const double Dstall = std::min(0.0815*std::pow(Kss, -7.0/9.0) + GK_TAU1, GK_DSTALL_MAX);
            dynamic_stall.tau1 = GK_TAU1*c/U;
            dynamic_stall.tau2 = Dstall*c/U;
            dynamic_stall.active = true;
          }
        }
      }
      else {
      dynamic_stall.X = X_eq;
        dynamic_stall.alpha = alpha;
        dynamic_stall.tau1 = 0.0;
        dynamic_stall.tau2 = 0.0;
        dynamic_stall.active = false;
        dynamic_stall.initialized = true;
      }

      double X_target = X_eq;
      if (dynamic_stall.active) {
        std::size_t target_alpha_idx;
        double k_target_alpha;
        param::coeff::update_alpha_lookup(target_alpha_idx, k_target_alpha, (alpha-dynamic_stall.tau2*alpha_dot)*RAD_TO_DEG);
        X_target = param::coeff::bilinear_interpolate(X0, target_alpha_idx, Re_idx, k_target_alpha, k_Re);
      }
      const double tau1 = dynamic_stall.active ? dynamic_stall.tau1 : GK_TAU1*c/U;
      // Stable implicit-Euler update of tau1*dX/dt+X=X0(alpha_eff,Re).
      dynamic_stall.X += param::SIM_DT_SEC/(tau1+param::SIM_DT_SEC)*(X_target-dynamic_stall.X);
      dynamic_stall.X = std::clamp(dynamic_stall.X, 0.0, 1.0);
      dynamic_stall.alpha = alpha;

      if (dynamic_stall.active && alpha_dot < 0.0 && X_eq > 0.98) {
        dynamic_stall.active = false;
        dynamic_stall.tau1 = 0.0;
        dynamic_stall.tau2 = 0.0;
      }

      const double one_plus_sqrt_X = 1.0 + std::sqrt(dynamic_stall.X);
      const double one_plus_sqrt_X_eq = 1.0 + std::sqrt(X_eq);
      Cl *= (one_plus_sqrt_X*one_plus_sqrt_X)/(one_plus_sqrt_X_eq*one_plus_sqrt_X_eq);

      const double k_f = HALF_RHO * U * area;
      qS = k_f * U;
      Fx = k_f * (Cd*vx - Cl*vz);
      Fz = k_f * (Cd*vz + Cl*vx);
      My = qS * c * Cm;
    }
    else {
      dynamic_stall.initialized = false;
      dynamic_stall.active = false;
    }

    // a and omega_dot contain only qdot-dependent bias in update_dynamics().
    // The qddot-dependent part is represented by MuJoCo's generalized mass matrix.
    const double omega_dot_y = omega_dot_y_at(i);
    const double added_mass = QUARTER_PI_RHO * c * area;
    const double normal_acceleration = a[idx].z() + omega_y*vx - omega.x()*vy + 0.5*c*omega_dot_y;
    const double added_force = added_mass * normal_acceleration;

    Fz += added_force;
    const double added_mass_c_c_inv32 = 0.03125*added_mass*c*c;
    My -= 0.25*c*added_force + added_mass_c_c_inv32*omega_dot_y;

    // Add each strip effect
    const Eigen::Matrix3d& bRsi = rotation_at(i);
    const Eigen::Vector3d quarter_chord = 0.25*c*bRsi.col(0);
    const Eigen::Vector3d aerodynamic_center = p[idx] + quarter_chord;
    const Eigen::Vector3d bF = Fx*bRsi.col(0) + Fz*bRsi.col(2);

    // Aggregate strip inertia at one reference point per rigid wing segment.
    const Eigen::Vector3d normal = bRsi.col(2);
    added_direction.head<3>() = normal;
    added_direction.tail<3>() = (p[idx] + 2.0*quarter_chord - added_mass_pos).cross(normal);
    added_mass_matrix.noalias() += added_mass*added_direction*added_direction.transpose();
    added_mass_matrix.bottomRightCorner<3, 3>().noalias() += added_mass_c_c_inv32*bRsi.col(1)*bRsi.col(1).transpose();

    force_accum += bF;
    moment_accum += aerodynamic_center.cross(bF) + My * bRsi.col(1);
    weighted_pos += qS * aerodynamic_center;
    weight += qS;
  }

  Eigen::Vector3d reference_pos = Eigen::Vector3d::Zero();
  if (weight > 0.0) {reference_pos = weighted_pos / weight;}
  
  // Apply the equivalent wrench at the qS-weighted quarter-chord point.
  // This remains bounded when strip forces cancel near stroke reversal.
  aero_pos_[load_idx] = reference_pos;
  aero_force_[load_idx] = force_accum;
  aero_torque_[load_idx] = moment_accum - reference_pos.cross(force_accum);
}

void MST::update(const State& s, const bool acceleration_bias_only, const bool update_loads) {
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const Eigen::Vector3d body_acc = acceleration_bias_only ? zero : s.acc;
  const Eigen::Vector3d body_w_dot = acceleration_bias_only ? zero : s.w_dot;
  const Eigen::Matrix3d Rt = s.R.transpose();
  const Eigen::Vector3d RtVrel = Rt * (s.vel_f - s.vel);
  const Eigen::Vector3d RtArel = -(Rt * body_acc); // Steady freestream: acc_f = 0.

  for (std::size_t wing=0; wing<2; ++wing) { // wing=0 : right wing, wing=1 : left wing
    const std::size_t j0 = 6*wing;

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
    for (std::size_t local_j=0; local_j<6; ++local_j) {
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
        humerus_rotation.sin_phi[0] = 0.0;
        humerus_rotation.cos_phi[0] = 1.0;
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
        manus_rotation.sin_phi[0] = 0.0;
        manus_rotation.cos_phi[0] = 1.0;
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

      update_segment_aerodynamics<param::coeff::NACA_CD, param::coeff::NACA_CL, param::coeff::NACA_CM, param::coeff::NACA_GK_X0, param::coeff::NACA_GK_ALPHA_STALL, param::NH>(
        strip_state_.p_h, strip_state_.v_h, strip_state_.a_h, wing*param::NH, wing*param::NH, load_idx0,
        [&bRhi](const std::size_t) -> const Eigen::Matrix3d& {return bRhi;},
        [this, wing](const std::size_t) -> const Eigen::Vector3d& {return strip_state_.w_h[wing];},
        [this, wing](const std::size_t) {return strip_state_.wdot_h[wing].y();},
        [](const std::size_t i) {return param::L_ROOT + param::DL_H*static_cast<double>(i);},
        [&humerus_rotation](const std::size_t i) {return param::DY_H*std::abs(humerus_rotation.cos_psi[0]);}
      );

      update_segment_aerodynamics<param::coeff::S20_CD, param::coeff::S20_CL, param::coeff::S20_CM, param::coeff::S20_GK_X0, param::coeff::S20_GK_ALPHA_STALL, param::NR>(
        strip_state_.p_r, strip_state_.v_r, strip_state_.a_r, wing*param::NR, 2*param::NH+wing*param::NR, load_idx0+1,
        [&radius_rotation](const std::size_t i) -> const Eigen::Matrix3d& {return radius_rotation.bRri[i];},
        [this, wing](const std::size_t i) -> const Eigen::Vector3d& {return strip_state_.w_r[wing*param::NR+i];},
        [this, wing](const std::size_t i) {return strip_state_.wdot_r[wing*param::NR+i].y();},
        [](const std::size_t i) {return param::L_TRI + param::DL_R*static_cast<double>(i);},
        [&radius_rotation](const std::size_t i) {return param::DY_R*std::abs(radius_rotation.cos_psi[i]);}
      );

      update_segment_aerodynamics<param::coeff::S40_CD, param::coeff::S40_CL, param::coeff::S40_CM, param::coeff::S40_GK_X0, param::coeff::S40_GK_ALPHA_STALL, param::NM>(
        strip_state_.p_m, strip_state_.v_m, strip_state_.a_m, wing*param::NM, 2*(param::NH+param::NR)+wing*param::NM, load_idx0+2,
        [&bRmi](const std::size_t) -> const Eigen::Matrix3d& {return bRmi;},
        [this, wing](const std::size_t) -> const Eigen::Vector3d& {return strip_state_.w_m[wing];},
        [this, wing](const std::size_t) {return strip_state_.wdot_m[wing].y();},
        [](const std::size_t i) {return i < param::DECLINE_IDX ? param::L_SEC + param::DL_M1*static_cast<double>(i) : param::L_MPRI + param::DL_M2*static_cast<double>(i-param::DECLINE_IDX);},
        [&manus_rotation](const std::size_t i) {return param::DY_M*std::abs(manus_rotation.cos_psi[0]);}
      );
    }
  }
}
