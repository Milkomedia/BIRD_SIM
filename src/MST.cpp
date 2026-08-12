#include "MST.hpp"

#include "utils.hpp" // State, StripRotation

namespace MST {

static inline void update_atan2_dot_ddot(double& angle_dot, double& angle_ddot, const double y, const double x, const double y_dot, const double x_dot, const double y_ddot, const double x_ddot) {
  constexpr double EPS2 = 1e-16;

  const double r2 = x*x + y*y;
  if (r2 <= EPS2) {angle_dot = 0.0; angle_ddot = 0.0; return;}

  const double q = x*y_dot - y*x_dot;
  const double inv_r2 = 1.0 / r2;
  angle_dot = q * inv_r2;
  angle_ddot = (x*y_ddot - y*x_ddot)*inv_r2 - 2.0*q*(x*x_dot + y*y_dot)*inv_r2*inv_r2;
}

static inline void update_relative_vector_dot_ddot(Eigen::Vector3d& x_dot_0, Eigen::Vector3d& x_ddot_0, const Eigen::Vector3d& x_0, const Eigen::Matrix3d& bR0, const Eigen::Vector3d& b_omega_0, const Eigen::Vector3d& b_omega_dot_0, const Eigen::Vector3d& b_omega_x, const Eigen::Vector3d& b_omega_dot_x) {
  const Eigen::Vector3d b_omega_rel = b_omega_x - b_omega_0;
  const Eigen::Vector3d omega_rel_0 = bR0.transpose() * b_omega_rel;
  const Eigen::Vector3d omega_dot_rel_0 = bR0.transpose() * (b_omega_dot_x - b_omega_dot_0 - b_omega_0.cross(b_omega_rel));

  x_dot_0 = omega_rel_0.cross(x_0);
  x_ddot_0 = omega_dot_rel_0.cross(x_0) + omega_rel_0.cross(x_dot_0);
}

static inline void update_Rz_psi(Eigen::Matrix3d& bRsi, double& sin_psi, double& cos_psi, double& psi_dot, double& psi_ddot, const Eigen::Vector3d& n, const Eigen::Matrix3d& bRs0, const Eigen::Vector3d& b_omega_n, const Eigen::Vector3d& b_omega_dot_n, const Eigen::Vector3d& b_omega_s0, const Eigen::Vector3d& b_omega_dot_s0) {
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

static inline void update_Ryphi_Rzpsi(Eigen::Matrix3d& bRri, double& sin_psi, double& cos_psi, double& sin_phi, double& cos_phi, double& psi_dot, double& psi_ddot, double& phi_dot, double& phi_ddot, const Eigen::Vector3d& xi, const Eigen::Matrix3d& bRr0, const Eigen::Vector3d& b_omega_xi, const Eigen::Vector3d& b_omega_dot_xi, const Eigen::Vector3d& b_omega_r0, const Eigen::Vector3d& b_omega_dot_r0, const bool initialized) {
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

static inline void update_radius_strip_rotation(StripRotation<param::NR>& rotation, const Eigen::Vector3d& n, const Eigen::Matrix3d& bRhi, const Eigen::Matrix3d& bRmi, const Eigen::Matrix3d& bRr0, const Eigen::Vector3d& b_omega_n, const Eigen::Vector3d& b_omega_dot_n, const Eigen::Vector3d& b_omega_hi, const Eigen::Vector3d& b_omega_dot_hi, const Eigen::Vector3d& b_omega_mi, const Eigen::Vector3d& b_omega_dot_mi, const Eigen::Vector3d& b_omega_r0, const Eigen::Vector3d& b_omega_dot_r0) {
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

static inline void update_humerus_stream_p_v_a(std::array<Eigen::Vector3d, 2*param::NH>& p, std::array<Eigen::Vector3d, 2*param::NH>& v, std::array<Eigen::Vector3d, 2*param::NH>& a, const std::size_t idx0, const Eigen::Matrix3d& bRh0, const Eigen::Vector3d& bph0, const Eigen::Matrix3d& bRhi, const Eigen::Vector3d& RtVrel, const Eigen::Vector3d& RtArel, const Eigen::Vector3d& bvh0, const Eigen::Vector3d& bah0, const Eigen::Vector3d& omega, const Eigen::Vector3d& omega_dot, const double omega2, const double dy) {
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

static inline void update_radius_stream_p_v_a(std::array<Eigen::Vector3d, 2*param::NR>& p, std::array<Eigen::Vector3d, 2*param::NR>& v, std::array<Eigen::Vector3d, 2*param::NR>& a, const std::size_t idx0, const Eigen::Matrix3d& bRr0, const Eigen::Vector3d& bpr0, const std::array<Eigen::Matrix3d, param::NR>& bRri, const Eigen::Vector3d& RtVrel, const Eigen::Vector3d& RtArel, const Eigen::Vector3d& bvr0, const Eigen::Vector3d& bar0, const Eigen::Vector3d& omega, const Eigen::Vector3d& omega_dot, const double omega2, const double dy) {
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

static inline void update_manus_stream_p_v_a(std::array<Eigen::Vector3d, 2*param::NM>& p, std::array<Eigen::Vector3d, 2*param::NM>& v, std::array<Eigen::Vector3d, 2*param::NM>& a, const std::size_t idx0, const Eigen::Matrix3d& bRm0, const Eigen::Vector3d& bpm0, const Eigen::Matrix3d& bRmi, const Eigen::Vector3d& RtVrel, const Eigen::Vector3d& RtArel, const Eigen::Vector3d& bvm0, const Eigen::Vector3d& bam0, const Eigen::Vector3d& omega, const Eigen::Vector3d& omega_dot, const double omega2, const double dy) {
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

static inline void update_strip_w_wdot(Eigen::Vector3d& omega_i, Eigen::Vector3d& omega_dot_i, const Eigen::Matrix3d& bRsi, const Eigen::Vector3d& b_omega_b_theta, const Eigen::Vector3d& b_omega_dot_b_theta, const Eigen::Vector3d& omega_phi_psi, const Eigen::Vector3d& omega_dot_phi_psi) {
  const Eigen::Vector3d omega_b_theta = bRsi.transpose() * b_omega_b_theta;
  omega_i = omega_b_theta + omega_phi_psi;
  omega_dot_i = bRsi.transpose() * b_omega_dot_b_theta + omega_b_theta.cross(omega_phi_psi) + omega_dot_phi_psi;
}

void update_strip(State& s) {
  const Eigen::Matrix3d Rt = s.R.transpose();
  const Eigen::Vector3d RtVrel = Rt * (s.vel_f - s.vel);
  const Eigen::Vector3d RtArel = -(Rt * s.acc); // Steady freestream: acc_f = 0.

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
    Eigen::Vector3d b_omega_dot_b_theta = s.w_dot;
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

      b_omega_dot_b_theta += s.theta_ddot[j] * bRje1 + b_omega_b_theta.cross(omega_j);
      b_omega_b_theta += omega_j;

      if (local_j == 1) {
        omega_theta_sec = b_omega_b_theta - s.w;
        omega_dot_theta_sec = b_omega_dot_b_theta - s.w_dot - s.w.cross(omega_theta_sec);
      }
      else if (local_j == 2) {
        omega_theta_h = b_omega_b_theta - s.w;
        omega_dot_theta_h = b_omega_dot_b_theta - s.w_dot - s.w.cross(omega_theta_h);
        b_omega_b_theta_h = b_omega_b_theta;
        b_omega_dot_b_theta_h = b_omega_dot_b_theta;
        bvj_h = bvj;
        baj_h = baj;
      }
      else if (local_j == 3) {
        omega_theta_r = b_omega_b_theta - s.w;
        omega_dot_theta_r = b_omega_dot_b_theta - s.w_dot - s.w.cross(omega_theta_r);
        b_omega_b_theta_r = b_omega_b_theta;
        b_omega_dot_b_theta_r = b_omega_dot_b_theta;
        bvj_r = bvj;
        baj_r = baj;
      }
      else if (local_j == 5) {
        omega_theta_m = b_omega_b_theta - s.w;
        omega_dot_theta_m = b_omega_dot_b_theta - s.w_dot - s.w.cross(omega_theta_m);
        b_omega_b_theta_m = b_omega_b_theta;
        b_omega_dot_b_theta_m = b_omega_dot_b_theta;
        bvj_m = bvj;
        baj_m = baj;
      }

      bpj_prev = bpj;
      bvj_prev = bvj;
      baj_prev = baj;
    }

    StripRotation<1>& humerus_rotation = s.humerus_rotation[wing];
    StripRotation<param::NR>& radius_rotation = s.radius_rotation[wing];
    StripRotation<1>& manus_rotation = s.manus_rotation[wing];

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
        update_humerus_stream_p_v_a(s.p_h, s.v_h, s.a_h, wing*param::NH, bRh0, bph0, bRhi, RtVrel, RtArel, bvh0, bah0, b_omega_b_theta_h, b_omega_dot_b_theta_h, omega2, param::STRIP_SPAN_SIGN[wing]*param::DY_H);
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
        update_manus_stream_p_v_a(s.p_m, s.v_m, s.a_m, wing*param::NM, bRm0, bpm0, bRmi, RtVrel, RtArel, bvm0, bam0, b_omega_b_theta_m, b_omega_dot_b_theta_m, omega2, param::STRIP_SPAN_SIGN[wing]*param::DY_M);
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
        update_radius_stream_p_v_a(s.p_r, s.v_r, s.a_r, wing*param::NR, bRr0, bpr0, radius_rotation.bRri, RtVrel, RtArel, bvr0, bar0, b_omega_b_theta_r, b_omega_dot_b_theta_r, omega2, param::STRIP_SPAN_SIGN[wing]*param::DY_R);
      }
    }

    { // Update ang vel&acc
      // Humerus
      const Eigen::Vector3d omega_phi_psi_h(0.0, 0.0, humerus_rotation.psi_dot[0]);
      const Eigen::Vector3d omega_dot_phi_psi_h(0.0, 0.0, humerus_rotation.psi_ddot[0]);
      update_strip_w_wdot(s.w_h[wing], s.wdot_h[wing], bRhi, b_omega_b_theta_h, b_omega_dot_b_theta_h, omega_phi_psi_h, omega_dot_phi_psi_h);

      // Manus
      const Eigen::Vector3d omega_phi_psi_m(0.0, 0.0, manus_rotation.psi_dot[0]);
      const Eigen::Vector3d omega_dot_phi_psi_m(0.0, 0.0, manus_rotation.psi_ddot[0]);
      update_strip_w_wdot(s.w_m[wing], s.wdot_m[wing], bRmi, b_omega_b_theta_m, b_omega_dot_b_theta_m, omega_phi_psi_m, omega_dot_phi_psi_m);

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

        update_strip_w_wdot(s.w_r[idx0+i], s.wdot_r[idx0+i], radius_rotation.bRri[i], b_omega_b_theta_r, b_omega_dot_b_theta_r, omega_phi_psi, omega_dot_phi_psi);
      }
    }
  }
}

}  // namespace MST