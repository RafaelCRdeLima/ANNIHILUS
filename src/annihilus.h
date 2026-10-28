/*
 * ANNIHILUS: neutrino pair annihilation near compact objects with backward ray tracing.
 *
 * Geometric units G = c = 1. Lengths are in the same unit as M (usually M = 1).
 * At a point (r, theta) the local static frame is (e_r, e_theta, e_phi). A neutrino that
 * arrives there moving along
 *     n = mu e_r + s (cos(ph) e_theta + sin(ph) e_phi),   s = sqrt(1 - mu^2),
 * is traced backward to its source. The energy moments are written per unit
 * (g_s/h^3) Gamma(n+1) eta(n+1) (k T_ref)^(n+1), so that, for a ray that ends on the emitter,
 *     G3 = (g tau)^4,   G4 = (g tau)^5,
 * with g = E_loc/E_em and tau = T_em/T_ref.
 */
#ifndef ANNIHILUS_H
#define ANNIHILUS_H

enum { ANN_SPHERE = 0, ANN_DISK = 1 };

/* outcome of a backward ray */
enum { ANN_ESCAPE = 0, ANN_HORIZON = 1, ANN_LOST = 2, ANN_HIT_SPHERE = 3, ANN_HIT_DISK = 4 };

/* doubles per point returned by ann_moments: one block for n = 3 and one for n = 4, each
 * [M, M_i (3), M_ij (6, i <= j), M_ijk (10, i <= j <= k)], indices in the order r, th, ph */
#define ANN_NBLK 20
#define ANN_NMOM (2 * ANN_NBLK)

typedef struct {
    double M;          /* mass */
    int emitter;       /* ANN_SPHERE or ANN_DISK */
    double R;          /* sphere radius */
    double r_in;       /* disc inner radius (also T_ref radius for the disc) */
    double r_out;      /* disc outer radius */
    double t_index;    /* disc: T_em = T_ref (r_em/r_in)^(-t_index) */
    int keplerian;     /* disc: 1 Keplerian rotation, 0 static */
    int n_scan;        /* coarse samples in mu per azimuth */
    int n_gl;          /* Gauss-Legendre nodes per smooth segment */
    int n_phi;         /* azimuths (periodic trapezoid) */
    double h;          /* RK4 step in orbital angle */
    double r_max;      /* rays that leave this radius have escaped */
    int max_orders;    /* rays with more equatorial crossings are dropped */
    double tol_edge;   /* bisection tolerance in mu */
    double spin;       /* Kerr parameter a (same unit as M) */
    int metric;        /* 0: Schwarzschild (requires spin = 0), 1: Kerr */
    double m_rot;      /* flat space (M = 0): Keplerian velocities of a mass m_rot */
    int n_w;           /* weight blocks: 1, or 4 for w = 1, rho, mu_em, rho mu_em (below) */
    double l_p;        /* rho = exp(-(r_em - r_p)/l_p) */
    double r_p;
    int profile;       /* disc temperature: 0 power law (t_index), 1 Page-Thorne */
    double pt_spin;    /* Page-Thorne: spin (lengths in units of the hole's mass) */
    double t_norm;     /* Page-Thorne: T_em/T_ref = (F(r_em)/t_norm)^(1/4) */
    int rot_law;       /* flat space: 0 v = sqrt(m_rot/r); 1 v = speed of the prograde Keplerian
                          orbit of a Kerr hole (mass m_rot, spin rot_spin) relative to its ZAMO */
    double rot_spin;
} ann_config;

typedef struct {
    int status;        /* ANN_ESCAPE ... ANN_HIT_DISK */
    int order;         /* equatorial crossings before the end of the ray */
    int path;          /* bit k set if crossing k passed outside the disc (r > r_out) */
    double r_em;       /* emission radius */
    double g;          /* E_loc / E_em */
    double tau;        /* T_em / T_ref */
    double mu_em;      /* cosine between the ray and the disc normal, emitter frame */
} ann_hit;

void ann_default_config(ann_config *c);

/* one backward ray */
void ann_trace(const ann_config *c, double r, double th, double mu, double ph, ann_hit *out);

/* many backward rays from one point (sky maps) */
void ann_trace_many(const ann_config *c, double r, double th, int n, const double *mu,
                    const double *ph, int *status, int *order, double *r_em, double *g,
                    double *tau, int nthreads);

/* Kerr: backward ray in the local ZAMO frame (kerr.c) */
void ann_trace_kerr(const ann_config *c, double r, double th, double mu, double ph, ann_hit *out);
void ann_kerr_zamo(double M, double a, double r, double th, double *alpha, double *omega,
                   double *sqgpp, double *sigma, double *delta);

/* emission properties of a ray that ended on the emitter (emitter.c);
 * lam = L_z/E and eta = Q/E^2 of the ray, den = E/E_loc for the local observer */
void ann_emission(const ann_config *c, double lam, double eta, double den, ann_hit *h);

/* Page-Thorne flux shape F(r) (M = 1), zero at the ISCO */
double ann_page_thorne(double a, double r);

/* Kerr first integrals along a backward ray: res = {r'^2 - R, x'^2 - Theta_x} at sigma = 0
 * and at sigma_end (or where the ray ends), each divided by R + r'^2 (resp. Theta_x + x'^2) */
void ann_kerr_residual(const ann_config *c, double r, double th, double mu, double ph,
                       double sigma_end, double *res);

/* angular moments of G3 and G4 at npts points; mom has n_w*ANN_NMOM*npts entries
 * (weight blocks outermost: [point][weight][n = 3, 4][20]).
 * Returns the number of rays traced. */
long ann_moments(const ann_config *c, int npts, const double *r, const double *th, double *mom,
                 int nthreads);

#endif
