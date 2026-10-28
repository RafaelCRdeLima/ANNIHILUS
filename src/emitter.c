/*
 * Emission at the end of a backward ray: frequency ratio g = E_loc/E_em and T_em/T_ref.
 *
 * With E the conserved energy and lam = L_z/E, g = 1 / (den * E_em/E), where den = E/E_loc
 * for the local observer and E_em/E = -p.u_em/E for the emitting matter:
 *   static sphere:          E_em/E = 1/sqrt(1 - 2M/R);
 *   Keplerian disc (Kerr):  E_em/E = u^t (1 - Omega_K lam), Omega_K = sqrt(M)/(r^(3/2) + a sqrt(M)),
 *                           u^t = (r^(3/2) + a sqrt(M)) / (r^(3/4) sqrt(r^(3/2) - 3M r^(1/2) + 2a sqrt(M)));
 *   non-rotating disc:      zero angular momentum (ZAMO) matter, E_em/E = (1 - omega lam)/alpha,
 *                           which is the static emitter when a = 0;
 *   flat space (M = 0):     E_em/E = (1 - Omega lam)/sqrt(1 - v^2), Omega = v/r, with the Newtonian
 *                           Keplerian speed of a mass m_rot, v = sqrt(m_rot/r) (rot_law = 0), or the
 *                           speed of the Kerr Keplerian orbit relative to the ZAMO at the same r,
 *                           v = (Omega_K - omega) sqrt(g_phph)/alpha (rot_law = 1).
 * Disc temperature: T_em = T_ref (r_em/r_in)^(-t_index), or T_em = T_ref (F/t_norm)^(1/4) with the
 * Page-Thorne flux F of a disc around a hole of spin pt_spin (profile = 1).
 * The emission angle: at the equator p_theta^2 = eta E^2, and the boost of the disc is azimuthal,
 * so the cosine between the ray and the disc normal in the emitter frame is
 * mu_em = sqrt(eta) / (r_em E_em/E).
 */
#include <math.h>
#include "annihilus.h"

static double r_isco(double a)
{
    double z1 = 1.0 + cbrt(1.0 - a * a) * (cbrt(1.0 + a) + cbrt(1.0 - a));
    double z2 = sqrt(3.0 * a * a + z1 * z1);
    return 3.0 + z2 - sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2));
}

/* Page & Thorne (1974), prograde disc, M = 1: F(r) up to a constant factor */
double ann_page_thorne(double a, double r)
{
    double x = sqrt(r), x0 = sqrt(r_isco(a));
    if (x <= x0) return 0.0;
    double t = acos(a) / 3.0;
    double xs[3] = { 2.0 * cos(t - M_PI / 3.0), 2.0 * cos(t + M_PI / 3.0), -2.0 * cos(t) };
    double br = x - x0 - 1.5 * a * log(x / x0);
    for (int i = 0; i < 3; i++) {
        double xi = xs[i], xj = xs[(i + 1) % 3], xk = xs[(i + 2) % 3];
        if (fabs(xi) < 1e-9) continue;      /* a = 0: the coefficient vanishes with x_i */
        br -= 3.0 * (xi - a) * (xi - a) / (xi * (xi - xj) * (xi - xk)) * log((x - xi) / (x0 - xi));
    }
    return br / (x * x * x * x * (x * x * x - 3.0 * x + 2.0 * a));
}

void ann_emission(const ann_config *c, double lam, double eta, double den, ann_hit *h)
{
    const double M = c->M, a = c->spin;
    double re = h->r_em, eem;
    if (h->status == ANN_HIT_SPHERE) {
        eem = 1.0 / sqrt(1.0 - 2.0 * M / c->R);
        h->tau = 1.0;
    } else if (h->status == ANN_HIT_DISK) {
        if (c->profile == 1)
            h->tau = pow(fmax(ann_page_thorne(c->pt_spin, re), 0.0) / c->t_norm, 0.25);
        else
            h->tau = (c->t_index == 0.0) ? 1.0 : pow(re / c->r_in, -c->t_index);
        if (c->keplerian && M == 0.0) {
            /* flat space: special-relativistic Doppler shift of the disc velocities */
            double v = sqrt(c->m_rot / re);
            if (c->rot_law == 1 && c->m_rot > 0.0) {
                double al, om, sg, Sig, Del, sm = sqrt(c->m_rot);
                ann_kerr_zamo(c->m_rot, c->rot_spin, re, 0.5 * M_PI, &al, &om, &sg, &Sig, &Del);
                v = (sm / (re * sqrt(re) + c->rot_spin * sm) - om) * sg / al;
            }
            double Om = v / re;
            eem = (1.0 - Om * lam) / sqrt(1.0 - v * v);
        } else if (c->keplerian) {
            double sm = sqrt(M), r32 = re * sqrt(re);
            double Om = sm / (r32 + a * sm);
            double ut = (r32 + a * sm) / (pow(re, 0.75) * sqrt(r32 - 3.0 * M * sqrt(re) + 2.0 * a * sm));
            eem = ut * (1.0 - Om * lam);
        } else {
            double al, om, sg, Sig, Del;
            ann_kerr_zamo(M, a, re, 0.5 * M_PI, &al, &om, &sg, &Sig, &Del);
            eem = (1.0 - om * lam) / al;
        }
    } else {
        h->g = 0.0;
        h->tau = 0.0;
        return;
    }
    h->g = 1.0 / (den * eem);
    if (h->status == ANN_HIT_DISK) {
        double m = sqrt(fmax(eta, 0.0)) / (re * eem);
        h->mu_em = m > 1.0 ? 1.0 : m;
    } else {
        h->mu_em = 1.0;
    }
}
