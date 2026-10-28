/*
 * Backward ray tracing in Schwarzschild.
 *
 * The ray lies in the plane of e1 = e_r and e2 = (backward direction minus its radial part).
 * Along it, u = 1/r obeys u'' = -u + 3 M u^2 in the orbital angle phi (phi = 0 at the local
 * point, increasing backward in time). The point x(phi) = r (cos(phi) e1 + sin(phi) e2)
 * crosses the equatorial plane at phi_k = phi_0 + k pi.
 */
#include <math.h>
#include "annihilus.h"

void ann_default_config(ann_config *c)
{
    c->M = 1.0;
    c->emitter = ANN_SPHERE;
    c->R = 10.0;
    c->r_in = 6.0;
    c->r_out = 50.0;
    c->t_index = 0.0;
    c->keplerian = 1;
    c->n_scan = 64;
    c->n_gl = 16;
    c->n_phi = 64;
    c->h = 0.01;
    c->r_max = 1.0e6;
    c->max_orders = 8;
    c->tol_edge = 1.0e-12;
    c->spin = 0.0;
    c->metric = 0;
    c->m_rot = 0.0;
    c->n_w = 1;
    c->l_p = 1.0;
    c->r_p = 0.0;
    c->profile = 0;
    c->pt_spin = 0.0;
    c->t_norm = 1.0;
    c->rot_law = 0;
    c->rot_spin = 0.0;
}

static inline void rk4(double M, double *u, double *w, double h)
{
    double u0 = *u, w0 = *w;
    double k1u = w0, k1w = -u0 + 3.0 * M * u0 * u0;
    double u1 = u0 + 0.5 * h * k1u, w1 = w0 + 0.5 * h * k1w;
    double k2u = w1, k2w = -u1 + 3.0 * M * u1 * u1;
    double u2 = u0 + 0.5 * h * k2u, w2 = w0 + 0.5 * h * k2w;
    double k3u = w2, k3w = -u2 + 3.0 * M * u2 * u2;
    double u3 = u0 + h * k3u, w3 = w0 + h * k3w;
    double k4u = w3, k4w = -u3 + 3.0 * M * u3 * u3;
    *u = u0 + h / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
    *w = w0 + h / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
}

/* periapsis: root of u^2 (1 - 2 M u) = 1/b^2 in (0, 1/(3M)) */
static double periapsis(double M, double b)
{
    if (M <= 0.0) return 1.0 / b;
    double lo = 0.0, hi = 1.0 / (3.0 * M), ib2 = 1.0 / (b * b);
    for (int i = 0; i < 64; i++) {
        double m = 0.5 * (lo + hi);
        if (m * m * (1.0 - 2.0 * M * m) < ib2) lo = m; else hi = m;
    }
    return 0.5 * (lo + hi);
}

void ann_trace(const ann_config *c, double r, double th, double mu, double ph, ann_hit *out)
{
    if (c->metric == 1 || c->spin != 0.0) { ann_trace_kerr(c, r, th, mu, ph, out); return; }
    const double M = c->M;
    const int disk = (c->emitter == ANN_DISK);
    double s = sqrt(fmax(0.0, 1.0 - mu * mu));
    if (s < 1e-12) s = 1e-12;
    double A = 1.0 - 2.0 * M / r;
    double b = r * s / sqrt(A);
    double u = 1.0 / r;
    double w = (mu >= 0.0 ? 1.0 : -1.0) * sqrt(A) * fabs(mu) / (r * s);

    out->status = ANN_LOST;
    out->order = 0;
    out->path = 0;
    out->mu_em = 0.0;
    out->r_em = 0.0;
    out->g = 0.0;
    out->tau = 0.0;
    if (A <= 0.0 || (!disk && r < c->R)) return;

    double r_emax = disk ? c->r_out : c->R;
    double r_emin = disk ? c->r_in : c->R;
    double u_esc = 1.0 / fmax(3.0 * M, r_emax);
    /* inside the photon sphere an ingoing ray cannot come back out */
    double u_cap = (r_emin >= 3.0 * M) ? 1.0 / (3.0 * M) : 1.0 / (2.0 * M);
    double u_hor = 1.0 / (2.0 * M);
    double u_R = disk ? HUGE_VAL : 1.0 / c->R;
    double u_rmax = 1.0 / c->r_max;

    double phinode = HUGE_VAL;
    if (disk) {
        phinode = atan2(-cos(th), cos(ph) * sin(th));
        while (phinode <= 1e-12) phinode += M_PI;
    }
    double phimax = (c->max_orders + 2) * M_PI;
    double phi = 0.0;
    int order = 0, path = 0;

    for (;;) {
        if (u >= u_R) { out->status = ANN_HIT_SPHERE; break; }
        if (u >= u_hor || (w > 0.0 && u >= u_cap)) { out->status = ANN_HORIZON; break; }
        if (w < 0.0 && (u < u_esc || u < u_rmax)) { out->status = ANN_ESCAPE; break; }
        if (phi > phimax) { out->status = ANN_LOST; break; }

        double hh = c->h;
        if (fabs(w) * hh > u * c->h) hh = c->h * u / fabs(w);
        int at_node = 0;
        if (phinode <= phi + hh) { hh = phinode - phi; at_node = 1; }
        double wold = w;
        rk4(M, &u, &w, hh);
        phi += hh;

        if (at_node) {
            double rem = 1.0 / u;
            if (u > 0.0 && rem >= c->r_in && rem <= c->r_out) {
                out->status = ANN_HIT_DISK;
                out->r_em = rem;
                break;
            }
            if (rem > c->r_out && order < 30) path |= 1 << order;
            order++;
            phinode += M_PI;
            if (order > c->max_orders) { out->status = ANN_LOST; break; }
        }
        /* periapsis between steps: decide the sphere exactly from the first integral */
        if (!disk && wold > 0.0 && w <= 0.0) {
            if (periapsis(M, b) >= u_R) { out->status = ANN_HIT_SPHERE; break; }
        }
    }
    out->order = order;
    out->path = path;
    if (out->status == ANN_HIT_SPHERE) out->r_em = c->R;
    if (out->status >= ANN_HIT_SPHERE) {
        double lam = b * sin(th) * sin(ph);
        ann_emission(c, lam, b * b - lam * lam, sqrt(A), out);
    }
}

void ann_trace_many(const ann_config *c, double r, double th, int n, const double *mu,
                    const double *ph, int *status, int *order, double *r_em, double *g,
                    double *tau, int nthreads)
{
    if (nthreads <= 0) nthreads = 1;
#pragma omp parallel for schedule(dynamic, 64) num_threads(nthreads)
    for (int i = 0; i < n; i++) {
        ann_hit h;
        ann_trace(c, r, th, mu[i], ph[i], &h);
        status[i] = h.status;
        order[i] = h.order;
        r_em[i] = h.r_em;
        g[i] = h.g;
        tau[i] = h.tau;
    }
}
