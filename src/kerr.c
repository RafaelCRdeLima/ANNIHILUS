/*
 * Backward ray tracing in Kerr (Boyer-Lindquist coordinates, local ZAMO frame).
 *
 * With E = 1, lambda = L_z/E and eta = Q/E^2, the radial and polar motions decouple in Mino
 * time sigma (d sigma = d lambda_affine / Sigma):
 *     (dr/dsigma)^2 = R(r) = (r^2 + a^2 - a lambda)^2 - Delta [eta + (lambda - a)^2],
 *     (dx/dsigma)^2 = eta (1 - x^2) + a^2 x^2 (1 - x^2) - lambda^2 x^2,    x = cos(theta).
 * We integrate the second-order forms r'' = R'/2, x'' = Theta_x'/2, which pass through
 * turning points without special treatment. The equatorial crossings x = 0 are located by
 * Newton iterations on the Runge-Kutta step.
 */
#include <math.h>
#include "annihilus.h"

typedef struct { double a, M, lam, K, C; } kerr_t;   /* C = eta + lambda^2 - a^2 */

static inline void deriv(const kerr_t *k, const double *y, double *dy)
{
    double r = y[0], x = y[2];
    dy[0] = y[1];
    dy[1] = 2.0 * r * (r * r + k->a * k->a - k->a * k->lam) - (r - k->M) * k->K;
    dy[2] = y[3];
    dy[3] = -k->C * x - 2.0 * k->a * k->a * x * x * x;
}

static void rk4(const kerr_t *k, const double *y0, double h, double *y)
{
    double k1[4], k2[4], k3[4], k4[4], t[4];
    deriv(k, y0, k1);
    for (int i = 0; i < 4; i++) t[i] = y0[i] + 0.5 * h * k1[i];
    deriv(k, t, k2);
    for (int i = 0; i < 4; i++) t[i] = y0[i] + 0.5 * h * k2[i];
    deriv(k, t, k3);
    for (int i = 0; i < 4; i++) t[i] = y0[i] + h * k3[i];
    deriv(k, t, k4);
    for (int i = 0; i < 4; i++) y[i] = y0[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

/* ZAMO quantities at (r, theta) */
void ann_kerr_zamo(double M, double a, double r, double th, double *alpha, double *omega,
                   double *sqgpp, double *sigma, double *delta)
{
    double sth = sin(th), cth = cos(th);
    double Sig = r * r + a * a * cth * cth;
    double Del = r * r - 2.0 * M * r + a * a;
    double AK = (r * r + a * a) * (r * r + a * a) - a * a * Del * sth * sth;
    *alpha = sqrt(Sig * Del / AK);
    *omega = 2.0 * M * a * r / AK;
    *sqgpp = sqrt(AK / Sig) * sth;
    *sigma = Sig;
    *delta = Del;
}

static double Rpot(const kerr_t *k, double r)
{
    double Del = r * r - 2.0 * k->M * r + k->a * k->a, P = r * r + k->a * k->a - k->a * k->lam;
    return P * P - Del * k->K;
}

static double Thpot(const kerr_t *k, double x)
{
    double eta = k->C - k->lam * k->lam + k->a * k->a;
    return eta * (1.0 - x * x) + k->a * k->a * x * x * (1.0 - x * x) - k->lam * k->lam * x * x;
}

void ann_kerr_residual(const ann_config *c, double r, double th, double mu, double ph,
                       double sigma_end, double *res)
{
    const double M = c->M, a = c->spin;
    double s = sqrt(fmax(0.0, 1.0 - mu * mu));
    double nr = mu, nth = s * cos(ph), nph = s * sin(ph);
    double al, om, sg, Sig, Del;
    ann_kerr_zamo(M, a, r, th, &al, &om, &sg, &Sig, &Del);
    double den = al + om * sg * nph, lam = sg * nph / den;
    double sth = sin(th), cth = cos(th), pth = sqrt(Sig) * nth / den;
    double eta = pth * pth + cth * cth * (lam * lam / (sth * sth) - a * a);
    kerr_t k = { a, M, lam, eta + (lam - a) * (lam - a), eta + lam * lam - a * a };
    double y[4] = { r, -sqrt(Sig * Del) * nr / den, cth, sth * sqrt(Sig) * nth / den };
    double rate0 = sqrt(fabs(eta) + lam * lam) + 1e-3, rfl = fmax(0.5 * (M + fabs(a)), 1e-3);
    double rh = (M + sqrt(fmax(0.0, M * M - a * a))) * (1.0 + 1e-3);
    for (int pass = 0; pass < 2; pass++) {
        double R = Rpot(&k, y[0]), T = Thpot(&k, y[2]);
        res[2 * pass] = (y[1] * y[1] - R) / (fabs(R) + y[1] * y[1] + 1e-300);
        res[2 * pass + 1] = (y[3] * y[3] - T) / (fabs(T) + y[3] * y[3] + 1e-300);
        if (pass == 1) break;
        double sig = 0.0;
        while (sig < sigma_end && y[0] > rh && y[0] < c->r_max) {
            double hh = c->h / (fabs(y[1]) / fmax(y[0], rfl) + rate0), y1[4];
            if (sig + hh > sigma_end) hh = sigma_end - sig;
            rk4(&k, y, hh, y1);
            for (int i = 0; i < 4; i++) y[i] = y1[i];
            sig += hh;
        }
    }
}

void ann_trace_kerr(const ann_config *c, double r, double th, double mu, double ph, ann_hit *out)
{
    const double M = c->M, a = c->spin;
    const int disk = (c->emitter == ANN_DISK);
    double s = sqrt(fmax(0.0, 1.0 - mu * mu));
    double nr = mu, nth = s * cos(ph), nph = s * sin(ph);
    double al, om, sg, Sig, Del;
    ann_kerr_zamo(M, a, r, th, &al, &om, &sg, &Sig, &Del);
    double rp_hor = M + sqrt(fmax(0.0, M * M - a * a));

    out->status = ANN_LOST;
    out->order = 0;
    out->path = 0;
    out->mu_em = 0.0;
    out->r_em = 0.0;
    out->g = 0.0;
    out->tau = 0.0;
    if (r <= rp_hor || (!disk && r < c->R)) return;

    double den = al + om * sg * nph;          /* E / E_zamo */
    double lam = sg * nph / den;
    double sth = sin(th), cth = cos(th);
    double pth = sqrt(Sig) * nth / den;       /* p_theta / E */
    double eta = pth * pth + cth * cth * (lam * lam / (sth * sth) - a * a);
    kerr_t k = { a, M, lam, eta + (lam - a) * (lam - a), eta + lam * lam - a * a };

    /* backward: reverse the forward Mino-time velocities */
    double y[4] = { r, -sqrt(Sig * Del) * nr / den, cth, sth * sqrt(Sig) * nth / den };
    double rate0 = sqrt(fabs(eta) + lam * lam) + 1e-3;
    double rfl = fmax(0.5 * (M + fabs(a)), 1e-3);     /* step floor near r = 0 (M = 0) */

    double r_emax = disk ? c->r_out : c->R;
    double r_esc = fmax(4.0 * M, r_emax);
    double r_hor = rp_hor * (1.0 + 1e-6);
    double smax = (c->max_orders + 4) * M_PI / rate0 * 4.0;
    double sig = 0.0;
    int order = 0, path = 0;
    long nstep = 0;

    for (;;) {
        if (!disk && y[0] <= c->R) { out->status = ANN_HIT_SPHERE; out->r_em = c->R; break; }
        if (y[0] <= r_hor) { out->status = ANN_HORIZON; break; }
        if (y[1] > 0.0 && (y[0] > r_esc || y[0] > c->r_max)) { out->status = ANN_ESCAPE; break; }
        if (sig > smax || ++nstep > 4000000) { out->status = ANN_LOST; break; }

        double hh = c->h / (fabs(y[1]) / fmax(y[0], rfl) + rate0);
        if (!(hh > 0.0) || !isfinite(y[0]) || !isfinite(y[1])) { out->status = ANN_LOST; break; }
        double y1[4];
        rk4(&k, y, hh, y1);

        if (disk && y[2] != 0.0 && y[2] * y1[2] <= 0.0) {
            /* crossing inside the step: Newton on the step length */
            double t = hh * y[2] / (y[2] - y1[2]), yc[4];
            for (int it = 0; it < 4; it++) {
                rk4(&k, y, t, yc);
                if (yc[3] == 0.0) break;
                double dt = -yc[2] / yc[3];
                t += dt;
                if (fabs(dt) < 1e-14 * hh) break;
            }
            rk4(&k, y, t, yc);
            double rem = yc[0];
            if (rem >= c->r_in && rem <= c->r_out) {
                out->status = ANN_HIT_DISK;
                out->r_em = rem;
                break;
            }
            if (rem > c->r_out && order < 30) path |= 1 << order;
            order++;
            if (order > c->max_orders) { out->status = ANN_LOST; break; }
        }
        for (int i = 0; i < 4; i++) y[i] = y1[i];
        sig += hh;
    }
    out->order = order;
    out->path = path;
    if (out->status >= ANN_HIT_SPHERE) ann_emission(c, lam, eta, den, out);
}
