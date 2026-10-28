/*
 * Angular moments of G3 = (g tau)^4 and G4 = (g tau)^5 over the local sky,
 *     M^0 = int G dOmega,  M^i = int G n^i dOmega,  M^ij = int G n^i n^j dOmega,
 *     M^ijk = int G n^i n^j n^k dOmega (needed for the momentum deposition),
 * in the static frame (e_r, e_theta, e_phi), with dOmega = dmu dph.
 *
 * For each azimuth, a coarse scan in mu classifies the rays (outcome and image order);
 * recursive bisection finds the edges between classes, and each segment that ends on the
 * emitter is integrated with Gauss-Legendre nodes. Azimuths use the periodic trapezoid rule.
 */
#include <math.h>
#include <stdlib.h>
#include "annihilus.h"

#define MAX_EDGES 512
#define NPOLE 14   /* extra scan points next to each pole */
#define TOL_DARK 1e-8

/* Class of a ray: outcome, number of equatorial crossings and, for each crossing, whether it
 * passed inside or outside the disc. Node radii vary continuously with direction, so a disc
 * image between two samples always shows up as a change of class, however thin it is. */
static int ray_class(const ann_hit *h)
{
    return h->status | (h->order << 3) | (h->path << 8);
}

static int emitting(int cls) { return (cls & 7) >= ANN_HIT_SPHERE; }

/* Gauss-Legendre nodes and weights on [-1, 1] */
static void gauleg(int n, double *x, double *w)
{
    for (int i = 0; i < (n + 1) / 2; i++) {
        double z = cos(M_PI * (i + 0.75) / (n + 0.5)), z1, pp;
        do {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 0; j < n; j++) {
                double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
            }
            pp = n * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z = z1 - p1 / pp;
        } while (fabs(z - z1) > 1e-15);
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = w[n - 1 - i] = 2.0 / ((1.0 - z * z) * pp * pp);
    }
}

typedef struct {
    const ann_config *c;
    double r, th, ph;
    double edge[MAX_EDGES];   /* edge positions in mu, increasing */
    int right[MAX_EDGES];     /* class to the right of each edge */
    int ne;
    long nrays;
} scan_t;

static int classify(scan_t *S, double mu)
{
    ann_hit h;
    ann_trace(S->c, S->r, S->th, mu, S->ph, &h);
    S->nrays++;
    return ray_class(&h);
}

static void bisect(scan_t *S, double a, int ca, double b, int cb)
{
    /* between two dark classes only an image wider than TOL_DARK matters */
    if (!emitting(ca) && !emitting(cb) && b - a < TOL_DARK) return;
    if (b - a < S->c->tol_edge) {
        if (S->ne < MAX_EDGES) {
            S->edge[S->ne] = 0.5 * (a + b);
            S->right[S->ne] = cb;
            S->ne++;
        }
        return;
    }
    double m = 0.5 * (a + b);
    int cm = classify(S, m);
    if (cm != ca) bisect(S, a, ca, m, cm);
    if (cm != cb) bisect(S, m, cm, b, cb);
}

static void accumulate(const ann_config *c, double *acc, double w, const ann_hit *h, double mu,
                       double s, double ph)
{
    double gt = h->g * h->tau;
    double G3 = gt * gt * gt * gt, G4 = G3 * gt;
    double n[3] = { mu, s * cos(ph), s * sin(ph) };
    double v[ANN_NBLK];
    int k = 0;
    v[k++] = 1.0;
    for (int i = 0; i < 3; i++) v[k++] = n[i];
    for (int i = 0; i < 3; i++)
        for (int j = i; j < 3; j++) v[k++] = n[i] * n[j];
    for (int i = 0; i < 3; i++)
        for (int j = i; j < 3; j++)
            for (int l = j; l < 3; l++) v[k++] = n[i] * n[j] * n[l];
    /* weight blocks for flavour conversion that depends on the emission point and angle */
    double wt[4] = { 1.0, 1.0, 1.0, 1.0 };
    int nw = c->n_w == 4 ? 4 : 1;
    if (nw == 4) {
        double rho = h->status == ANN_HIT_DISK ? exp(-(h->r_em - c->r_p) / c->l_p) : 1.0;
        wt[1] = rho;
        wt[2] = h->mu_em;
        wt[3] = rho * h->mu_em;
    }
    for (int b = 0; b < nw; b++) {
        double *a = acc + b * ANN_NMOM;
        for (k = 0; k < ANN_NBLK; k++) {
            a[k] += w * wt[b] * G3 * v[k];
            a[ANN_NBLK + k] += w * wt[b] * G4 * v[k];
        }
    }
}

static long moments_point(const ann_config *c, double r, double th, double *acc)
{
    int nu = c->n_scan < 2 ? 2 : c->n_scan;
    int ns = nu + 2 * NPOLE;
    int ng = c->n_gl < 1 ? 1 : c->n_gl;
    double *xg = malloc(ng * sizeof(double)), *wg = malloc(ng * sizeof(double));
    int *cls = malloc(ns * sizeof(int));
    double *mus = malloc(ns * sizeof(double));
    gauleg(ng, xg, wg);
    /* scan points uniform in the polar angle, refined geometrically towards both poles,
     * where small images (distant emitter, shadow) sit; mus[] increases */
    {
        double da = M_PI / (nu - 1);
        int k = 0;
        mus[k++] = -1.0;
        for (int j = NPOLE; j >= 1; j--) mus[k++] = -cos(da * ldexp(1.0, -j));
        for (int i = 1; i < nu - 1; i++) mus[k++] = -cos(da * i);
        for (int j = 1; j <= NPOLE; j++) mus[k++] = cos(da * ldexp(1.0, -j));
        mus[k++] = 1.0;
    }
    for (int k = 0; k < (c->n_w == 4 ? 4 : 1) * ANN_NMOM; k++) acc[k] = 0.0;

    scan_t S;
    S.c = c; S.r = r; S.th = th; S.nrays = 0;
    double dph = 2.0 * M_PI / c->n_phi;
    for (int j = 0; j < c->n_phi; j++) {
        S.ph = (j + 0.5) * dph;
        S.ne = 0;
        for (int k = 0; k < ns; k++) cls[k] = classify(&S, mus[k]);
        for (int k = 0; k + 1 < ns; k++)
            if (cls[k] != cls[k + 1]) bisect(&S, mus[k], cls[k], mus[k + 1], cls[k + 1]);

        /* segments: [-1, e_0], [e_0, e_1], ..., [e_{ne-1}, 1] */
        double a = -1.0;
        int ca = cls[0];
        for (int e = 0; e <= S.ne; e++) {
            double bnd = (e < S.ne) ? S.edge[e] : 1.0;
            if (emitting(ca) && bnd > a) {
                double hl = 0.5 * (bnd - a), mid = 0.5 * (bnd + a);
                for (int q = 0; q < ng; q++) {
                    double mu = mid + hl * xg[q];
                    double s = sqrt(fmax(0.0, 1.0 - mu * mu));
                    ann_hit h;
                    ann_trace(c, r, th, mu, S.ph, &h);
                    S.nrays++;
                    if (h.status >= ANN_HIT_SPHERE) accumulate(c, acc, wg[q] * hl * dph, &h, mu, s, S.ph);
                }
            }
            if (e < S.ne) { a = bnd; ca = S.right[e]; }
        }
    }
    free(xg); free(wg); free(cls); free(mus);
    return S.nrays;
}

long ann_moments(const ann_config *c, int npts, const double *r, const double *th, double *mom,
                 int nthreads)
{
    long total = 0;
    size_t stride = (size_t)(c->n_w == 4 ? 4 : 1) * ANN_NMOM;
    if (nthreads <= 0) nthreads = 1;
#pragma omp parallel for schedule(dynamic, 1) reduction(+:total) num_threads(nthreads)
    for (int i = 0; i < npts; i++)
        total += moments_point(c, r[i], th[i], mom + stride * i);
    return total;
}
