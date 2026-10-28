"""ctypes interface to the C engine (libannihilus.so)."""
import ctypes
import os
from dataclasses import dataclass, fields

import numpy as np

_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libannihilus.so"))

SPHERE, DISK = 0, 1
ESCAPE, HORIZON, LOST, HIT_SPHERE, HIT_DISK = range(5)
NBLK = 20
NMOM = 2 * NBLK
# order of the 20 components of each moment block (indices r, th, ph)
_AX = ("r", "th", "ph")
COMPONENTS = (("0",) + _AX
              + tuple(_AX[i] + _AX[j] for i in range(3) for j in range(i, 3))
              + tuple(_AX[i] + _AX[j] + _AX[k]
                      for i in range(3) for j in range(i, 3) for k in range(j, 3)))


class _CConfig(ctypes.Structure):
    _fields_ = [
        ("M", ctypes.c_double),
        ("emitter", ctypes.c_int),
        ("R", ctypes.c_double),
        ("r_in", ctypes.c_double),
        ("r_out", ctypes.c_double),
        ("t_index", ctypes.c_double),
        ("keplerian", ctypes.c_int),
        ("n_scan", ctypes.c_int),
        ("n_gl", ctypes.c_int),
        ("n_phi", ctypes.c_int),
        ("h", ctypes.c_double),
        ("r_max", ctypes.c_double),
        ("max_orders", ctypes.c_int),
        ("tol_edge", ctypes.c_double),
        ("spin", ctypes.c_double),
        ("metric", ctypes.c_int),
        ("m_rot", ctypes.c_double),
        ("n_w", ctypes.c_int),
        ("l_p", ctypes.c_double),
        ("r_p", ctypes.c_double),
        ("profile", ctypes.c_int),
        ("pt_spin", ctypes.c_double),
        ("t_norm", ctypes.c_double),
        ("rot_law", ctypes.c_int),
        ("rot_spin", ctypes.c_double),
    ]


class _CHit(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int),
        ("order", ctypes.c_int),
        ("path", ctypes.c_int),
        ("r_em", ctypes.c_double),
        ("g", ctypes.c_double),
        ("tau", ctypes.c_double),
        ("mu_em", ctypes.c_double),
    ]


@dataclass
class Config:
    """Emitter and numerical parameters (geometric units, lengths in units of M)."""
    M: float = 1.0
    emitter: int = SPHERE
    R: float = 10.0
    r_in: float = 6.0
    r_out: float = 50.0
    t_index: float = 0.0
    keplerian: int = 1
    n_scan: int = 64
    n_gl: int = 16
    n_phi: int = 64
    h: float = 0.01
    r_max: float = 1.0e6
    max_orders: int = 8
    tol_edge: float = 1.0e-12
    spin: float = 0.0
    metric: int = 0           # 0 Schwarzschild, 1 Kerr (used whenever spin != 0)
    m_rot: float = 0.0        # flat space only: Keplerian velocities of this mass
    n_w: int = 1              # 4: weight blocks 1, rho, mu_em, rho*mu_em (flavour profiles)
    l_p: float = 1.0          # rho = exp(-(r_em - r_p)/l_p)
    r_p: float = 0.0
    profile: int = 0          # 0 power law (t_index), 1 Page-Thorne
    pt_spin: float = 0.0
    t_norm: float = 1.0
    rot_law: int = 0          # flat space: 0 v = sqrt(m_rot/r); 1 speed of the Kerr Keplerian
    rot_spin: float = 0.0     # orbit (mass m_rot, spin rot_spin) relative to the ZAMO at r

    def _c(self):
        return _CConfig(**{f.name: getattr(self, f.name) for f in fields(self)})


_dp = ctypes.POINTER(ctypes.c_double)
_ip = ctypes.POINTER(ctypes.c_int)
_lib.ann_trace.argtypes = [ctypes.POINTER(_CConfig), ctypes.c_double, ctypes.c_double,
                           ctypes.c_double, ctypes.c_double, ctypes.POINTER(_CHit)]
_lib.ann_trace.restype = None
_lib.ann_trace_many.argtypes = [ctypes.POINTER(_CConfig), ctypes.c_double, ctypes.c_double,
                                ctypes.c_int, _dp, _dp, _ip, _ip, _dp, _dp, _dp, ctypes.c_int]
_lib.ann_trace_many.restype = None
_lib.ann_moments.argtypes = [ctypes.POINTER(_CConfig), ctypes.c_int, _dp, _dp, _dp, ctypes.c_int]
_lib.ann_moments.restype = ctypes.c_long
_lib.ann_page_thorne.argtypes = [ctypes.c_double, ctypes.c_double]
_lib.ann_page_thorne.restype = ctypes.c_double
_lib.ann_kerr_residual.argtypes = [ctypes.POINTER(_CConfig), ctypes.c_double, ctypes.c_double,
                                   ctypes.c_double, ctypes.c_double, ctypes.c_double, _dp]
_lib.ann_kerr_residual.restype = None


def page_thorne(a, r):
    """Page-Thorne flux shape F(r) for spin a (M = 1); zero at and inside the ISCO."""
    return np.vectorize(lambda x: _lib.ann_page_thorne(a, x))(r)


def kerr_residual(cfg, r, th, mu, ph, sigma_end):
    """Relative residuals of the Kerr first integrals at the start and at sigma_end."""
    out = np.zeros(4)
    c = cfg._c()
    _lib.ann_kerr_residual(ctypes.byref(c), r, th, mu, ph, sigma_end, out.ctypes.data_as(_dp))
    return out


def _threads(n):
    return n if n and n > 0 else (os.cpu_count() or 1)


def trace(cfg, r, th, mu, ph):
    """One backward ray; returns dict(status, order, r_em, g, tau)."""
    c, h = cfg._c(), _CHit()
    _lib.ann_trace(ctypes.byref(c), r, th, mu, ph, ctypes.byref(h))
    return {k: getattr(h, k) for k, _ in _CHit._fields_}


def trace_many(cfg, r, th, mu, ph, nthreads=0):
    """Many backward rays from (r, th); mu, ph arrays of equal shape."""
    mu = np.ascontiguousarray(mu, dtype=float)
    ph = np.ascontiguousarray(np.broadcast_to(ph, mu.shape), dtype=float)
    n = mu.size
    st = np.zeros(n, dtype=np.int32)
    od = np.zeros(n, dtype=np.int32)
    re, g, tau = np.zeros(n), np.zeros(n), np.zeros(n)
    c = cfg._c()
    _lib.ann_trace_many(ctypes.byref(c), r, th, n, mu.ctypes.data_as(_dp), ph.ctypes.data_as(_dp),
                        st.ctypes.data_as(_ip), od.ctypes.data_as(_ip), re.ctypes.data_as(_dp),
                        g.ctypes.data_as(_dp), tau.ctypes.data_as(_dp), _threads(nthreads))
    shp = mu.shape
    return dict(status=st.reshape(shp), order=od.reshape(shp), r_em=re.reshape(shp),
                g=g.reshape(shp), tau=tau.reshape(shp))


def moments(cfg, r, th, nthreads=0):
    """Angular moments at points (r, th).

    Returns (mom, nrays): mom has shape (..., 2, 20); mom[..., 0, :] are the moments of
    G3 = (g tau)^4 and mom[..., 1, :] those of G4 = (g tau)^5, components as in COMPONENTS.
    With cfg.n_w = 4 the shape is (..., 4, 2, 20), one block per weight 1, rho, mu_em, rho mu_em.
    """
    r, th = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(th, dtype=float))
    shp = r.shape
    rr = np.ascontiguousarray(r.ravel())
    tt = np.ascontiguousarray(th.ravel())
    nw = 4 if cfg.n_w == 4 else 1
    out = np.zeros(nw * NMOM * rr.size)
    c = cfg._c()
    nrays = _lib.ann_moments(ctypes.byref(c), rr.size, rr.ctypes.data_as(_dp),
                             tt.ctypes.data_as(_dp), out.ctypes.data_as(_dp), _threads(nthreads))
    if nw == 4:
        return out.reshape(shp + (4, 2, NBLK)), nrays
    return out.reshape(shp + (2, NBLK)), nrays
