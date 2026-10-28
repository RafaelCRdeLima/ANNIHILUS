"""ANNIHILUS: neutrino pair annihilation near compact objects with backward ray tracing."""
from .engine import (Config, trace, trace_many, moments, page_thorne, kerr_residual, SPHERE,
                     DISK, ESCAPE, HORIZON, LOST, HIT_SPHERE, HIT_DISK, COMPONENTS)
from . import physics
