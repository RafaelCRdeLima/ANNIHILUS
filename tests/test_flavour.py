import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import flavour as F  # noqa: E402


class TestFlavour(unittest.TestCase):
    def test_lepton_kbar(self):
        # Delta(N_e - N_ebar) = -P (N_e - N_x) + Pbar (N_ebar - N_x) vanishes for Pbar = k P
        N = {"e": 10.0, "ebar": 6.0, "x": 2.0}
        k = F.lepton_kbar(N)
        self.assertAlmostEqual(k, 2.0)
        P = 0.3
        self.assertAlmostEqual(-P * (N["e"] - N["x"]) + k * P * (N["ebar"] - N["x"]), 0.0)
        # degenerate cases: N_e = N_x (k = 0), N_ebar = N_x != N_e, both, opposite signs
        self.assertEqual(F.lepton_kbar({"e": 5.0, "ebar": 7.0, "x": 5.0}), 0.0)
        self.assertIsNone(F.lepton_kbar({"e": 6.0, "ebar": 5.0, "x": 5.0}))
        self.assertEqual(F.lepton_kbar({"e": 5.0, "ebar": 5.0, "x": 5.0}), 1.0)
        self.assertIsNone(F.lepton_kbar({"e": 4.0, "ebar": 7.0, "x": 5.0}))

    def test_maxdev_exact(self):
        # ratio of two quadratics with its largest deviation inside the range of P
        g, f = np.array([-8.0, 2.2, 1.0]), np.array([0.0, 0.1, 1.0])
        P = np.linspace(0, 0.3, 4)
        rows = [dict(P=p, R_gr=np.polyval(g, p), R_flat=np.polyval(f, p)) for p in P]
        Pd = np.linspace(0, 0.3, 200001)
        dense = np.max(np.abs(np.polyval(g, Pd) / np.polyval(f, Pd) - 1))
        self.assertGreater(dense, F.maxdev(rows, "flat") + 1e-3)
        self.assertAlmostEqual(F.maxdev_exact(rows, "flat"), dense, places=9)


if __name__ == "__main__":
    unittest.main()
