import unittest

import numpy as np

from dmi.sst.mw_oe.retrieval import Retrieval


class RetrievalTest(unittest.TestCase):

    def test_prepare_first_guess(self):
        ws = np.float64(22.3)
        tcwv = np.float64(33.4)
        tclw = np.float64(44.5)
        sst = np.float64(55.6)
        eps = np.array([0.2, 0.1, 0.02, 0.25], dtype=np.float64)

        retrieval = Retrieval()
        [p, p_0] = retrieval.prepare_first_guess(ws, tcwv, tclw, sst, eps)

        self.assertEqual((4, 3), p.shape)
        self.assertAlmostEqual(22.1, p[0, 0], 8)
        self.assertAlmostEqual(33.5, p[1, 1], 8)
        self.assertAlmostEqual(44.5, p[2, 2], 8)
        self.assertAlmostEqual(55.35, p[3, 0], 8)

        self.assertEqual((4,), p_0.shape)
        self.assertAlmostEqual(22.3, p_0[0], 8)
        self.assertAlmostEqual(33.4, p_0[1], 8)
        self.assertAlmostEqual(44.5, p_0[2], 8)
        self.assertAlmostEqual(55.6, p_0[3], 8)

