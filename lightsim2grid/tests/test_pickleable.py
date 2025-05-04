# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LightSim2grid, LightSim2grid implements a c++ backend targeting the Grid2Op platform.

import tempfile
import pickle
import os
import unittest
import warnings

import numpy as np

from grid2op import make
from lightsim2grid.lightSimBackend import LightSimBackend
import pdb


class TestPickle(unittest.TestCase):
    def test_save_load(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.env = make("rte_case5_example", test=True, backend=LightSimBackend())
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "test_pickle.pickle"), "wb") as f:
                    pickle.dump(self.env.backend, f)
                with open(os.path.join(tmpdir, "test_pickle.pickle"), "rb") as f:
                    backend_1 = pickle.load(f)
                nb_bus_total = self.env.n_sub * 2
                max_it = 10
                tol = 1e-8

                # TODO test in case the pickle file is corrupted...

                # test dc_pf
                V_0 = np.ones(nb_bus_total, dtype=np.complex_)
                V_0 = self.env.backend._grid.dc_pf(V_0, max_it, tol)

                V_1 = np.ones(nb_bus_total, dtype=np.complex_)
                V_1 = backend_1._grid.dc_pf(V_1, max_it, tol)

                assert np.all(np.abs(V_0 - V_1) <= 1e-7), "dc pf does not lead to same results"

                # test ac_pf
                V_0 = self.env.backend._grid.ac_pf(V_0, max_it, tol)
                V_1 = backend_1._grid.ac_pf(V_1, max_it, tol)
                assert np.all(np.abs(V_0 - V_1) <= 1e-7), "ac pf does not lead to same results"


if __name__ == "__main__":
    unittest.main()
