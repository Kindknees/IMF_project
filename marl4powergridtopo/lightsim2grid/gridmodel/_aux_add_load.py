# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of LightSim2grid, LightSim2grid implements a c++ backend targeting the Grid2Op platform.
import numpy as np


def _aux_add_load(model, pp_net):
    """
    Add the load of the pp_net into the lightsim2grid "model"

    Parameters
    ----------
    model
    pp_net

    Returns
    -------

    """
    if "parallel" in pp_net.load and np.any(pp_net.load["parallel"].values != 1):
        raise RuntimeError("Cannot handle 'parallel' load columns. Please duplicate the rows if that is the case. "
                           "Some pp_net.load[\"parallel\"] != 1 it is not handled by lightsim yet.")

    model.init_loads(pp_net.load["p_mw"].values,
                     pp_net.load["q_mvar"].values,
                     pp_net.load["bus"].values
                     )
    for load_id, is_connected in enumerate(pp_net.load["in_service"].values):
        if not is_connected:
            # load is deactivated
            model.deactivate_load(load_id)
