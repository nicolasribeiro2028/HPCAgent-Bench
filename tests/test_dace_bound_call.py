# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""A compiled SDFG's timed call runs on the arguments marshalled for that rep.

``BoundCallPlan`` marshals in ``before_each`` and the timed ``run`` passes the raw pointers on. Every rep copies
fresh inputs (and ``vary_inputs`` draws new ones), so pointers marshalled once would point at a freed copy.
"""

import types

import dace
import numpy as np

from hpcagent_bench.frameworks.dace_framework import BoundCallPlan, TimedCompiledSDFG


@dace.program
def doubled(x: dace.float64[64], y: dace.float64[64]) -> None:
    y[:] = 2.0 * x


class FakeFramework:
    """The ``CallPlan`` hooks of a CPU framework calling by keyword."""

    def copy_func(self):  # noqa: ANN201 -- the CopyFunc the plan stores
        return np.copy

    def after_setup(self) -> None:
        return

    def call_args(self, bench: object, impl: object, resolved: dict, bdata: dict) -> tuple[list, dict]:
        return [], dict(resolved)

    def post_call(self, result: object) -> object:
        return result


def test_every_rep_runs_on_its_own_fresh_arguments() -> None:
    sdfg = doubled.to_sdfg()
    impl = TimedCompiledSDFG(sdfg.compile(), sdfg, "doubled")
    spec = types.SimpleNamespace(input_args=["x", "y"], array_args=["x", "y"], output_args=["y"])
    bdata: dict = {"y": np.zeros(64)}
    plan = BoundCallPlan(FakeFramework(), types.SimpleNamespace(spec=spec), impl, bdata)
    for rep in range(3):
        bdata["x"] = np.full(64, float(rep + 1))
        plan.before_each()
        plan.run()
        np.testing.assert_array_equal(plan.inout_values()[0], 2.0 * bdata["x"], err_msg=f"rep {rep}")


if __name__ == "__main__":
    test_every_rep_runs_on_its_own_fresh_arguments()
