"""EXPLORATORY ablation of the revival-check interval (2026-11-19 evening; not pre-registered).

The library and PR #5915 run the revival check after every 5th batch (base._CHECK_UNUSED_EVERY). Here the interval is a
parameter, check_every = c: the check runs after batch t when (t + 1) % c == 0, exactly as the base does with 5 (the
rest of _finish_step is the base's, copied). c = 5 must reproduce the default runs bit for bit; c = 1 must reproduce
trigger="every". The event trigger (ablation.py, trigger="event") gives the decisions of c = 1 with a full check only
when an O(k) bound says a move may pass; n_full_checks counts the full checks.

Configs (name -> base config of ngp_variants2, options):
  pr@c, medg_pm@c, medg_fh@c for c in 1, 2, 5, 10, 20;  medg_pm@ev (event trigger)
"""
import ngp_variants2 as V
from ngp_variants2 import TIMES  # noqa: F401

INTERVALS = (1, 2, 5, 10, 20)
BASE = {"pr": "pr", "medg_pm": "ngp2_full_medg_pm", "medg_fh": "ngp2_full_medg_fh"}
CONFIGS = {}
for _name, _cfg in BASE.items():
    for _c in INTERVALS:
        _spec = V.CONFIGS[_cfg]
        _extra = {} if len(_spec) <= 4 else (dict(_spec[4]) if isinstance(_spec[4], dict) else {"w": _spec[4]})
        CONFIGS[f"{_name}@{_c}"] = _spec[:4] + (dict(_extra, check_every=_c),)
_spec = V.CONFIGS["ngp2_full_medg_pm"]
CONFIGS["medg_pm@ev"] = _spec[:4] + (dict(_spec[4], trigger="event"),)


class AblationEmeans(V.AblationEmeans):
    def __init__(self, *a, check_every=None, **kw):
        super().__init__(*a, **kw)
        self.check_every = None if check_every is None else int(check_every)
        self.n_checks = 0

    def _finish_step(self, expected_n):
        if self.check_every is None:
            return super()._finish_step(expected_n)
        assert getattr(self, "trigger", "periodic") == "periodic"
        # base._EmeansBase._finish_step (PR #5915 at 98deaca) with the interval as a parameter
        if (self._step + 1) % self.check_every == 0:
            self.n_checks += 1
            self._revive_unused(self._split_rng)
        self._invalidate_readout()
        self._step += 1
        self._epoch_steps += 1
        self._epoch_rows += expected_n
        self._pending_stats = None
        self._batch_accumulation = self._batch_objective = None
        self._batch_centroids = None
