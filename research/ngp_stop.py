"""Automatic early end of the move phase for the net-gain rule (2026-11-19 night).

A defensive mechanism against redundant computation: once revival has stopped paying, end the move phase early, exactly
as the PR's freeze ends it before the last pass. From then on: plain nearest-centre search (no second-nearest), no
removal/axis statistics, no margin statistics, no nearest-other-centre search, no revival checks. Nothing else changes
(EMA updates, schedule, closing mean of the last pass).

Stop rule ("quiet for one half-life"). A check is QUIET if
    early_stop = "hl"   : it accepts no move                                   (parameter-free), or
    early_stop = eps    : the predicted net gain of its accepted moves, sum over pairs of (w * gain - cost), is below
                          eps times the last batch's squared error (both in squared-error units per batch).
The move phase ends at a quiet check once the EMA has decayed by at least one half-life since the last check that was
not quiet (or since the start): prod (1 - alpha_t) over the batches in between <= 1/2, i.e. the statistics that the
quiet checks looked at have been renewed at least by half, and still nothing (worth eps) qualified. The moves accepted
at the stopping check itself are applied. "hl" is the limit eps -> 0 (an accepted move always has a positive net gain).

Implementation: one flag read by _frozen() (the switch the PR's driver and our class already use for the freeze), a
running sum of log(1 - alpha_t) since the last non-quiet check (no underflow), and the last batch's objective.
Class: ngp_fast.AblationEmeans (C++ decision) + early_stop=None | "hl" | eps (float). Records: es_stop_step (None if the
rule never fired), es_stopped_frac (share of the move phase completed when it fired).
"""
import math

import ngp_fast as _F
from ngp_fast import CONFIGS, TIMES  # noqa: F401


class AblationEmeans(_F.AblationEmeans):
    def __init__(self, *a, early_stop=None, **kw):
        super().__init__(*a, **kw)
        assert early_stop is None or early_stop == "hl" or (isinstance(early_stop, float) and early_stop > 0)
        self.early_stop = early_stop

    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        self._es_stopped = False
        self.es_stop_step = None
        self._es_logq = 0.0          # sum of log(1 - alpha_t) since the last non-quiet check
        self._es_obj = None          # the last batch's squared error

    def _frozen(self):
        return getattr(self, "_es_stopped", False) or super()._frozen()

    def _apply_statistics(self, ids, sums, counts, objective, alpha, decay, replace_usage=False):
        super()._apply_statistics(ids, sums, counts, objective, alpha, decay, replace_usage)
        if self.early_stop is not None and not self._es_stopped:
            g = 0.0 if replace_usage else float(decay)
            self._es_logq += math.log(g) if g > 0.0 else -math.inf
            self._es_obj = None if objective is None else float(objective)

    def _choose_splits(self, rng):
        src, tgt = super()._choose_splits(rng)
        if self.early_stop is None or self._es_stopped or self._calib_pred is None:
            return src, tgt                       # no estimate at this check (frozen): nothing to judge
        if self.early_stop == "hl":
            quiet = src is None
        else:
            net = 0.0
            if src is not None:
                gain, cost = self._calib_pred["gain"], self._calib_pred["cost"]
                net = float((self.w * gain[src] - cost[tgt]).sum())
            quiet = self._es_obj is not None and net < self.early_stop * self._es_obj
        if not quiet:
            self._es_logq = 0.0
        elif self._es_logq <= math.log(0.5):
            self._es_stopped = True
            self.es_stop_step = int(self._step)
        return src, tgt
