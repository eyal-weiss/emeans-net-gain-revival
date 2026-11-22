"""Net-gain revival on the GPU class of FAISS PR #5915 (contrib/e_means/e_means_torch.Emeans at 98deaca), optimized.

Differences from the proof of concept (research/netgain_torch.py):
  * per batch, the five per-centre sums are computed row-parallel in FP64 and added per centroid by a Triton
    segmented sum in a fixed order (netgain_triton.ng_accumulate: no atomics, deterministic, balanced when a few
    clusters hold most of a batch), and the other per-centre state (points and batches since birth, inherited weight,
    prod(1 - alpha), the early-stop clock) is updated with device operations only, so the driver keeps capturing and
    replaying its training step as a CUDA graph;
  * per check (every five batches), every centroid's nearest other centroid is found on the device (FP32 screen of
    8 candidates, FP64 choice; netgain_triton.nearest_other_all); the k-length state is copied to the host once and the
    release's C++ decision
    (net_gain_revival.native.decide, identical to the reference) chooses the moves; the driver's split applies them.
Options as in the release's CPU class: stat ("medg_pm" or "medg_fh"), early_stop ("hl" or None), and freeze (True:
no move once less than one pass remains, the pull request's freeze at 98deaca; False: moves until the early stop or the
end; None, the default: False on the head driver, True on 98deaca).
knn="faiss" (tests only) replaces the device search by the CPU classes' FP32 faiss.knn, for exact comparisons.

Two drivers (whichever faiss.contrib.e_means is on the path):
  * 98deaca: the driver freezes moves itself; without the freeze, a centroid moved in the last pass keeps its EMA
    position (its closing mean would mix rows from before the move).
  * the PR head 79440bc (HEAD): the driver has no freeze; a centroid moved in the last pass gets its rows sorted out at
    the end (the closing handover: rows it held before moving go to their second-nearest centroid or are assigned
    again, then every centroid returns the mean of its last-pass rows). The freeze (freeze=True) is then this class's
    own (98deaca's rule, copied); our device check records its moves for the handover as the driver's own check does,
    and the driver's second-nearest route of the handover is kept.
"""
import math
import time

import numpy as np
import torch

from faiss.contrib.e_means.e_means_torch import Emeans
from faiss.contrib.e_means.base import _scheduled_alpha, _same_operation
from net_gain_revival import native

import netgain_triton as NT

HEAD = hasattr(Emeans, "_handover_start")       # the PR head (79440bc): closing handover, no freeze in the driver


class NetGainEmeansFast(Emeans):
    """check: "device" (default on CUDA: the whole check on the device, captured as a CUDA graph; medg_pm only),
    "host" (device search, C++ decision on the host after one copy), or "both" (tests: both decisions on the same
    state, the host's applied; agreement counted in check_agree / check_total)."""

    def __init__(self, d, k, *a, stat="medg_pm", freeze=None, early_stop="hl", knn="device", check="device", **kw):
        assert stat in ("medg_pm", "medg_fh")
        assert early_stop in (None, "hl")
        assert knn in ("device", "faiss")
        assert check in ("device", "host", "both")
        super().__init__(d, k, *a, **kw)
        if not native.available():
            raise RuntimeError("net_gain_revival/native/libnetgain.so is required (native/build.sh)")
        # freeze=None: the default for the driver: no freeze on the head (moves until the early stop or the end; the
        # paper's default since version 1.1), the freeze on 98deaca (where a centroid moved in the last pass would keep
        # its EMA position). The paper's runs passed freeze explicitly.
        freeze = (not HEAD) if freeze is None else bool(freeze)
        self.stat, self.freeze, self.early_stop, self.knn, self.check = stat, freeze, early_stop, knn, check
        self._ng = None
        self.n_checks = self.n_candidates = 0
        self.rule_seconds = 0.0
        self.es_stop_step = None
        self.check_agree = self.check_total = 0
        self._ng_graph = self._ng_pool = None
        self.check_captured, self.capture_error = None, None
        self._es_event, self._es_stopped, self._device = None, False, None

    # The overrides below only add device work (no host reads), so the driver may capture and replay them.
    def _capture_operations_match(self, owner=None):
        return super()._capture_operations_match(NetGainEmeansFast if owner is None else owner)

    def _chunked_capture_matches(self, owner=None):
        return super()._chunked_capture_matches(NetGainEmeansFast if owner is None else owner)

    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        if self._unused_threshold is not None:           # the driver uses cost-based revival only for k > 1
            self._ng = None
            return
        dev, k = self._usage.device, self.k
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
        self._ng = z(5, k)                                   # rows, sum m, sum m^2, sum p^2, sum p^4 since birth
        self._cum_n, self._age, self._stale = z(k), z(k), z(k)
        self._scal = torch.tensor([1.0, 0.0], dtype=torch.float64, device=dev)   # [prod(1 - alpha), early-stop clock]
        self._decay_prod, self._es_logq = self._scal[0], self._scal[1]          # views
        self._es_stopped = False
        self.es_stop_step = None
        self._prev_delta = None
        # device check state
        self._alpha_dev, self._step_dev = z(()), z(())
        self._es_stopped_dev = torch.zeros((), dtype=torch.int32, device=dev)
        self._es_step_dev = torch.full((), -1, dtype=torch.int64, device=dev)
        self._ncand_dev = torch.zeros((), dtype=torch.int64, device=dev)
        self._check_splits = torch.zeros((), dtype=torch.int64, device=dev)    # the driver's device split counter
        self._ng_graph = self._ng_pool = None
        self._es_host = self._es_event = None
        self._device = None                                   # decided at the first check (the backend is set then)
        if self.check == "both":
            import netgain_device as ND
            self._decider = ND.DeviceDecider(k, dev)

    def _frozen(self):
        # (no event query while the driver captures a CUDA graph: it re-captures once frozen on large data)
        if (self._es_event is not None and not self._es_stopped and not torch.cuda.is_current_stream_capturing()
                and self._es_event.query()):
            # the device's early-stop flag, copied after the last check (no wait): stops the tracking
            self._es_stopped = bool(self._es_host.item())
        if getattr(self, "_es_stopped", False):
            return True
        return self.freeze and self._pr_frozen()

    def _pr_frozen(self):
        """The freeze of 98deaca (base._frozen there, copied: the head has none): no move once less than one pass
        remains after this batch."""
        rows = self.recipe["epoch_rows"]
        done = sum(rows[:self._epoch]) + min(rows[self._epoch], self._epoch_rows + self.recipe["batch_size"])
        return (self._unused_threshold is None and len(rows) > 1 and done >= self._n
                and sum(rows) - done < self._n)

    def _second_labels_supported(self):
        """(head) The driver's test, with this class as the owner: our _assign_rows passes every output through, so
        the closing handover keeps its second-nearest route (the plain class's)."""
        return getattr(self._execution_backend, "second_labels", False) and all(
            _same_operation(getattr(self, name), getattr(NetGainEmeansFast, name), self)
            for name in ("_assign_rows", "_accumulate_core"))

    def finalize(self):
        if self._ng is not None and self._device:
            self.n_candidates = int(self._ncand_dev)
            step = int(self._es_step_dev)
            self.es_stop_step = None if step < 0 else step
        self._ng_graph = self._ng_pool = None
        return super().finalize()

    # ------------------------------------------------------------------ the check on the device
    def _device_state(self):
        import netgain_triton as NTr
        C = self._readout(force=True)
        # nearest other centroid, its squared distance rounded to FP32 (as the reference's faiss.knn) and whether the
        # two are equal: the tiered search with its glue fused (netgain_triton.nearest_other_state)
        nn, delta, dup = NTr.nearest_other_state(C)
        norm = torch.linalg.vector_norm(self._removal_stats[:, :-1], dim=1, dtype=torch.float64)
        st = dict(usage=self._usage.double(), raw=(2.0 / math.pi) * norm, stale=self._stale, age=self._age,
                  cum=self._cum_n, m_n=self._ng[0], m_sum=self._ng[1], m_sq=self._ng[2], p2=self._ng[3],
                  p4=self._ng[4], nn=nn, delta=delta, dup=dup, decay_prod=self._decay_prod, alpha=self._alpha_dev,
                  step=self._step_dev, stopped=self._es_stopped_dev)
        return st, norm

    def _device_check_body(self):
        from faiss.contrib.e_means.triton_assignment import _write_float_splits, removal_split
        st, norm = self._device_state()
        src, tgt, m, nc = self._decider.decide(st)
        self._ncand_dev.add_(nc)
        _write_float_splits(self._centroids, self._usage, src, tgt, self._input_dtype, self._axes, norm, count=m)
        if HEAD:
            removal_split(self._removal_stats, self._axes, src, tgt, count=m)
        else:
            removal_split(self._removal_stats, self._axes, src, tgt, count=m, moved=self._moved)
        self._check_pairs = src, tgt, m                    # (replays refresh them in place)
        h = getattr(self, "_handover", None)
        # one launch: the split counter, both ends' statistics restart, (head) when each moved centroid moved in
        # closing-pass rows (as the driver's _removal_check), and the early-stop clock
        if getattr(self, "_post_cst", None) is None:
            self._post_cst = torch.tensor([math.log(0.5)], dtype=torch.float64, device=src.device)
        NT.ng_post(src, tgt, m, self._ng, self._cum_n, self._age, self._stale,
                   None if h is None else h["moved_at"], None if h is None else h["done"], self._check_splits,
                   self._scal, self._es_stopped_dev, self._es_step_dev, self._step_dev, self._post_cst,
                   es=self.early_stop == "hl")

    def _ng_device_check(self):
        """One check on the device: captured once as a CUDA graph (after one eager run that compiles the kernels),
        replayed at every later check; this check's step and alpha are written to device scalars first."""
        alpha = float(_scheduled_alpha(self.recipe["alpha0"], self._step, self.recipe["total_steps"]))
        self._alpha_dev.fill_(alpha)
        self._step_dev.fill_(float(self._step))
        h = getattr(self, "_handover", None)
        if h is not None:
            h["done"].fill_(h["rows"])                    # (head) closing-pass rows so far, counted on the host
        if self._ng_graph is None:
            self._device_check_body()
            pairs = self._check_pairs                     # this check's; the capture's are the replays'
            dev = self._usage.device
            current, stream = torch.cuda.current_stream(dev), torch.cuda.Stream(dev)
            graph, self._ng_pool = torch.cuda.CUDAGraph(), torch.cuda.MemPool()
            stream.wait_stream(current)
            try:
                with torch.cuda.stream(stream):
                    graph.capture_begin(pool=self._ng_pool.id, capture_error_mode="thread_local")
                    try:
                        self._device_check_body()
                    finally:
                        graph.capture_end()
                self._ng_graph, self._graph_pairs = graph, self._check_pairs
                self.check_captured = True
            except RuntimeError as e:
                self._ng_graph = False                               # not capturable here: eager checks
                self.check_captured = False
                self.capture_error = str(e)[:300]
            current.wait_stream(stream)
            self._check_pairs = pairs
        elif self._ng_graph:
            self._ng_graph.replay()
            self._check_pairs = self._graph_pairs
        else:
            self._device_check_body()
        if HEAD:
            src, tgt, m = self._check_pairs
            if h is not None:
                # this check's pairs (one copy), read once at the end of the pass by the driver's handover
                k = self.k
                ev = torch.cat((tgt, src, m.view(1)))
                h["events"].append((ev[:k], ev[k:2 * k], ev[2 * k], h["rows"]))
            elif self._pass_sums is not None:
                # no handover (as the driver's check): neither end keeps closing sums from before the move
                live = torch.arange(len(src), device=src.device) < m
                hit = torch.zeros(self.k + 1, dtype=torch.bool, device=src.device)
                hit[torch.where(live, src, self.k)] = True
                hit[torch.where(live, tgt, self.k)] = True
                self._pass_sums.masked_fill_(hit[:-1, None], 0)
                self._pass_counts.masked_fill_(hit[:-1], 0)
        self._invalidate_readout()
        if self._es_host is not None:
            self._es_host.copy_(self._es_stopped_dev, non_blocking=True)
            if self._es_event is None:
                self._es_event = torch.cuda.Event()
            self._es_event.record()

    # ------------------------------------------------------------------ statistics (captured with the step)
    def _assign_rows(self, x, centroids, centroid_norm=None, distance_out=None, labels_out=None, second_out=None,
                     second_labels_out=None):
        extra = {} if second_labels_out is None else {"second_labels_out": second_labels_out}   # (head only)
        distance, labels = super()._assign_rows(x, centroids, centroid_norm, distance_out, labels_out, second_out,
                                                **extra)
        # 98deaca asks for d2 exactly while moves are allowed and its removal statistics are kept; the head always
        # does (the statistics after a freeze or an early stop are then unused: no check reads them).
        if second_out is not None and self._ng is not None and len(x):
            NT.ng_accumulate(x, labels, distance, second_out, self._axes, self._readout(), self._ng)
        return distance, labels

    def _apply_statistics(self, ids, sums, counts, objective, alpha, decay, replace_usage=False):
        super()._apply_statistics(ids, sums, counts, objective, alpha, decay, replace_usage)
        if self._ng is None:
            return
        if self._stale.is_cuda:                      # one fused kernel (captured with the step)
            NT.ng_step(self._stale, self._cum_n, self._age, counts, ids, self._scal, decay, replace_usage, True)
            return
        if replace_usage:
            self._stale.zero_()
            self._decay_prod.zero_()
            self._es_logq.fill_(-math.inf)
        else:
            self._stale.mul_(decay)
            self._decay_prod.mul_(decay)
            if torch.is_tensor(decay):
                self._es_logq.add_(torch.log(decay.double()))
            else:
                self._es_logq.add_(math.log(decay) if decay > 0 else -math.inf)
        c = counts.double()
        if ids is None:
            self._cum_n.add_(c)
        else:
            self._cum_n.index_add_(0, ids, c)
        self._age.add_(1.0)

    # ------------------------------------------------------------------ the check (eager, between replays)
    def _choose_splits(self, rng):
        if self._ng is None:
            return super()._choose_splits(rng)
        if self._frozen():
            return None, None
        if self._device is None:
            dev = self._usage.device
            self._device = (self.check == "device" and self.stat == "medg_pm" and self.knn == "device"
                            and dev.type == "cuda" and self.d <= 1024 and self._input_dtype in (None, torch.float32, torch.bfloat16)
                            and type(self._execution_backend).__name__ in ("TritonBackend", "CuteBackend"))
            if self._device:
                import netgain_device as ND
                self._decider = ND.DeviceDecider(self.k, dev)
                if self.early_stop is not None:
                    self._es_host = torch.zeros((), dtype=torch.int32).pin_memory()
            self.device_check = self._device
        if self._device:
            self.n_checks += 1
            self._ng_device_check()
            return None, None
        dev_moves = None
        if self.check == "both":
            alpha = float(_scheduled_alpha(self.recipe["alpha0"], self._step, self.recipe["total_steps"]))
            self._alpha_dev.fill_(alpha)
            self._step_dev.fill_(float(self._step))
            st, _norm = self._device_state()
            s_d, t_d, m_d, _nc = self._decider.decide(st)
            md = int(m_d)
            dev_moves = (s_d[:md].cpu().numpy(), t_d[:md].cpu().numpy())
        k = self.k
        C = self._readout().float().contiguous()
        if C.is_cuda:
            torch.cuda.synchronize(C.device)          # rule_seconds then measures the check alone
        t0 = time.perf_counter()
        if self.knn == "faiss":
            import faiss
            Ch = np.ascontiguousarray(C.cpu().numpy())
            D, I = faiss.knn(Ch, Ch, 2)
            D, I = np.ascontiguousarray(D, dtype=np.float32), np.ascontiguousarray(I, dtype=np.int64)
            packed = torch.cat((self._usage.double()[None], self._stale[None], self._age[None], self._cum_n[None],
                                self._ng, torch.stack((self._decay_prod, self._es_logq)).expand(k, 2).T)).cpu()
        else:
            nn, delta = NT.nearest_other_all(C)
            packed = torch.cat((self._usage.double()[None], self._stale[None], self._age[None], self._cum_n[None],
                                self._ng, torch.stack((self._decay_prod, self._es_logq)).expand(k, 2).T,
                                nn.double()[None], delta[None])).cpu()
        h = packed.numpy()
        usage, stale, age, cum = (np.ascontiguousarray(h[i]) for i in range(4))
        m_n, m_sum, m_sq, p2, p4 = (np.ascontiguousarray(h[4 + i]) for i in range(5))
        decay_prod, es_logq = float(h[9, 0]), float(h[10, 0])
        if self.knn != "faiss":
            nn_h = h[11].astype(np.int64)
            D = np.zeros((k, 2), dtype=np.float32)
            D[:, 1] = h[12]
            I = np.ascontiguousarray(np.stack((np.arange(k, dtype=np.int64), nn_h), axis=1))
        rs = np.ascontiguousarray(self._removal_stats.float().cpu().numpy())
        readout = np.ascontiguousarray(C.cpu().numpy())
        alpha = float(_scheduled_alpha(self.recipe["alpha0"], self._step, self.recipe["total_steps"]))
        alpha = min(max(alpha, 1e-6), 1.0)
        src, tgt, delta_h, info = native.decide(usage, rs, 2.0 / np.pi, stale, age, cum, decay_prod, alpha, self._step,
                                                m_n, m_sum, m_sq, p2, p4, D, I, self.stat, self._prev_delta, readout)
        self._prev_delta = delta_h
        self.n_checks += 1
        self.n_candidates += info["n_candidates"]
        if dev_moves is not None:
            hs = np.zeros(0, np.int64) if src is None else src
            ht = np.zeros(0, np.int64) if tgt is None else tgt
            self.check_total += 1
            self.check_agree += int(np.array_equal(hs, dev_moves[0]) and np.array_equal(ht, dev_moves[1]))
        if self.early_stop == "hl":
            if src is not None:
                self._es_logq.zero_()
            elif es_logq <= math.log(0.5):
                self._es_stopped = True
                self.es_stop_step = int(self._step)
        self.rule_seconds += time.perf_counter() - t0
        if src is None:
            return None, None
        dev = self._usage.device
        return torch.from_numpy(src).to(dev), torch.from_numpy(tgt).to(dev)

    def _apply_splits(self, source, target):
        n = super()._apply_splits(source, target)
        if self._ng is None:
            return n
        for ids in (source, target):
            self._ng[:, ids] = 0.0
            self._cum_n[ids] = 0.0
            self._age[ids] = 0.0
            self._stale[ids] = 1.0
        return n
