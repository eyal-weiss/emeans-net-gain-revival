"""A/B timing of the device decision, fused (netgain_fused) vs torch estimates: both captured as CUDA graphs and
replayed alternately (the same contention for both), minimum and median over replays.
usage: python bench_ab.py K[,K...] [REPS]"""
import statistics
import sys
import time

import torch

import netgain_device as ND
from bench_decide import state, dev

ks = [int(x) for x in sys.argv[1].split(",")]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 200


def graph_of(fn):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        g.capture_begin()
        fn()
        g.capture_end()
    torch.cuda.current_stream().wait_stream(s)
    return g


for k in ks:
    st, C = state(k)
    import os
    a, b = ND.DeviceDecider(k, dev, fused=False), ND.DeviceDecider(k, dev, fused=True)
    os.environ["NG_PM_REPLAY"], os.environ["NG_PM_PRE32"] = "1", "0"       # the previous decision
    ga = graph_of(lambda: a.decide(st))
    os.environ["NG_PM_REPLAY"], os.environ["NG_PM_PRE32"] = "0", "1"       # the new one
    gb = graph_of(lambda: b.decide(st))
    ta, tb = [], []
    for _ in range(reps):
        for g, t in ((ga, ta), (gb, tb)):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            t.append(1e3 * (time.perf_counter() - t0))
    print(f"k={k:6d}  previous (torch estimates, replayed PM): min {min(ta):6.3f} median {statistics.median(ta):6.3f} ms | new (fused, Newton PM): min "
          f"{min(tb):6.3f} median {statistics.median(tb):6.3f} ms", flush=True)
