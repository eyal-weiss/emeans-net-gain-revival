"""Time of one device decision (DeviceDecider.decide) and of the nearest-other search, each captured as a CUDA graph
and replayed (minimum over replays: robust to a shared GPU), on random states of k centres.
usage: python bench_decide.py K[,K...] [REPS]"""
import sys
import time

import torch

import netgain_device as ND
import netgain_triton as NT

ks = [int(x) for x in sys.argv[1].split(",")] if __name__ == "__main__" else []
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 50
dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)


def state(k, d=96):
    u = lambda *s: torch.rand(*s, generator=g, device=dev, dtype=torch.float64)
    m_n = torch.floor(u(k) * 200)
    m_sum = m_n * u(k)
    st = dict(usage=torch.floor(u(k) * 1000), raw=u(k) * 3, stale=(u(k) < 0.05).double(), age=torch.floor(u(k) * 300),
              cum=torch.floor(u(k) * 3e4), m_n=m_n, m_sum=m_sum, m_sq=m_sum * m_sum / m_n.clamp_min(1) * 1.3,
              p2=m_n * u(k), p4=m_n * u(k) * 3, decay_prod=torch.tensor(0.3, dtype=torch.float64, device=dev),
              alpha=torch.tensor(0.01, dtype=torch.float64, device=dev),
              step=torch.tensor(500.0, dtype=torch.float64, device=dev),
              stopped=torch.zeros((), dtype=torch.int32, device=dev))
    C = torch.randn(k, d, generator=g, device=dev)
    nn, delta = NT.nearest_other_all(C)
    st.update(nn=nn, delta=delta.float().double(), dup=(C == C[nn]).all(dim=1).to(torch.int8))
    return st, C


def timed(fn):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        graph.capture_begin()
        fn()
        graph.capture_end()
    torch.cuda.current_stream().wait_stream(s)
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return 1e3 * best


for k in ks:
    st, C = state(k)
    dd = ND.DeviceDecider(k, dev)
    t_dec = timed(lambda: dd.decide(st))
    t_nn = timed(lambda: NT.nearest_other_all(C))
    lg = torch.log(st["raw"] + 0.1)
    s2 = torch.rand(k, generator=g, device=dev, dtype=torch.float64) + 0.05
    pos = torch.ones(k, dtype=torch.bool, device=dev)
    t_pm = timed(lambda: ND.paule_mandel_device(lg, s2, pos))
    print(f"k={k:6d}  decide {t_dec:7.3f} ms  (of which Paule-Mandel ~{t_pm:6.3f} ms)  nearest_other_all {t_nn:7.3f} ms",
          flush=True)
