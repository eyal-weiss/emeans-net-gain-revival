"""A/B timing of nearest_other_all: TF32 screen + compacted second tier (default) vs TF32x3 screen + the previous
uncompacted second tier (netgain_triton_prev, the code before 2026-11-21 midday); graphs replayed alternately.
usage: python bench_nn.py"""
import os, statistics, time
import torch
import netgain_triton as NT
import netgain_triton_prev as NP
dev = torch.device("cuda")
g = torch.Generator(device="cpu").manual_seed(0)


def graph_of(fn):
    fn(); torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        gr.capture_begin(); fn(); gr.capture_end()
    torch.cuda.current_stream().wait_stream(s)
    return gr


if __name__ == "__main__":
  for name, k, d, near in (("gaussian", 16384, 96, False), ("near-dup", 16384, 96, True), ("gaussian", 4096, 96, False),
                           ("gaussian", 4096, 784, False), ("gaussian", 1024, 96, False), ("gaussian", 16384, 8, False)):
      C = torch.randn(k, d, generator=g)
      if near:
          C = torch.cat([C[:k // 2], C[:k // 2] + 1e-4 * torch.randn(k // 2, d, generator=g)])
      C = C.to(dev).contiguous()
      ga = graph_of(lambda: NP.nearest_other_all(C))
      gb = graph_of(lambda: NT.nearest_other_all(C))
      ta, tb = [], []
      for _ in range(100):
          for gr, t in ((ga, ta), (gb, tb)):
              torch.cuda.synchronize(); t0 = time.perf_counter(); gr.replay(); torch.cuda.synchronize()
              t.append(1e3 * (time.perf_counter() - t0))
      print(f"{name:9s} k={k:6d} d={d:4d}: previous min {min(ta):7.3f} median {statistics.median(ta):7.3f} ms | "
            f"new min {min(tb):7.3f} median {statistics.median(tb):7.3f} ms", flush=True)
