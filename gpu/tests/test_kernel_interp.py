"""The Triton statistics kernel against the PyTorch reference, in Triton's CPU interpreter (TRITON_INTERPRET=1)."""
import os
assert os.environ.get("TRITON_INTERPRET") == "1"
import torch
import triton
import netgain_triton as NT

torch.manual_seed(0)
worst = 0.0
for n, k, d in ((500, 7, 5), (2000, 33, 17), (3000, 64, 96)):
    rows = torch.randn(n, d)
    centroids = torch.randn(k, d)
    axes = torch.nn.functional.normalize(torch.randn(k, d), dim=1)
    labels = torch.randint(0, k, (n,))
    labels[labels == 3] = 2                                   # an empty centroid
    distance = torch.rand(n) * 4
    second = distance + torch.rand(n) * 3
    second[::97] = float("inf")                               # non-finite margins count as 0
    second[::101] = float("nan")
    ref = NT.ng_accumulate_torch(rows, labels, distance, second, axes, centroids, torch.zeros(5, k, dtype=torch.float64))
    out = torch.zeros(5, k, dtype=torch.float64)
    dtype = torch.int16
    sorted_labels, order = torch.sort(labels.to(dtype), stable=True)
    offsets = torch.searchsorted(sorted_labels, torch.arange(k + 1, dtype=dtype))
    bd = triton.next_power_of_2(d)
    NT._ng_stats_kernel[(k,)](rows, order, offsets, distance, second, axes, centroids, out, k, d, rows.stride(0),
                              BLOCK_R=8, BLOCK_D=bd)
    rel = ((out - ref).abs() / ref.abs().clamp_min(1e-300)).max().item()
    worst = max(worst, rel)
    print(f"n={n} k={k} d={d}: counts equal {torch.equal(out[0], ref[0])}, max relative difference {rel:.2e}")
print("PASS" if worst < 1e-12 else "FAIL", f"(worst {worst:.2e})")
