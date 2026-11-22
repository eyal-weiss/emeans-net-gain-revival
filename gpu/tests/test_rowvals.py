import torch, netgain_triton as NT
torch.manual_seed(0)
dev = torch.device("cuda")
for B, k, d in ((1000, 17, 9), (5000, 64, 38), (4096, 256, 96)):
    rows = torch.randn(B, d, device=dev)
    C = torch.randn(k, d, device=dev)
    axes = torch.nn.functional.normalize(torch.randn(k, d, device=dev), dim=1).contiguous()
    lab = torch.randint(0, k, (B,), device=dev)
    dist = torch.rand(B, device=dev); sec = dist + torch.rand(B, device=dev)
    _, v_ref, _, p_ref = NT.row_values(rows, lab, dist, sec, axes, C)
    v, p = NT.row_values_fused(rows, lab, dist, sec, axes, C)
    torch.cuda.synchronize()
    print(B, k, d, "max |v - ref| per column:", [f"{x:.2e}" for x in (v - v_ref).abs().max(0).values.tolist()],
          "p:", f"{(p - p_ref).abs().max().item():.2e}")
    bad = (v - v_ref).abs().max(1).values.argmax().item()
    print("   worst row", bad, "fused", v[bad].tolist(), "ref", v_ref[bad].tolist())
