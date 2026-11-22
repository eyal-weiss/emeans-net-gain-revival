"""Re-verification of the optimized decision (fused estimates, Newton Paule-Mandel; 2026-11-21 midday) against the
records of the previous code (final-centroid sha per run) and the head proof of concept; decision agreement with C++.
usage: python check_v2.py HEAD_DIR(results/gpu_head) FINAL_DIR(results/gpu_final) NEW_DIR(t3 records)"""
import collections, json, os, sys

head_dir, final_dir, new_dir = sys.argv[1:4]


def load(path):
    R = collections.defaultdict(dict)
    if not os.path.exists(path):
        return R
    for line in open(path):
        if line.startswith("{"):
            r = json.loads(line)
            R[(r["dataset"], r["k"], r["seed"], r["T"])][r["config"]] = r
    return R


def merge(*paths):
    out = collections.defaultdict(dict)
    for p in paths:
        for key, v in load(p).items():
            out[key].update(v)
    return out


def compare(label, new, new_cfg, old, old_cfg):
    n = same = 0
    diff = []
    for key, v in sorted(new.items()):
        if new_cfg in v and old_cfg in old.get(key, {}):
            n += 1
            if v[new_cfg]["sha"] == old[key][old_cfg]["sha"]:
                same += 1
            else:
                diff.append((key, v[new_cfg]["n_splits"], old[key][old_cfg]["n_splits"]))
    print(f"{label}: {same}/{n} identical" + ("" if not diff else f"; differ: {diff}"))


def agree(label, R, cfg):
    a = sum(v[cfg]["check_agree"] for v in R.values() if cfg in v)
    t = sum(v[cfg]["check_total"] for v in R.values() if cfg in v)
    print(f"{label}: device decision = C++ decision in {a}/{t} checks")


old_head = merge(os.path.join(head_dir, "eq_T2.jsonl"), os.path.join(head_dir, "eq_T5.jsonl"))
new_head = merge(os.path.join(new_dir, "head_T2.jsonl"), os.path.join(new_dir, "head_T5.jsonl"))
print("== head driver (79440bc)")
for poc, fast in (("poch0", "fdev_e"), ("pochn0", "fdevn_e"), ("poch", "fdevh_e"), ("pochn", "fdevnh_e")):
    compare(f"new {fast} vs proof of concept {poc}", new_head, fast, old_head, poc)
for cfg in ("fdev_e", "fdevn_e", "fdevh_e", "fdevnh_e", "fdev", "fdevn"):
    compare(f"new {cfg} vs previous code", new_head, cfg, old_head, cfg)
compare("new fboth (C++ applied) vs new fdev", new_head, "fboth", new_head, "fdev")
compare("new fbothn (C++ applied) vs new fdevn", new_head, "fbothn", new_head, "fdevn")
agree("head, freeze", new_head, "fboth")
agree("head, no freeze", new_head, "fbothn")

print("== 98deaca driver")
A = load(os.path.join(final_dir, "A_emeans.jsonl"))
C = load(os.path.join(final_dir, "C_large_em.jsonl"))
compare("new ng vs part A ng (previous code)", load(os.path.join(new_dir, "v98_A.jsonl")), "ng", A, "ng")
compare("new ng vs part C ng (previous code)", load(os.path.join(new_dir, "v98_C.jsonl")), "ng", C, "ng")
agree("98deaca, freeze, no early stop", load(os.path.join(new_dir, "v98_A.jsonl")), "fboth")

print("== o-means")
B = load(os.path.join(final_dir, "B_omeans.jsonl"))
Co = load(os.path.join(final_dir, "C_large_om.jsonl"))
compare("new om_ng vs part B om_ng (previous code)", load(os.path.join(new_dir, "om_B.jsonl")), "om_ng", B, "om_ng")
compare("new om_ng vs part C om_ng (previous code)", load(os.path.join(new_dir, "om_C.jsonl")), "om_ng", Co, "om_ng")
agree("o-means, no early stop", load(os.path.join(new_dir, "om_B.jsonl")), "om_both")
