"""Round 3 (confirm3/PROTOCOL.md), Part F: prhead_grid.py on new seeds; this file only adds the part definitions.
  F2 : 34 cells x seeds 200-259, T = 2      F5 : 34 cells x seeds 200-239, T = 5
usage (launch_r3.sh runs F2, then F5 once F2 has ended without error):
  FAISS_OVERLAY=<faiss_head_79440bc> python -u round3_prhead.py --host H --out DIR --parts F2 \
      --configs pr,ngh:medg_pm:freeze:guard,ngh:medg_pm:nofreeze:guard [--workers 8]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import confirm2_grid as C2  # noqa: E402
import prhead_grid as PG  # noqa: E402

PG.PARTS["F2"] = (C2.CELLS34, list(range(200, 260)), 2)
PG.PARTS["F5"] = (C2.CELLS34, list(range(200, 240)), 5)

if __name__ == "__main__":
    PG.main()
