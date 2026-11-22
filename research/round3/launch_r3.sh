#!/bin/bash
# Third confirmatory round (confirm3/PROTOCOL.md), seeds 200-299 (gate: 30-31). On each host, in order:
#   G  : interval_grid.py, medg_pm@5, seeds 30-31          -> r3_gate     then r3_gatecheck.py (the host stops if it fails)
#   I  : interval_grid.py, medg_pm@1,2,5,10,20, seeds 200-299 -> r3_interval
#   F2 : round3_prhead.py (driver 79440bc), pr + ours freeze/nofreeze (guard), seeds 200-259, T=2 -> r3_prhead
#   F5 : as F2, seeds 200-239, T=5                            -> r3_prhead
# Each step starts only after the previous one ended with rc=0. Resumable: finished runs are kept; launching again
# continues. usage: setsid nohup bash launch_r3.sh > /dev/null 2>&1 < /dev/null &     stop: kill -TERM -- -$(cat r3.pgid)
case "$(hostname)" in
  host_a*) HOST=host_a; BASE=$HOME/emeans ;;
  host_b*)  HOST=host_b; BASE=$HOME/emeans ;;
  *) echo "unknown host"; exit 1 ;;
esac
W=8      # 8 workers x 4 threads = 32 threads
cd $BASE/pr5915
exec 8> r3.lock
flock -n 8 || { echo "$(date) already running" >> r3_stage.txt; exit 1; }
echo $$ > r3.pgid
export DISTANCE_REVIVAL_DATA=$BASE/data EMEANS_RR_DATA=$BASE/data EXPERIMENTS_DIR=$BASE/code/experiments NGP_IMPL=cpp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
PY="nice -n 10 $HOME/miniforge3/envs/emeans_pr5915/bin/python -u"
step() { echo "$(date) $1 start" >> r3_stage.txt; local N=$1; shift; "$@"; RC=$?
         echo "$(date) $N finished rc=$RC" >> r3_stage.txt; return $RC; }
SEEDS=$(seq -s, 200 299)
HEAD="env FAISS_OVERLAY=$BASE/pr5915/faiss_head_79440bc $PY round3_prhead.py --host $HOST --out $BASE/pr5915/r3_prhead --workers $W"
NGH=pr,ngh:medg_pm:freeze:guard,ngh:medg_pm:nofreeze:guard
step G env FAISS_OVERLAY= $PY interval_grid.py --host $HOST --out $BASE/pr5915/r3_gate --round3 --seeds 30,31 \
       --configs medg_pm@5 --workers $W >> r3_log.txt 2>&1 \
  && step GC $PY r3_gatecheck.py --host $HOST --gate $BASE/pr5915/r3_gate --round2 $BASE/pr5915/cfm2_grid >> r3_log.txt 2>&1 \
  && step I env FAISS_OVERLAY= $PY interval_grid.py --host $HOST --out $BASE/pr5915/r3_interval --round3 --seeds $SEEDS \
       --configs medg_pm@1,medg_pm@2,medg_pm@5,medg_pm@10,medg_pm@20 --workers $W >> r3_log.txt 2>&1 \
  && step F2 $HEAD --parts F2 --configs $NGH >> r3_log.txt 2>&1 \
  && step F5 $HEAD --parts F5 --configs $NGH >> r3_log.txt 2>&1 \
  && echo "$(date) all done" >> r3_stage.txt
