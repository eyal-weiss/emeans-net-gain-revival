#!/bin/bash
# Final comparison on the released implementation (final_grid.py): parts R (reproduction of round 3 without the early
# stop), F2 and F5, in order, on this host's datasets (round-2 placement). 8 workers x 4 threads = 32 threads, nice 10.
# Resumable: launching again continues. usage: setsid nohup bash final_launch.sh > /dev/null 2>&1 < /dev/null &
case "$(hostname)" in
  host_a*) HOST=host_a; BASE=$HOME/emeans ;;
  host_b*)  HOST=host_b;  BASE=$HOME/emeans ;;
  *) echo "unknown host"; exit 1 ;;
esac
cd $BASE/final_cpp
exec 8> fin.lock
flock -n 8 || { echo "$(date) already running" >> fin_stage.txt; exit 1; }
echo $$ > fin.pgid
export DISTANCE_REVIVAL_DATA=$BASE/data EMEANS_RR_DATA=$BASE/data EXPERIMENTS_DIR=$BASE/code/experiments
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
export FAISS_OVERLAY=$BASE/pr5915/faiss_head_79440bc PYTHONPATH=$BASE/final_cpp/pkg:$BASE/pr5915:$BASE/code/experiments
echo "$(date) start" >> fin_stage.txt
nice -n 10 $HOME/miniforge3/envs/emeans_pr5915/bin/python -u final_grid.py --host $HOST --out $BASE/final_cpp/fin \
    --parts R,F2,F5 --workers 8 --threads 4 >> fin_log.txt 2>&1
RC=$?
echo "$(date) finished rc=$RC" >> fin_stage.txt
echo "rc=$RC" > FIN_DONE
