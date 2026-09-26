#!/bin/zsh
# Launch the kangaroo solver against Bitcoin puzzle #140.
# Runs detached (survives terminal close) and logs to run-140.log.
# The instant a key is found it is written to FOUND.txt.

cd "$(dirname "$0")"

# ============================================================================
#  TO TARGET A DIFFERENT PUZZLE, edit these three lines.
#  For puzzle N:  L = 2^(N-1),  R = 2^N - 1  (in hex). PUB = its public key.
#  The puzzle must have an EXPOSED public key (its address has spent before).
# ============================================================================
PUB=031f6a332d3c5c4f2de2378c012f429cd109ba07d69690c6c701b6bb87860d6640
L=80000000000000000000000000000000000          # 2^139      (puzzle 140 start)
R=fffffffffffffffffffffffffffffffffff          # 2^140 - 1  (puzzle 140 end)
# ============================================================================
THREADS=${1:-10}                              # default: all 10 cores

echo "Starting puzzle-140 kangaroo on $THREADS threads. Log: run-140.log"
echo "Watch:  tail -f run-140.log"
echo "Stop:   btc140-stop   (or ./stop-140.sh)"
# dpbits=25 -> a distinguished point every ~33M jumps (~2-3 net markers/sec)
# slots_log2=26 -> ~64M-slot table (~4 GB) used as the collision net
nohup ./kangaroo solve $PUB $L $R $THREADS 25 26 >> run-140.log 2>&1 &
echo "PID $! (also in run-140.pid)"
echo $! > run-140.pid
