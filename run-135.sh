#!/bin/zsh
# Launch the kangaroo solver against Bitcoin puzzle #135.
# Runs detached (survives terminal close) and logs to run-135.log.
# The instant a key is found it is written to FOUND.txt.

cd "$(dirname "$0")"

PUB=02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
L=4000000000000000000000000000000000          # 2^134
R=7fffffffffffffffffffffffffffffffff          # 2^135 - 1
THREADS=${1:-10}                              # default: all 10 cores

echo "Starting puzzle-135 kangaroo on $THREADS threads. Log: run-135.log"
echo "Watch:  tail -f run-135.log"
echo "Stop:   btc135-stop   (or ./stop-135.sh)"
# dpbits=25 -> a distinguished point every ~33M jumps (~2-3 net markers/sec)
# slots_log2=26 -> ~64M-slot table (~4 GB) used as the collision net
nohup ./kangaroo solve $PUB $L $R $THREADS 25 26 >> run-135.log 2>&1 &
echo "PID $! (also in run-135.pid)"
echo $! > run-135.pid
