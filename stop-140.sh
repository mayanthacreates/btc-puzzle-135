#!/bin/zsh
# Gracefully stop the puzzle-140 bot: it saves checkpoint.bin, then exits.
cd "$(dirname "$0")"

PID=""
[ -f run-140.pid ] && PID=$(cat run-140.pid)

if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "Stopping puzzle-140 bot (PID $PID) - saving checkpoint..."
    kill -TERM "$PID"
    for i in {1..30}; do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
    echo "Stopped. Progress is in checkpoint.bin (relaunch to resume)."
    rm -f run-140.pid
elif pgrep -f "kangaroo solve" >/dev/null; then
    echo "Stopping running solver (no PID file)..."
    pkill -TERM -f "kangaroo solve"
    sleep 2
    echo "Stopped. Progress is in checkpoint.bin."
    rm -f run-140.pid
else
    echo "No puzzle-140 bot is running."
    rm -f run-140.pid
fi
