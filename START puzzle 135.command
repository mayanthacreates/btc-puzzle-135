#!/bin/zsh
# Double-click to start GREENROO with the live green dashboard.
# Closing this window stops the bot; progress is saved to checkpoint.bin
# every 2 minutes and resumes automatically next time you start.
cd "$(dirname "$0")"

PUB=02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
L=4000000000000000000000000000000000          # 2^134
R=7fffffffffffffffffffffffffffffffff          # 2^135 - 1

clear
exec ./kangaroo solve $PUB $L $R 10 25 26
