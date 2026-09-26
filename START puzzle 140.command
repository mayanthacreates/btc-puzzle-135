#!/bin/zsh
# Double-click to start GREENROO with the live green dashboard.
# Closing this window stops the bot; progress is saved to checkpoint.bin
# every 2 minutes and resumes automatically next time you start.
cd "$(dirname "$0")"

PUB=031f6a332d3c5c4f2de2378c012f429cd109ba07d69690c6c701b6bb87860d6640
L=80000000000000000000000000000000000          # 2^139
R=fffffffffffffffffffffffffffffffffff          # 2^140 - 1

clear
exec ./kangaroo solve $PUB $L $R 10 25 26
