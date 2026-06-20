#!/bin/zsh
# Double-click this file to start the puzzle-135 bot.
cd "$(dirname "$0")"
./run-135.sh
echo ""
echo "======================================================================"
echo " Live progress is below. You can CLOSE THIS WINDOW anytime -"
echo " the bot keeps running in the background until you Stop it."
echo "======================================================================"
echo ""
sleep 1
tail -f run-135.log
