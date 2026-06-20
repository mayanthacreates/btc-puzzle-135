#!/bin/zsh
# Double-click this ONCE on a new Apple Silicon Mac to install deps + build.
cd "$(dirname "$0")"

echo "=================================================="
echo "   Setting up the Puzzle-135 bot on this Mac"
echo "=================================================="
echo ""

# Make Homebrew reachable even on a fresh shell
[ -x /opt/homebrew/bin/brew ] && eval "$(/opt/homebrew/bin/brew shellenv)"

# 1. Xcode command-line tools (provides clang / make)
if ! xcode-select -p >/dev/null 2>&1; then
    echo "Installing Apple command-line tools (a system window will pop up)..."
    xcode-select --install
    echo "Finish that installer, then double-click this SETUP file again."
    echo ""; echo "Press any key to close."; read -k1; exit 0
fi

# 2. Homebrew
if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required but not installed."
    echo "Paste this into Terminal, let it finish, then run SETUP again:"
    echo ""
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    echo ""; echo "Press any key to close."; read -k1; exit 1
fi

# 3. Dependencies
echo "Installing dependencies (secp256k1, gmp)..."
brew install secp256k1 gmp

# 4. Build
echo ""
echo "Building the solver..."
if ! make kangaroo; then
    echo ""; echo "Build failed - see the messages above."
    echo "Press any key to close."; read -k1; exit 1
fi

echo ""
echo "=================================================="
echo "   Setup complete!"
echo "   Now double-click:  START puzzle 135.command"
echo "=================================================="
echo ""
echo "Press any key to close."
read -k1
