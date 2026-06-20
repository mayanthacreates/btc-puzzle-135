# GREENROO — CPU + Metal GPU Kangaroo (Apple Silicon)

A distinguished-point Pollard kangaroo interval-ECDLP solver for secp256k1,
hand-built and tuned for the M4. Runs the **CPU (10 cores) and the Metal GPU
together** as one engine, sharing a single distinguished-point net — either can
land the collision. Live green dashboard.

Measured: CPU ~80 M/s + GPU ~130 M/s = **~200+ million keys/sec combined**.
Validated end-to-end on a real solved puzzle (#40 — recovered the exact
published private key).

```
  ┌─ GREENROO ──────────────────── PUZZLE #135 ─┐
  │ uptime 13s
  │ CPU 10 cores     73 M/s ███████░░░░░
  │ GPU 524288 roos 133 M/s ████████████
  │ TOTAL      206 M keys/sec
  │ checked  1.68 billion
  │ DP net   59 markers
  └────────────────────────────────────────────┘
```

## Purpose & honest scope
This is an educational, readable, correctly-validated CPU implementation, aimed
at the **Bitcoin Puzzle challenge** - a set of addresses the creator deliberately
funded as a public cryptographic challenge for anyone to solve. It is not a tool
for attacking third-party wallets, and it provides no advantage against normal
256-bit Bitcoin keys.

Be realistic: even at ~200 million keys/sec **this cannot** solve puzzle #135
(its 2^134 range needs ~2^67.5 operations - astronomically out of reach on any
personal machine; large NVIDIA GPU pools are far faster and still treat #135 as
a long shot). The value here is a clean, native-Apple-Silicon CPU+GPU kangaroo
that you can read, trust, learn from, and use on the genuinely reachable
(smaller) puzzles. MIT licensed.

## What it is
- Correct, validated secp256k1 field + group arithmetic. CPU side (4×64 limbs,
  `__uint128_t`) verified vs Python bignum + libsecp256k1; GPU side (Metal,
  8×32 limbs) verified to match the CPU bit-for-bit over thousands of cases.
- **Two engines, one net:** CPU pthreads and a Metal GPU kernel both walk
  kangaroos (Montgomery batch inversion on each) into a single shared
  distinguished-point table — a GPU tame can collide with a CPU wild.
- Proven to actually solve: random 36–56-bit self-tests *and* the real solved
  puzzle #40 (recovered the exact published key) end-to-end.

## Measured speed
Both engines run together and share one distinguished-point net:

| engine | rate |
|---|---|
| CPU (10 cores, batch-inverted jump loop) | ~80 M keys/s |
| GPU (M4 Metal, 128 kangaroos/thread, batch inversion) | ~130 M keys/s |
| **combined** | **~200+ M keys/s** |

The GPU seeds its herds for ~8 s at startup before it kicks in (you'll see it
go from 0 to ~130 in the dashboard). Run `NOGPU=1 ./kangaroo solve …` to force
CPU-only.

## Setup on a new Mac (Apple Silicon)
```
brew install secp256k1 gmp     # one-time dependencies
make kangaroo                  # build the solver (only this is needed to run)
```
`make` (all targets) also builds the validators, which need the libraries above.

## Build
```
make            # builds kangaroo + the two validators
make check      # re-runs field + group validation
make kangaroo   # just the solver (enough to run the bot)
```

## Use
Self-test (proves correctness on a solvable size):
```
./kangaroo selftest 50 10          # random 50-bit key, 10 threads
```
Solve an arbitrary exposed-pubkey interval:
```
./kangaroo solve <compressed_pubkey_hex> <Lhex> <Rhex> [threads] [dpbits] [slots_log2]
```
Puzzle 135 (convenience launcher, runs detached, logs to `run-135.log`):
```
./run-135.sh
```
A found key is printed and written to `FOUND.txt` immediately.

## Targeting a different puzzle
This solver works on **any puzzle whose public key is exposed** (i.e. the address
has spent at least once). You need three things: the compressed public key, and
the key range.

For puzzle number `N`, the range is always:
```
L = 2^(N-1)        R = 2^N - 1
```
e.g. puzzle 40 -> L=`8000000000`, R=`ffffffffff`; puzzle 135 -> L=`4000...000`
(34 hex chars), R=`7fff...fff`.

Easiest way: open `run-135.sh` and edit the three values at the top (`PUB`, `L`,
`R`), then run it. Or call the binary directly:
```
./kangaroo solve <compressed_pubkey_hex> <Lhex> <Rhex> [threads]
```
Tip: to watch it actually *find* a key (proof it works), point it at a small,
already-reachable puzzle, or run `./kangaroo selftest 50`. Larger puzzles like
#135 run correctly but will not finish (see scope note above).

Derive a compressed pubkey from a private key (for testing):
```
./kangaroo pub <privkey_hex>
```

## Checkpoint / resume
In `solve` mode the distinguished-point table is saved to `checkpoint.bin` every
120 s and on exit. Restarting the same target automatically resumes from it
(restoring all stored DPs and prior jump count); a checkpoint for a different
target/range is ignored. Override the save interval with `CKPT_SEC=<seconds>`.
So a reboot, crash, or `pkill` loses at most ~2 minutes of progress.

## Honest expectation for #135
The engine is correct and runs at full speed against the real #135 key. But the
interval is 2^134 wide, so the expected work is ~2^67.5 jumps. At ~2^26 jumps/sec
this is on the order of 10^5–10^6 years on this machine. No CPU-only tuning
changes that exponent — it is the proven generic-group lower bound, not an
implementation limit. The same binary will, however, efficiently solve the
smaller (genuinely reachable) puzzles in the 50–80-bit range.

## Files
- `field.h` / `group.h` — CPU Fp arithmetic + EC group ops, pubkey decompression
- `kangaroo.c` — solver: CPU engine, shared DP table, threads, dashboard, GPU bridge
- `gpu_field.metal` — Metal GPU kernels (field/EC arithmetic + kangaroo)
- `gpu_driver.m` — GPU engine thread (seeds, launches, feeds the shared net)
- `gpu_bridge.h` — C↔Metal interface
- `test_field.c` / `check_field.py` — CPU field validation
- `test_group.c` — CPU group validation vs libsecp256k1
- `gpu_test.m` — GPU arithmetic validation vs CPU (`make check-gpu`)
- `gpu_solve.m` — standalone GPU-only solver (benchmark/diagnostic)
