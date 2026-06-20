#!/usr/bin/env python3
import sys
P = 2**256 - 2**32 - 977
ok = 0; bad = 0; firstbad = None
for line in sys.stdin:
    parts = line.split()
    if len(parts) != 4: continue
    op, a, b, r = parts
    a = int(a,16); b = int(b,16); r = int(r,16)
    if op == 'ADD': exp = (a+b) % P
    elif op == 'SUB': exp = (a-b) % P
    elif op == 'MUL': exp = (a*b) % P
    elif op == 'INV': exp = pow(a, P-2, P)
    else: continue
    if exp == r: ok += 1
    else:
        bad += 1
        if firstbad is None:
            firstbad = (op, hex(a), hex(b), 'got', hex(r), 'exp', hex(exp))
print(f"OK={ok} BAD={bad}")
if firstbad: print("FIRST BAD:", firstbad)
sys.exit(1 if bad else 0)
