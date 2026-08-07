#!/usr/bin/env python3
"""Measure the exact/approximate sizes of every search class with geng -u.

Big classes are estimated from a 1/SAMPLE_MOD res/mod slice (geng splits
its search tree, so slices are roughly uniform — good to the order of
magnitude we need for scheduling). Writes results/hadwiger/classes.jsonl.
"""

import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hadwiger.runner_util import jsonl_append, results_path

SAMPLE_MOD = 512
TIMEOUT_EXACT = 120
TIMEOUT_SAMPLE = 240


def geng_count(args, timeout):
    cmd = ["nauty-geng", "-u"] + args
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, " ".join(cmd) + " [timeout]"
    m = re.search(r">Z\s+([0-9]+) graphs generated", p.stderr)
    return (int(m.group(1)) if m else None), " ".join(cmd)


def classes():
    for n in (10, 11, 12, 13):
        tot = n * (n - 1) // 2
        e_lo, e_hi = (7 * n + 1) // 2, 5 * n - 15
        ce_lo, ce_hi = tot - e_hi, tot - e_lo
        yield (f"sweep-n{n}", [f"-D{n - 8}", str(n), f"{ce_lo}:{ce_hi}"])
    # The alpha<=2 window closes at n=19: C(n,2)-(5n-15) > n^2/4 from
    # there on, i.e. every alpha<=2 graph on >=19 vertices exceeds the
    # Mader bound and has a K7 minor by density alone. So n=13..18 is
    # the WHOLE open range for the alpha<=2 case of Hadwiger t=7.
    for n in (13, 14, 15, 16, 17, 18):
        tot = n * (n - 1) // 2
        ce_lo = tot - (5 * n - 15)
        ce_hi = (n * n) // 4
        if ce_lo > ce_hi:
            continue
        yield (f"alpha2-n{n}", ["-t", str(n), f"{ce_lo}:{ce_hi}"])


def main():
    for name, args in classes():
        t0 = time.time()
        cnt, cmd = geng_count(args, TIMEOUT_EXACT)
        rec = {"class": name, "cmd": cmd}
        if cnt is not None:
            rec.update(count=cnt, exact=True, wall_s=round(time.time() - t0, 1))
        else:
            t0 = time.time()
            scnt, scmd = geng_count(args + [f"0/{SAMPLE_MOD}"], TIMEOUT_SAMPLE)
            if scnt is not None:
                rec.update(count=scnt * SAMPLE_MOD, exact=False,
                           sample_mod=SAMPLE_MOD, sample_count=scnt,
                           wall_s=round(time.time() - t0, 1))
            else:
                rec.update(count=None, exact=False,
                           note="even 1/512 slice timed out")
        jsonl_append(results_path("classes.jsonl"), rec)
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
