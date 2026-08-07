"""Shared runner plumbing: JSONL persistence, shard manifests, stop flag."""

from __future__ import annotations

import json
import os
import time

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results", "hadwiger")


def results_path(*parts):
    p = os.path.join(RESULTS_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def jsonl_append(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def jsonl_load(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # torn tail write from an interrupted session
    return out


def done_shards(manifest_path):
    return {tuple(rec["shard"]) for rec in jsonl_load(manifest_path)
            if rec.get("status") == "done"}


STOP_FLAG = os.path.join(RESULTS_DIR, "STOP")


def stop_requested():
    return os.path.exists(STOP_FLAG)


def raise_stop(reason: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(STOP_FLAG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {reason}\n")


def record_candidate(kind, n, g6, extra=None):
    """A potential counterexample: log it and stop every lane."""
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "kind": kind, "n": n, "g6": g6}
    if extra:
        rec.update(extra)
    jsonl_append(results_path("candidates.jsonl"), rec)
    raise_stop(f"candidate from {kind}: {g6}")


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.wall = time.perf_counter() - self.t0
