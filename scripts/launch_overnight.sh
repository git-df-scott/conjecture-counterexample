#!/bin/bash
# Overnight Hadwiger t=7 search orchestrator (4 cores).
#
#   HOURS=<wall cap> scripts/launch_overnight.sh
#
# Lanes: the exhaustive chain (priority) runs its shards 4-wide; two
# annealing islands and the structgen sweep run niced in the background
# with their own time boxes. Every lane checks results/hadwiger/STOP and
# is resumable via its JSONL manifest, so a dead session loses nothing.
set -u
cd "$(dirname "$0")/.."
HOURS="${HOURS:-10}"
ANNEAL_HOURS="${ANNEAL_HOURS:-3}"
STRUCT_HOURS="${STRUCT_HOURS:-2}"
DEADLINE=$(( $(date +%s) + $(python3 -c "print(int(${HOURS}*3600))") ))
LOGD=results/hadwiger/logs
mkdir -p "$LOGD"

left_hours() { python3 -c "import time;print(max(0.02,($DEADLINE-time.time())/3600))"; }
stopped() { [ -f results/hadwiger/STOP ]; }

echo "cap=${HOURS}h anneal=${ANNEAL_HOURS}h struct=${STRUCT_HOURS}h"

nice -n 10 python3 -u scripts/run_anneal.py --island 0 --seed-kind 5tree \
  --n0 24 --hours "$ANNEAL_HOURS" > "$LOGD/anneal0.log" 2>&1 &
nice -n 10 python3 -u scripts/run_anneal.py --island 1 --seed-kind projquad \
  --n0 24 --hours "$ANNEAL_HOURS" > "$LOGD/anneal1.log" 2>&1 &
nice -n 10 python3 -u scripts/run_structgen.py --hours "$STRUCT_HOURS" \
  > "$LOGD/structgen.log" 2>&1 &

run_shards() { # name mod script extra-args...
  local name="$1" mod="$2" script="$3"; shift 3
  stopped && return
  echo "=== phase $name (x$mod) $(date -u +%H:%M:%S) ==="
  for res in $(seq 0 $((mod - 1))); do
    python3 -u scripts/"$script" "$@" --res "$res" --mod "$mod" \
      --budget-hours "$(left_hours)" > "$LOGD/${name}_r${res}.log" 2>&1 &
  done
  wait $(jobs -p | tail -n "$mod") 2>/dev/null
}

# Priority chain. Order: cheapest certain ground first, then the open
# frontier by expected value per CPU-hour.
run_shards sweep10  1 run_sweep.py  --n 10
run_shards sweep11  1 run_sweep.py  --n 11
run_shards alpha13  4 run_alpha2.py --n 13
run_shards alpha14  4 run_alpha2.py --n 14
run_shards alpha15  4 run_alpha2.py --n 15
run_shards alpha16  4 run_alpha2.py --n 16
run_shards alpha17  2 run_alpha2.py --n 17
run_shards alpha18  2 run_alpha2.py --n 18
run_shards sweep12  4 run_sweep.py  --n 12
run_shards sweep13  4 run_sweep.py  --n 13

wait
echo "=== all lanes finished $(date -u +%H:%M:%S) ==="
python3 - <<'EOF'
import glob, json
for mf in sorted(glob.glob("results/hadwiger/*/*manifest.jsonl")):
    for line in open(mf):
        rec = json.loads(line)
        print(mf.split("/")[-1], rec.get("shard"), rec.get("status"),
              json.dumps(rec.get("stats", {})))
EOF
