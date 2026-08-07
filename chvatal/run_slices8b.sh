#!/bin/bash
# n=8 bounded-m exhaustive slice ladder; stops instantly on any violation.
cd "$(dirname "$0")"
for spec in "5 7" "4 7" "3 7" "2 7" "5 8" "2 8"; do
  set -- $spec; k=$1; mm=$2
  out=results/n8/k${k}_m${mm}.json
  [ -s "$out" ] && continue
  timeout 5400 ./chv_enum test 8 "$k" -1 "$mm" > "$out.tmp"
  rc=$?
  if [ $rc -eq 17 ]; then mv "$out.tmp" "$out"; echo "VIOLATION_IN k=$k maxm=$mm"; exit 17; fi
  if [ $rc -ne 0 ]; then echo "SLICE_SKIP k=$k maxm=$mm rc=$rc"; rm -f "$out.tmp"; continue; fi
  mv "$out.tmp" "$out"; echo "SLICE_DONE k=$k maxm=$mm $(cat "$out" | head -c 160)"
done
echo ALL_SLICES_DONE
