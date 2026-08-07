#!/bin/bash
# Self-healing relauncher: run after any container restart. Idempotent.
D=/home/user/conjecture-counterexample/chvatal
cd "$D" || exit 1
# 1) Annealer rounds: 3 chains, each an endless loop of 5e9-step rounds (fresh seed per round).
for c in 1 2 3; do
  pgrep -f "anneal_chain_$c" >/dev/null || nohup setsid bash -c "
    while true; do
      seed=\$((RANDOM*32768+RANDOM))
      { ./search8 8 5000000000 \$seed 1000000000 ; echo round_exit=\$? ; } >> $D/anneal_chain_$c.log 2>&1
      grep -q VIOLATION $D/anneal_chain_$c.log && exit 17
    done" --name "anneal_chain_$c" >/dev/null 2>&1 &
done
# 2) k=3 maxm=8 unit sweep (one core, sequential, skip completed units).
pgrep -f "k3m8_sweep" >/dev/null || nohup setsid bash -c "
  # k3m8_sweep
  for S in \$(cat $D/k3m8_units.txt); do
    out=$D/results/n8/k3m8_units/u\$S.json
    [ -s \"\$out\" ] && continue
    ./chv_enum test 8 3 \$S 8 > \"\$out.tmp\" 2>/dev/null; rc=\$?
    if [ \$rc -eq 17 ]; then mv \"\$out.tmp\" \"\$out\"; echo VIOLATION_UNIT \$S >> $D/k3m8_sweep.log; exit 17; fi
    [ \$rc -eq 0 ] && mv \"\$out.tmp\" \"\$out\" && echo done \$S >> $D/k3m8_sweep.log || { rm -f \"\$out.tmp\"; echo fail \$S rc=\$rc >> $D/k3m8_sweep.log; }
  done; echo SWEEP_COMPLETE >> $D/k3m8_sweep.log" >/dev/null 2>&1 &
sleep 3; echo "alive: $(pgrep -c search8) annealers(procs), sweep: $(pgrep -fc k3m8_sweep 2>/dev/null || echo 0)"
