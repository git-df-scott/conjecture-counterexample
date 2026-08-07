#!/bin/bash
# Self-healing relauncher (v2): 1 annealer chain + 3 parallel k3m8 sweep workers.
D=/home/user/conjecture-counterexample/chvatal
cd "$D" || exit 1
pgrep -f "anneal_chain_1" >/dev/null || nohup setsid bash -c "
  while true; do
    seed=\$((RANDOM*32768+RANDOM))
    { ./search8 8 5000000000 \$seed 1000000000 ; echo round_exit=\$? ; } >> $D/anneal_chain_1.log 2>&1
    grep -q VIOLATION $D/anneal_chain_1.log && exit 17
  done" --name "anneal_chain_1" >/dev/null 2>&1 &
for W in 0 1 2; do
  pgrep -f "k3m8_sweep_w$W" >/dev/null || nohup setsid bash -c "
    # k3m8_sweep_w$W
    cd $D || exit 9
    i=0
    for S in \$(cat k3m8_units.txt); do
      i=\$((i+1)); [ \$((i % 3)) -ne $W ] && continue
      out=results/n8/k3m8_units/u\$S.json
      [ -s \"\$out\" ] && continue
      ./chv_enum test 8 3 \$S 8 > \"\$out.w$W.tmp\"; rc=\$?
      if [ \$rc -eq 17 ]; then mv \"\$out.w$W.tmp\" \"\$out\"; echo VIOLATION_UNIT \$S >> k3m8_sweep.log; exit 17; fi
      [ \$rc -eq 0 ] && mv \"\$out.w$W.tmp\" \"\$out\" && echo done \$S w$W >> k3m8_sweep.log || { rm -f \"\$out.w$W.tmp\"; echo fail \$S rc=\$rc w$W >> k3m8_sweep.log; }
    done; echo WORKER_${W}_COMPLETE >> k3m8_sweep.log" >/dev/null 2>&1 &
done
sleep 3; echo "alive: annealers=$(pgrep -c search8) sweep_workers=$(pgrep -fc 'k3m8_sweep_w')"
