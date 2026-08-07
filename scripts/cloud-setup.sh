#!/bin/bash
# ==============================================================================
#  Cloud environment setup -- math toolchain for counterexample searches.
#
#  Paste into: Update cloud environment -> Setup script.
#  Runs once per session, before Claude Code launches.
#
#  Package installs only. The container starts bare: no numpy, no PARI/GP,
#  no nauty. Every step is timeboxed and non-fatal so a slow mirror can never
#  block a session from starting, and the whole thing is idempotent.
#
#  Full install output: /tmp/setup-math-env.log
# ==============================================================================
set -u

LOG=/tmp/setup-math-env.log
: > "$LOG"
say() { printf '\n[setup] %s\n' "$*"; }
ok()  { printf '   ok   %s\n' "$*"; }
bad() { printf '   --   %s  (skipped -- see %s)\n' "$*" "$LOG"; }

SUDO=""
[ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo -n"

# ------------------------------------------------------------------------------
# System packages
# ------------------------------------------------------------------------------
say "installing system packages"
export DEBIAN_FRONTEND=noninteractive

APT_GROUPS=(
  # exact/bignum headers, for native inner loops
  "libgmp-dev libmpfr-dev libmpc-dev libntl-dev libflint-dev"
  # number theory: PARI/GP, and a very fast prime sieve (pi(10^9) in ~0.03s)
  "pari-gp libpari-dev primesieve libprimesieve-dev"
  # graphs: nauty's generators + canonical forms
  "nauty"
  # exact polyhedral computation -- lrs/redund, Normaliz, cddlib headers
  "lrslib normaliz libcdd-dev"
  # finite group theory, incl. the SmallGrp / transitive / primitive libraries
  "gap gap-libs gap-smallgrp gap-transgrp gap-primgrp"
  # misc
  "parallel bc"
)

if timeout 240 $SUDO apt-get update -qq >>"$LOG" 2>&1; then
  for grp in "${APT_GROUPS[@]}"; do
    if timeout 420 $SUDO apt-get install -y -qq --no-install-recommends $grp >>"$LOG" 2>&1; then
      ok "$grp"
    else
      # one bad package shouldn't take the rest of its group down with it
      for p in $grp; do
        timeout 180 $SUDO apt-get install -y -qq --no-install-recommends "$p" >>"$LOG" 2>&1 \
          && ok "$p" || bad "$p"
      done
    fi
  done
else
  bad "apt-get update"
fi

# Debian ships every nauty tool as nauty-geng, nauty-gentreeg, ... while the
# literature and the nauty manual say geng. Expose both, without shadowing
# anything already on PATH.
if compgen -G "/usr/bin/nauty-*" >/dev/null; then
  mkdir -p /usr/local/bin
  n=0
  for f in /usr/bin/nauty-*; do
    base="${f##*/nauty-}"
    command -v "$base" >/dev/null 2>&1 && continue
    $SUDO ln -sf "$f" "/usr/local/bin/$base" 2>>"$LOG" && n=$((n+1))
  done
  if [ "$n" -gt 0 ]; then
    ok "nauty tools aliased unprefixed ($n: geng, gentreeg, directg, cubhamg, ...)"
  else
    ok "nauty tools already on PATH unprefixed"
  fi
fi

# ------------------------------------------------------------------------------
# Python packages, in groups so one bad wheel cannot take out the rest
# ------------------------------------------------------------------------------
PIP="--no-input --disable-pip-version-check --prefer-binary -q"
if python3 -c 'import sysconfig,os,sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_path("stdlib"),"EXTERNALLY-MANAGED")) else 1)' 2>/dev/null; then
  PIP="$PIP --break-system-packages"      # future images may mark the interpreter managed
fi

pip_group() {
  local label="$1"; shift
  if timeout 600 pip3 install $PIP "$@" >>"$LOG" 2>&1; then
    ok "$label"
  else
    for p in "$@"; do
      timeout 300 pip3 install $PIP "$p" >>"$LOG" 2>&1 && ok "$p" || bad "$p"
    done
  fi
}

say "installing python packages"
pip_group "exact arithmetic"   sympy mpmath gmpy2 python-flint
pip_group "numeric search"     numpy scipy cma
pip_group "graphs"             networkx igraph
pip_group "constraint search"  z3-solver python-sat ortools
pip_group "polyhedra"          pycddlib
pip_group "misc"               galois tqdm joblib pytest matplotlib

# ------------------------------------------------------------------------------
# Thread pinning. Not a package, but these searches parallelise by PROCESS, and
# threaded BLAS under a process pool just oversubscribes the 4 available cores.
# PYTHONHASHSEED=0 keeps set/dict iteration order reproducible run to run.
# Delete this block if you'd rather set them yourself.
# ------------------------------------------------------------------------------
if ! grep -q 'math-env-threads' "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<'RC_EOF'

# --- math-env-threads ---
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
# --- end math-env-threads ---
RC_EOF
  ok "thread pinning -> ~/.bashrc"
fi

# ------------------------------------------------------------------------------
# Self-check: report what actually landed, not what we tried to install
# ------------------------------------------------------------------------------
say "verifying"
python3 - <<'PY'
import importlib
mods = ["sympy","mpmath","gmpy2","flint","numpy","scipy","cma","networkx",
        "igraph","z3","pysat","ortools","cdd","galois","pytest","matplotlib"]
have, miss = [], []
for m in mods:
    try:
        mod = importlib.import_module(m)
        have.append(f"{m} {getattr(mod,'__version__','')}".strip())
    except Exception:
        miss.append(m)
print("   python:", ", ".join(have) or "none")
if miss:
    print("   MISSING:", ", ".join(miss))
PY
BINS="gp geng gentreeg directg shortg cubhamg lrs normaliz primesieve gap parallel gcc"
found=""; missing=""
for t in $BINS; do
  command -v "$t" >/dev/null 2>&1 && found="$found $t" || missing="$missing $t"
done
printf '   binaries:%s\n' "$found"
[ -n "$missing" ] && printf '   MISSING:%s\n' "$missing"

say "ready. Log: $LOG"
exit 0
