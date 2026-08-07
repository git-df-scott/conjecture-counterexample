#!/bin/bash
# ==============================================================================
#  Counterexample-hunting cloud environment -- session setup script
#
#  Paste into: Update cloud environment -> Setup script.
#  Runs once per session, before Claude Code launches.
#
#  Does two things:
#    1. Installs the search + exact-certification toolchain (the container
#       starts bare -- no numpy, no PARI/GP, no nauty).
#    2. Installs the standing brief at ~/.claude/CLAUDE.md so every session
#       in this environment defaults to hunting counterexamples, under
#       verification rules that make a claimed hit worth believing.
#
#  Idempotent and fully best-effort: every step is timeboxed and non-fatal,
#  so a slow mirror can never block a session from starting.
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
# 1. The brief. Written first, so a slow or failed install never costs us it.
# ------------------------------------------------------------------------------
say "installing standing brief -> ~/.claude/CLAUDE.md"
mkdir -p "$HOME/.claude" "$HOME/math"

cat > "$HOME/.claude/CLAUDE.md" <<'BRIEF_EOF'
# Standing brief: counterexample hunting

This environment exists to find genuine counterexamples to open mathematical
conjectures. When a session arrives with no other task, that is the task.
When the user asks for something else, do that instead -- this is the default
objective, not a restriction on what you are allowed to work on.

## What counts as a result

Exactly two things are results:

1. **A counterexample certificate.** An explicit object, plus an exact
   verification that it violates the conjecture, reproducible from a script in
   the repo by someone who trusts none of your intermediate output.
2. **A certified negative.** What was exhausted, by what method, at what cost,
   with the best value found stated against the conjectured bound -- and an
   honest statement of what the search could have missed.

A promising direction is not a result. A near-miss is not a result. A float
that came out slightly under the bound is not a result.

## Verification rules -- non-negotiable

These exist because the failure mode of this work is not "found nothing", it
is "announced something false". Every rule below is a defence against that.

1. **Floating point never certifies.** A float hit is a hint. Confirm in exact
   arithmetic (`fractions.Fraction`, `gmpy2`, `python-flint`, `sympy`) or in
   validated interval arithmetic (`mpmath.iv`). Ill-conditioned inputs drift
   in whichever direction happens to be convenient; assume yours did.
2. **Two independent paths must agree.** The second path must not consume the
   first's intermediate output as trusted input -- re-derive from the raw
   object. Agreement of two runs of the same code is not agreement.
3. **Know your error direction.** Show that residual error in your pipeline can
   only push *away* from a false positive. If you cannot state which way the
   error goes, you do not have a certificate.
4. **Reproduce cold.** Script + seed + explicit inputs, from a clean checkout,
   must reconstruct the object. Log the git commit and the seed in every record.
5. **Re-read the conjecture before believing a hit.** Most "counterexamples"
   are hypothesis violations: wrong normalisation, an off-by-one in indexing, a
   degenerate or boundary case the statement never claimed to cover, the wrong
   variant of a conjecture that has several. Check the exact hypotheses against
   a primary source, not memory.
6. **Check status and known bounds before spending compute.** Conjectures get
   proven and disproven, and verified-to bounds move. Web-search the current
   status and the current search frontier first. Searching a range someone
   exhausted in 2019 is the most common way to waste a session.
7. **A trigger fires -> stop and freeze.** Do not keep searching past a
   possible hit. Snapshot the state, commit it, then certify.

## Reporting rules

- **A null result is the expected outcome and is a real deliverable.** Report
  it plainly with coverage numbers. Do not dress it up, and do not apologise
  for it.
- **Never claim to have proven the conjecture.** A finite search cannot settle
  a universally quantified statement unless the reduction to a finite check is
  itself proven and cited -- and then cite it.
- **Never round a near-miss into a hit.** Report the exact value and the exact
  gap to the bound.
- **State the residual risk in your own pipeline** in the write-up. You wrote
  the certifier; say where it could still be wrong.
- If a run did not happen, say it did not happen. Never report numbers you did
  not produce.

## Search engineering defaults

- **Two tiers.** A cheap filter that may be wrong, then an exact certifier that
  may not. Never certify everything; never trust the filter.
- **Normalise by the problem's invariance group** before evaluating or
  comparing -- it kills duplicate work and most of the conditioning problems.
- **Resumable by construction.** One JSONL record per unit of work, a
  `summary.json`, a wall-clock cap, partial results committed as you go. This
  container is ephemeral: uncommitted work is lost work.
- **Measure one unit before launching the sweep.** Extrapolate the cost, state
  it, then decide. 4 cores here -- budget accordingly.
- **Prefer canonical-form generation to generate-and-filter** for combinatorial
  families. Count with `geng -u` before you generate.
- **Parallelism is by process** (OMP_NUM_THREADS is pinned to 1 for this
  reason). Use `parallel` or a process pool, not threaded BLAS.

## Choosing a target

Prefer conjectures where a counterexample is a single finite object;
verification is cheap relative to search; the space admits a canonical form or
a strong invariant; and the smallest untested case is actually within reach of
4 cores. Curated candidates, with the tractability notes that matter, are in
`~/math/TARGETS.md` -- treat every status line there as needing confirmation.

## Toolchain available

- **Exact / arbitrary precision:** `sympy`, `gmpy2`, `python-flint`, `mpmath`
  (`mpmath.iv` for validated intervals, `mpmath.pslq` for integer relations)
- **Numeric search:** `numpy`, `scipy`, `cma`
- **Graphs:** `nauty` (`geng`, `genbg`, `directg`, `shortg`, `nauty-*`),
  `networkx`
- **Constraint / SMT search:** `z3-solver`, `python-sat` (cadical, glucose),
  `ortools` CP-SAT -- the right first tool for "find an object satisfying these
  constraints"
- **Number theory:** PARI/GP (`gp`), with `qflll`, `nfroots`, `ellrank`, etc.
- **Native kernels:** `gcc`/`g++` with GMP, MPFR, NTL headers -- for brute-force
  inner loops, hand-written C beats anything in Python by two orders of
  magnitude, and the search tier does not need to be certified.
BRIEF_EOF
ok "$HOME/.claude/CLAUDE.md"

# ------------------------------------------------------------------------------
# 2. Curated target list.
# ------------------------------------------------------------------------------
cat > "$HOME/math/TARGETS.md" <<'TARGETS_EOF'
# Candidate targets

Ranked by search tractability, not by fame. Every status and bound below is a
starting point that MUST be re-confirmed against current literature before you
spend compute -- these move, and some of them move without fanfare.

## Tier 1 -- counterexample is one finite object, verification is cheap

- **Lander-Parkin-Selfridge**: no solution to a sum of n k-th powers = sum of m
  k-th powers with n+m < k. Pure integer search, exact by construction, trivially
  parallel, no floating point anywhere. The best value-per-core-hour here.
- **Barnette's conjecture**: every 3-connected cubic bipartite planar graph is
  Hamiltonian. Counterexample is one graph; Hamiltonicity check is cheap.
  Generation is the bottleneck -- needs planar generation (`plantri`, not in
  apt; build from source) rather than `geng` filtering.
- **Seymour's second neighbourhood conjecture**: every oriented simple digraph
  has a vertex with |N++| >= |N+|. Per-digraph check is trivial; `directg` over
  `geng` output generates the space.
- **Graceful tree conjecture**: every tree has a graceful labelling. One tree
  falsifies it. Trees are cheap to generate canonically (`gentreeg`); the
  labelling search per tree is the cost -- CP-SAT handles it well.
- **Union-closed sets (Frankl)**: some element in >= half the sets. Finite check
  per family; canonical generation is the whole game.

## Tier 2 -- reachable, but the frontier is already far out

- **Erdos-Straus**: 4/n = 1/a + 1/b + 1/c for n > 1. Checked past 10^17, so any
  new range needs a genuinely fast sieve in C, not Python.
- **Perfect (Euler) cuboid**: all edges, face diagonals, and space diagonal
  integral. Long-searched; a hit resolves the question positively. Good C+GMP
  target with strong divisibility constraints to prune on.
- **Firoozbakht**: p_n^(1/n) strictly decreasing. Widely expected to be *false*
  eventually, which is exactly the property you want in a target -- but the
  expected first failure is far past any reachable range. Read the heuristics
  before committing.
- **Cycle double cover / Tutte 5-flow**: a counterexample must be a snark.
  Snark generation is the constraint; the per-graph check is fine.
- **Hadwiger's conjecture** for k >= 7: finite counterexample, but the graphs
  are large and minor-testing is expensive. Only with a sharp invariant to prune.

## Tier 3 -- structurally interesting, search space is not reachable

Odd perfect numbers (nothing below 10^1500), Giuga, Lehmer's totient problem,
Collatz (verified past 2^68), Goldbach (past 4x10^18), Riemann. Worth attacking
only via *structural constraints* -- narrowing the form a counterexample must
take -- never by extending the brute-force frontier.

## Known-settled -- do not "search" these

Sensitivity (proven 2019), Erdos discrepancy (2015), Hirsch (disproven 2010 --
but *polynomial* Hirsch is open), Keller (dim 7 settled 2020), Ringel (2020),
Borsuk (disproven 1993), Polya, Mertens, Euler's sum of powers. Kakeya in R^3
and the moving sofa both moved recently. Confirm before assuming anything on
this list, in either direction.

## In this repo

Mahler in dim 4: searched, no counterexample, floor certified at 32/3 on all
best bodies. Extending it means a genuinely different body class or a sharper
certifier -- not more random restarts.
TARGETS_EOF
ok "$HOME/math/TARGETS.md"

# ------------------------------------------------------------------------------
# 3. Shell + Claude Code environment.
#    OMP_NUM_THREADS=1 on purpose: these searches parallelise by process, and
#    threaded BLAS underneath a process pool just oversubscribes 4 cores.
#    PYTHONHASHSEED=0 so set/dict iteration order is reproducible run to run.
# ------------------------------------------------------------------------------
say "pinning search environment"
if ! grep -q 'counterexample-env' "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<'RC_EOF'

# --- counterexample-env ---
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export GP_DATA_DIR=/usr/share/pari
# --- end counterexample-env ---
RC_EOF
fi
python3 - <<'PY' >>"$LOG" 2>&1 && ok "shell + settings.json env" || bad "settings.json env merge"
import json, os, pathlib
p = pathlib.Path.home() / ".claude" / "settings.json"
try:
    cfg = json.loads(p.read_text()) if p.exists() and p.read_text().strip() else {}
except Exception:
    cfg = {}
env = cfg.setdefault("env", {})
for k, v in {
    "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg",
}.items():
    env.setdefault(k, v)          # never clobber something already set
p.write_text(json.dumps(cfg, indent=2) + "\n")
PY

# ------------------------------------------------------------------------------
# 4. System packages.
# ------------------------------------------------------------------------------
say "installing system packages (pari/gp, nauty, gmp/mpfr/ntl headers)"
export DEBIAN_FRONTEND=noninteractive
if timeout 240 $SUDO apt-get update -qq >>"$LOG" 2>&1; then
  APT="pari-gp nauty parallel bc libgmp-dev libmpfr-dev libmpc-dev libntl-dev"
  if timeout 420 $SUDO apt-get install -y -qq --no-install-recommends $APT >>"$LOG" 2>&1; then
    ok "$APT"
  else
    for p in $APT; do
      timeout 180 $SUDO apt-get install -y -qq --no-install-recommends "$p" >>"$LOG" 2>&1 \
        && ok "$p" || bad "$p"
    done
  fi
else
  bad "apt-get update"
fi

# Debian ships every nauty tool as nauty-geng, nauty-gentreeg, ... but the
# literature, the nauty manual and everyone's muscle memory say geng. Expose
# both, without shadowing anything that already exists on PATH.
if compgen -G "/usr/bin/nauty-*" >/dev/null; then
  mkdir -p /usr/local/bin
  n=0
  for f in /usr/bin/nauty-*; do
    base="${f##*/nauty-}"
    command -v "$base" >/dev/null 2>&1 && continue
    $SUDO ln -sf "$f" "/usr/local/bin/$base" 2>>"$LOG" && n=$((n+1))
  done
  ok "nauty tools aliased unprefixed ($n: geng, gentreeg, directg, cubhamg, ...)"
fi

# ------------------------------------------------------------------------------
# 5. Python packages, in groups so one bad wheel cannot take out the rest.
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
pip_group "exact arithmetic: sympy mpmath gmpy2 python-flint" sympy mpmath gmpy2 python-flint
pip_group "numeric search: numpy scipy cma"                   numpy scipy cma
pip_group "combinatorics: networkx tqdm joblib pytest"        networkx tqdm joblib pytest matplotlib
pip_group "constraint search: z3-solver python-sat ortools"   z3-solver python-sat ortools

# ------------------------------------------------------------------------------
# 6. Self-check. Prints what actually landed, not what we tried to install.
# ------------------------------------------------------------------------------
say "verifying"
python3 - <<'PY'
import importlib
mods = ["sympy","mpmath","gmpy2","flint","numpy","scipy","cma",
        "networkx","z3","pysat","ortools","pytest","matplotlib"]
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
for t in gp geng gentreeg directg shortg parallel gcc; do
  command -v "$t" >/dev/null 2>&1 && printf '   ok   %s -> %s\n' "$t" "$(command -v "$t")" || bad "$t"
done

say "ready. Brief: ~/.claude/CLAUDE.md   Targets: ~/math/TARGETS.md   Log: $LOG"
exit 0
