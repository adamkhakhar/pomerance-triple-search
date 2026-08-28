#!/usr/bin/env bash
# run_frontier.sh -- build, prove, measure, then search, on a RunPod GPU box.
#
#   ./run_frontier.sh                  # full sequence on the frontier target
#   ./run_frontier.sh --ab-only        # build + selftest + A/B, no production
#   P=<prime> ./run_frontier.sh        # a different target
#
# Sequence, in order, because each step gates the next:
#   1. build for the detected GPU architecture
#   2. on-device selftest (transport identity, order 32, points on E0)
#   3. same-seed A/B: legacy record-stack kernel vs cover kernel
#   4. production search, one process per GPU on disjoint seed ranges
#   5. verify any triple found with the official DANGER3 verifier
set -uo pipefail
cd "$(dirname "$0")"

P="${P:-1000000000000000000000000103}"          # 10^27 + 103
BUDGET="${BUDGET:-2400000000000}"               # cover curves, ~3x expectation
CHUNK="${CHUNK:-2000000000}"
AB_TRIALS="${AB_TRIALS:-40000000}"
SEED_BASE="${SEED_BASE:-1}"

# ---- 1. build -------------------------------------------------------------
if [ -z "${ARCH:-}" ]; then
  CC_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
  ARCH="sm_${CC_RAW:-89}"
fi
echo "== build (ARCH=$ARCH, LANES=${LANES:-4}) =="
make -s clean
make -s search_cuda ARCH="$ARCH" LANES="${LANES:-4}" || { echo "build failed"; exit 1; }
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader

# ---- 2. correctness gate --------------------------------------------------
echo
echo "== on-device selftest =="
POM_SELFTEST=only ./search_cuda "$P" 1 1000 x16halvenonsplit || {
  echo "SELFTEST FAILED - stopping"; exit 3; }

# ---- 3. same-seed A/B ------------------------------------------------------
echo
echo "== A/B: legacy record-stack kernel vs cover kernel =="
echo "-- legacy ($AB_TRIALS marks) --"
t0=$(date +%s.%N)
POM_SOURCE=legacy POM_SELFTEST=0 ./search_cuda "$P" 7 "$AB_TRIALS" x16halvenonsplit "$CHUNK" \
  2>&1 | tail -3
t1=$(date +%s.%N)
echo "-- cover ($((AB_TRIALS / 4)) cover curves = same hazard) --"
t2=$(date +%s.%N)
POM_SELFTEST=0 ./search_cuda "$P" 7 "$((AB_TRIALS / 4))" x16halvenonsplit "$CHUNK" \
  2>&1 | tail -3
t3=$(date +%s.%N)
awk -v a="$t0" -v b="$t1" -v c="$t2" -v d="$t3" 'BEGIN{
  L=b-a; C=d-c;
  printf "\nequal-hazard A/B: legacy %.1fs, cover %.1fs -> %.2fx\n", L, C, L/C;
  printf "(1 cover curve = 2 distinct curves = 4 legacy marks of hazard)\n"}'

[ "${1:-}" = "--ab-only" ] && { echo "A/B only; stopping before production."; exit 0; }

# ---- 4. production --------------------------------------------------------
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
PER=$(( BUDGET / NGPU ))
echo
echo "== production: $NGPU GPU(s), $PER cover curves each, p = $P =="
mkdir -p out
pids=()
for g in $(seq 0 $((NGPU - 1))); do
  seed=$(( SEED_BASE + g * 1000003 ))
  CUDA_VISIBLE_DEVICES=$g POM_SELFTEST=0 \
    ./search_cuda "$P" "$seed" "$PER" x16halvenonsplit "$CHUNK" \
    > "out/gpu${g}.log" 2>&1 &
  pids+=($!)
  echo "  gpu $g: seed $seed -> out/gpu${g}.log"
done

# Stop every worker as soon as one reports a hit.
while :; do
  if grep -lq "Verified: PASS" out/gpu*.log 2>/dev/null; then
    echo; echo "== HIT =="; grep -h -B3 "Verified: PASS" out/gpu*.log | head -20
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null; done
    break
  fi
  alive=0
  for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
  [ "$alive" = 0 ] && { echo; echo "all workers finished with no hit in budget"; break; }
  sleep 30
done
wait 2>/dev/null

# ---- 5. verify -------------------------------------------------------------
line=$(grep -h -A2 "^p = \|Verified: PASS" out/gpu*.log 2>/dev/null | \
       grep -oE "^[0-9]+ [0-9]+ [0-9]+$" | head -1)
if [ -n "$line" ]; then
  set -- $line
  echo
  echo "== official DANGER3 verification =="
  echo "p=$1 A=$2 x0=$3"
  python3 ../tools/vpp.py "$1" "$2" "$3"
fi
