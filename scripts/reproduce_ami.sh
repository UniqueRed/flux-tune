#!/bin/sh
# Reproduce the AMI meeting benchmark end to end. No Deepgram key needed: every Flux pass is
# already in data/flux_cache, keyed by a hash of the exact audio, and the audio rebuilds
# byte-identically from the public AMI corpus.
#   sh scripts/reproduce_ami.sh            # cut-off / latency results (~120 MB download)
#   sh scripts/reproduce_ami.sh --noise    # plus the background-talker test (~+270 MB)
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
POOL="--pre 1.0 --post 1.6 --out data/pool/ami_v2 --seed 1337"
$PY scripts/fetch_ami_segments.py --split val --n 900 $POOL
$PY scripts/fetch_ami_segments.py --split test --n 900 $POOL
NOISE=""
if [ "$1" = "--noise" ]; then
  $PY scripts/fetch_ami_segments.py --split train --n 3000 $POOL
  NOISE="--noise"
fi
$PY scripts/make_ami_bench.py --out data/ami_bench
B=data/ami_bench
$PY scripts/flux_tune.py $B --agent_times $B/agent_times.json --split_map $B/split.json \
  --groups $B/groups.json --target 0.1 $NOISE --yes --out reports/ami_bench_repro
$PY scripts/build_report.py reports/ami_bench_repro --title "flux-tune AMI Benchmark" \
  --dataset "AMI meetings: 28 dev speakers to tune, 43 unseen test speakers to report"
echo "Compare with the published run: reports/ami_bench/results.json"
