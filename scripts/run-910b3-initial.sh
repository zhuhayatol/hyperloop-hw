#!/usr/bin/env bash
set -eo pipefail

if [ "$#" -ne 1 ]; then
    printf 'Usage: bash scripts/run-910b3-initial.sh NEW_OUTPUT_DIRECTORY\n' >&2
    exit 2
fi

repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo"
if [ -n "$(git status --porcelain)" ]; then
    printf 'The Hyperloop checkout has local changes. Use a clean committed checkout for this run.\n' >&2
    exit 1
fi

platform_root=/mnt/workspace/sglang-install
venv=/mnt/workspace/src/.venvs/sglang-v0.5.16
source "$venv/bin/activate"
source "$HOME/Ascend/nnal/atb/set_env.sh" --cxx_abi=1
source "$platform_root/opp/vendors/customize/bin/set_env.bash"
source "$platform_root/opp/vendors/custom_transformer/bin/set_env.bash"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="$platform_root/triton-cache" OMP_NUM_THREADS=4
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"

output=$(python -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1")
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"

finish() {
    local status=$?
    npu-smi info > "$output/npu-after.txt" 2>&1 || true
    printf '%s\n' "$status" > "$output/exit-code.txt"
}
trap finish EXIT
exec > >(tee "$output/deployment.console.log") 2>&1

date -u +%FT%TZ > "$output/start-utc.txt"
git rev-parse HEAD > "$output/hyperloop-commit.txt"
git status --short --branch > "$output/git-status.txt"
npu-smi info > "$output/npu-before.txt"
printf 'Hyperloop %s\nEvidence %s\n' "$(cat "$output/hyperloop-commit.txt")" "$output"

task=configs/task.910b3.initial.json
target=configs/target.910b3.sglang.json
scenario=fixtures/scenario.910b3.initial.json
session=910b3-tp1-initial

printf 'Capturing reference from the original SGLang service.\n'
python -m hyperloop capture-reference --task "$task" --target "$target" --output "$output/reference" \
    | tee "$output/reference.run.json"

printf 'Starting independent baseline and R01 evaluation.\n'
python -m hyperloop run --task "$task" --target "$target" --scenario "$scenario" \
    --reference "$output/reference/reference.json" --reference-receipt "$output/reference/capture.json" \
    --session-root "$output" --session-id "$session" | tee "$output/session.run.json"

python - "$output/real/$session/summary.json" <<'PY'
import json
import sys
from pathlib import Path

report = Path(sys.argv[1])
summary = json.loads(report.read_text(encoding="utf-8"))
baseline = summary.get("baseline") or {}
decisions = [row for row in summary["decisions"] if row["candidate_id"] == "R01"]
valid = (
    summary["mode"] == "real"
    and summary["phase"] == "DONE"
    and (baseline.get("measurement") or {}).get("mode") == "real"
    and len(decisions) == 1
    and decisions[0]["version_action"] in {"KEEP", "REVERT"}
    and decisions[0]["evidence_state"] == "sufficient"
)
if not valid:
    raise SystemExit(f"Initial real run did not reach an evidence-backed decision. See {report}")
print(f"R01 {decisions[0]['version_action']} ({decisions[0]['reason']})")
print(report.with_name("report.html"))
PY
