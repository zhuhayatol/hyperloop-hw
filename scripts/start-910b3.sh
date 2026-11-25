#!/usr/bin/env bash
set -eo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && $1 != --check ) ]]; then
    printf 'Usage: bash %s [--check]\n' "$0" >&2
    exit 2
fi
if [[ $(id -u) != 0 ]]; then
    exec sudo -n bash "$(realpath -e -- "$0")" "$@"
fi

base=/opt/hyperloop-910b3-trial
repo="$base/closure-forward"
deployed="$base/closure-inputs"
runner="$deployed/protected-python.sh"
history=/mnt/workspace/r1/real/r/evidence/recipes/final.json
key_file=/home/developer/.config/hyperloop/deepseek.key

if [[ -n $(git -c safe.directory="$repo" -C "$repo" status --porcelain) ]]; then
    printf 'The deployed Hyperloop checkout has local changes: %s\n' "$repo" >&2
    exit 1
fi
commit=$(git -c safe.directory="$repo" -C "$repo" rev-parse HEAD)
if [[ ! -s $key_file || $(stat -c %a -- "$key_file") != 600 ]]; then
    printf 'Expected the existing DeepSeek key file with mode 0600: %s\n' "$key_file" >&2
    exit 1
fi

# Reserve a short absolute path for SGLang's ZMQ sockets.
output=
if [[ ${1:-} != --check ]]; then
    output=$(mktemp -d /opt/h15-XXXXXX)
    chmod 755 "$output"
    printf '%s\n' "$commit" > "$output/hyperloop-commit.txt"
    export TRITON_CACHE_DIR="$output/tool-cache"
fi
printf 'Hyperloop %s\n' "$commit"

bash "$runner" "$repo" - "$deployed" "$history" "$output" <<'PY'
import subprocess
import sys
from pathlib import Path

from hyperloop.actions.knowledge import load_verified_recipe
from hyperloop.agents.structured import source_excerpts
from hyperloop.contracts import json_bytes, read_json, validate_target, validate_task

deployed, history = map(Path, sys.argv[1:3])
output = sys.argv[3]
task = read_json(deployed / "combined-task.json")
target = read_json(deployed / "combined-target.json")
excerpts = read_json(deployed / "excerpts.json")
recipe = load_verified_recipe(history)
chunk = recipe["parameters"]["chunked_prefill_size"]
task["id"] = "910b3-optimization"
task["agent_objective"] = (
    "Use the current hardware Profile to choose between parameter exploration and a bounded framework patch. "
    f"Earlier accepted chunked_prefill_size={chunk} is a historical lead; remeasure it if relevant. "
    "Aim for at least 15% higher throughput and 15% lower TTFT P95 against the original baseline. "
    "Keep outputs exact, respect TPOT limits and avoid duplicate proposals."
)
task["settings"].update(max_cycles=3, max_turns_per_cycle=2, budget=45)
validate_task(task)
validate_target(target)
source_excerpts(excerpts["excerpts"], target)

npu = subprocess.check_output(["npu-smi", "info"], text=True)
if "910B3" not in npu or "No running processes found" not in npu:
    raise SystemExit("The installed 910B3 must be available and idle.")
if not output:
    print(f"Launch inputs available. Prior measured prefill chunk: {chunk}. 910B3 idle.")
    raise SystemExit(0)

root = Path(output)
inputs = root / "inputs"
inputs.mkdir()
for name, record in (
    ("task", task), ("target", target), ("scenario", {}), ("excerpts", excerpts), ("historical-recipe", recipe),
):
    (inputs / f"{name}.json").write_bytes(json_bytes(record))
(root / "npu-before.txt").write_text(npu, encoding="utf-8")
PY

if [[ -z $output ]]; then
    exit 0
fi

exec > >(tee "$output/console.log") 2>&1
inputs="$output/inputs"
printf 'Session output: %s\n' "$output"
bash "$runner" "$repo" -m hyperloop capture-reference \
    --task "$inputs/task.json" --target "$inputs/target.json" --output "$output/reference" \
    | tee "$output/reference.run.json"

(
    set +x
    export DEEPSEEK_API_KEY="$(< "$key_file")"
    exec bash "$runner" "$repo" -m hyperloop run \
        --task "$inputs/task.json" --target "$inputs/target.json" --scenario "$inputs/scenario.json" \
        --reference "$output/reference/reference.json" --reference-receipt "$output/reference/capture.json" \
        --agent-source-excerpts "$inputs/excerpts.json" --agent-provider deepseek \
        --session-root "$output" --session-id h15
) | tee "$output/session.run.json"
printf 'Report: %s/real/h15/report.html\n' "$output"
