#!/usr/bin/env bash
set -eo pipefail

if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
    printf 'Usage: %s NEW_OUTPUT_DIRECTORY TASK TARGET SCENARIO {deepseek|replay} [SOURCE_EXCERPTS]\n' "$0" >&2
    exit 2
fi

output=$1
task=$2
target=$3
scenario=$4
provider=$5
excerpts=${6:-}
if [ "$provider" != deepseek ] && [ "$provider" != replay ]; then
    printf 'Unknown Agent provider: %s\n' "$provider" >&2
    exit 2
fi

repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo"
if [ -n "$(git status --porcelain)" ]; then
    printf 'Use a clean Hyperloop checkout so the evidence identifies the executed code.\n' >&2
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

output=$(python -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "$output")
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
npu-smi info > "$output/npu-before.txt"
printf 'Hyperloop %s\nEvidence %s\nAgent provider %s\n' "$(cat "$output/hyperloop-commit.txt")" "$output" "$provider"

python -m hyperloop capture-reference --task "$task" --target "$target" --output "$output/reference" \
    | tee "$output/reference.run.json"

agent_args=(--agent-provider "$provider")
if [ -n "$excerpts" ]; then
    agent_args+=(--agent-source-excerpts "$excerpts")
fi
(
    set +x
    if [ "$provider" = deepseek ] && [ -z "${DEEPSEEK_API_KEY:-}" ]; then
        key_file="$HOME/.config/hyperloop/deepseek.key"
        if [ ! -s "$key_file" ] || [ "$(stat -c '%a' -- "$key_file")" != 600 ] || [ ! -O "$key_file" ]; then
            printf 'Set DEEPSEEK_API_KEY or create the owned 0600 file %s.\n' "$key_file" >&2
            exit 1
        fi
        DEEPSEEK_API_KEY=$(cat -- "$key_file")
        export DEEPSEEK_API_KEY
    fi
    python -m hyperloop run --task "$task" --target "$target" --scenario "$scenario" \
        --reference "$output/reference/reference.json" --reference-receipt "$output/reference/capture.json" \
        --session-root "$output" --session-id 910b3-tp1-live "${agent_args[@]}"
) | tee "$output/session.run.json"
