#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo 'usage: bash run.sh full|noverify|nomemory|noabort SCENE.json MODEL EPISODE' >&2; exit 2
fi
ablation="$1" scene="$2" model="$3" episode="$4"
[[ "$episode" =~ ^[1-9][0-9]*$ ]] || { echo 'EPISODE must be a positive integer' >&2; exit 2; }
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python="${PYTHON:-python}"
name="$ablation-$(basename "$scene" .json)-episode$(printf %02d "$episode")-$(date +%m%d-%H%M%S)"
args=(reset.py --ablation "$ablation" --scene "$scene" --model "$model" --run-name "$name"
    --gpt-model "${GPT_MODEL:-gpt-6.1-sol}" --effort "${GPT_EFFORT:-medium}"
    --region-seed "${REGION_SEED:-0}" --seed 0 --seconds 60 --bon 5 --noise-scale 0.5
    --cap-per-object 3 --max-attempts 9
    --holding-n-min 5 --holding-w-min 0.12 --holding-delta 0.15 --holding-m 3 --holding-eps 0.001)
if [ -n "${HOME_REFERENCE:-}" ]; then args+=(--home-reference "$HOME_REFERENCE"); fi
if [ -n "${CALIB:-}" ]; then args+=(--calib "$CALIB"); fi
export SIS_POLICY_SERVER="${SIS_POLICY_SERVER:-tcp://127.0.0.1:8770}"
export SIS_EARLY_STOP=1 SIS_ACTION_SMOOTH=3 SIS_BON_ENGINE=async_shared SIS_BON_EXEC_STEPS=auto SIS_BON_BLEND_STEPS=5
exec "$python" -B "${args[@]}"
