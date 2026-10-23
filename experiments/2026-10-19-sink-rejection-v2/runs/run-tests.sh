#!/usr/bin/env bash
# Unit tests for sink rejection v2, in throwaway containers from the promoted r6x
# image on dgx4, with the candidate's changed files mounted over the image's
# sources (the image's vLLM tree is the candidate's base, r6x-candidate
# f3b58ed9d, tree 4d6851c8). No network; dgx4's own GPU; no cluster node touched.
#
#   run-tests.sh sync [vllm-worktree]   (on the Mac) copies the files that differ from
#                                       f3b58ed9d into dgx4:$LAB/candidate
#   run-tests.sh cpu                    the CPU suites
#   run-tests.sh gpu                    the suites that need the GPU, and the existing
#                                       scheduler suites (their DeviceConfig infers the
#                                       device; the CPU container has no platform)
#   run-tests.sh baseline               the existing suites on the untouched image, for
#                                       the offline-failure comparison
#
# --confcutdir=tests/v1 skips the root tests/conftest.py (tblib is not in the image).
# The scheduler suites build a ModelConfig for facebook/opt-125m (and a few for
# llava-1.5-7b, not cached: those fail offline on the untouched image too); its
# config and tokenizer files, fetched once for 0221, are mounted read-only.
set -u
LAB=${LAB:-$HOME/spark-lab/sink-rejection-v2}
HF_CACHE=${HF_CACHE:-$HOME/spark-lab/sink-rejection/hf-cache}
V=/opt/spark3/candidate/vllm
IMG=vllm-ds41f-kkref:19f2c20ed4d6-r6x
BASE=f3b58ed9d

mounts() {
  local f
  for f in $(cd "$LAB/candidate" && find . -type f -name '*.py' | sed 's#^\./##'); do
    printf -- '-v %s/candidate/%s:%s/%s:ro ' "$LAB" "$f" "$V" "$f"
  done
}

run() {
  local gpu=$1 mounted=$2; shift 2
  local args=(--rm --network none --cpuset-cpus 0-9 --memory 24g -w "$V" -e HF_HUB_OFFLINE=1
              -v "$HF_CACHE:/root/.cache/huggingface:ro")
  [ "$gpu" = gpu ] && args+=(--gpus all --ipc host)
  local extra=()
  [ "$mounted" = candidate ] && extra=($(mounts))
  docker run "${args[@]}" "${extra[@]}" --entrypoint python3 "$IMG" \
    -m pytest -q -p no:cacheprovider --confcutdir=tests/v1 "$@"
}

EXISTING_CPU=(tests/v1/core/test_output.py tests/v1/core/test_repetition_detection.py
              tests/v1/engine/test_output_processor.py)
EXISTING_GPU=(tests/v1/worker/test_gpu_gumbel_sample.py
              tests/v1/worker/test_gpu_rejection_sampler_chunking.py
              tests/v1/worker/test_gpu_rejection_sampler_i64.py
              tests/v1/worker/test_gpu_sampler_flags.py
              tests/v1/worker/test_dead_rows.py
              tests/v1/worker/test_draft_tokens_handler.py
              tests/v1/worker/test_async_verified_draft_counts.py
              tests/v1/worker/test_gpu_input_batch_v2.py
              tests/v1/spec_decode/test_rejection_sampler_utils.py
              tests/v1/spec_decode/test_synthetic_rejection_sampler_utils.py
              tests/v1/spec_decode/test_dflash_prepare_inputs.py
              tests/v1/spec_decode/test_adaptive_verification.py
              tests/v1/sample/test_rejection_sampler.py tests/v1/sample/test_sampler.py)
SCHEDULER=(tests/v1/core/test_scheduler.py tests/v1/core/test_async_scheduler.py)
PARSERS=(tests/tokenizers_/test_deepseek_v41.py tests/tokenizers_/test_deepseek_v4.py
         tests/tokenizers_/test_deepseek_v41_effort.py
         tests/parser/engine/test_deepseek_v4.py tests/parser/engine/test_deepseek_v41.py)

case ${1:-} in
  sync)
    src=${2:-$HOME/projects/vllm-ds41-sink-rejection}
    (cd "$src" && { git diff --name-only $BASE -- vllm tests; git ls-files --others --exclude-standard vllm tests; } \
      | sort -u | while read -r f; do [ -f "$f" ] && echo "$f"; done > /tmp/sink-v2-files.txt
     tar -C "$src" -cf - -T /tmp/sink-v2-files.txt) \
      | ssh dgx4 "rm -rf ~/spark-lab/sink-rejection-v2/candidate && mkdir -p ~/spark-lab/sink-rejection-v2/candidate && tar -C ~/spark-lab/sink-rejection-v2/candidate -xf -"
    wc -l < /tmp/sink-v2-files.txt
    ;;
  cpu)
    echo "== sink rejection: scheduler and frontend (CPU)"
    run cpu candidate tests/v1/core/test_sink_draft_zero.py
    echo "== existing CPU suites: scheduler output, repetition stop, output processor"
    run cpu candidate "${EXISTING_CPU[@]}"
    echo "== V4.1 and V4 tokenizers and parsers (CPU)"
    run cpu candidate --confcutdir=tests/parser "${PARSERS[@]}"
    ;;
  gpu)
    echo "== sink rejection: kernels, samplers and the collapse sequence (GPU)"
    run gpu candidate tests/v1/worker/test_gpu_sink.py
    echo "== existing sampling, verification and drafter-input suites (GPU)"
    run gpu candidate "${EXISTING_GPU[@]}"
    echo "== scheduler suites (GPU container)"
    run gpu candidate "${SCHEDULER[@]}"
    ;;
  baseline)
    echo "== existing suites on the untouched image"
    run cpu image "${EXISTING_CPU[@]}"
    run gpu image "${EXISTING_GPU[@]}"
    run gpu image "${SCHEDULER[@]}"
    ;;
  *) echo "usage: $0 sync|cpu|gpu|baseline" >&2; exit 2 ;;
esac
