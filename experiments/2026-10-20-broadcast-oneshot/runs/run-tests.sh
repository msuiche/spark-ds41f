#!/usr/bin/env bash
# Unit tests for vLLM 0185 (the TP group's broadcast on the RoCE policy), in a throwaway
# CPU container from the promoted r6x image on an idle node, with the candidate's changed
# files mounted over the image's sources (the image's vLLM tree is the candidate's base,
# r6x 7316c7f9f, tree 4d6851c8). No network, no GPU, no serving node touched.
#
#   run-tests.sh sync [node]       (on the Mac) copies overlay/ to <node>:$LAB/candidate
#   run-tests.sh cpu               the policy suite with the candidate's files
#   run-tests.sh baseline          the same suite on the untouched image
#
# --confcutdir=tests/distributed skips the root tests/conftest.py (tblib is not in the image).
set -u
LAB=${LAB:-$HOME/spark-lab/broadcast-oneshot}
V=/opt/spark3/candidate/vllm
IMG=vllm-ds41f-kkref:19f2c20ed4d6-r6x
SUITE=tests/distributed/test_roce_policy_no_fallback.py

mounts() {
  local f
  for f in $(cd "$LAB/candidate" && find . -type f -name '*.py' | sed 's#^\./##'); do
    printf -- '-v %s/candidate/%s:%s/%s:ro ' "$LAB" "$f" "$V" "$f"
  done
}

run() {
  local mounted=$1; shift
  local extra=()
  [ "$mounted" = candidate ] && extra=($(mounts))
  docker run --rm --network none --cpuset-cpus 0-4 --memory 8g -w "$V" "${extra[@]}" \
    --entrypoint python3 "$IMG" -m pytest -q -p no:cacheprovider --confcutdir=tests/distributed "$@"
}

case ${1:-} in
  sync)
    node=${2:-dgx4}
    here=$(cd "$(dirname "$0")/.." && pwd)
    tar -C "$here/overlay" -cf - . \
      | ssh "$node" "rm -rf ~/spark-lab/broadcast-oneshot/candidate && mkdir -p ~/spark-lab/broadcast-oneshot/candidate && tar -C ~/spark-lab/broadcast-oneshot/candidate -xf -"
    ;;
  cpu)
    echo "== RoCE policy, broadcast included (candidate)"
    run candidate "$SUITE"
    ;;
  baseline)
    echo "== RoCE policy (untouched image)"
    run image "$SUITE"
    ;;
  *) echo "usage: $0 sync [node]|cpu|baseline" >&2; exit 2 ;;
esac
