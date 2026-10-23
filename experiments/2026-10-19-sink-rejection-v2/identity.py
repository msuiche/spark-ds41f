"""Gate item 7: greedy tokens with and without speculation.

The first 128 of servebench's default prompts (the screen's model-step prompts, server-default
rendering) and the 16 collapse probes (thinking high), temperature 0, 256 and 300 tokens, at eight
streams; every reply's token ids. Run on the correction arm with speculation and on
correction-nospec.json (port 8001); analysis/gate_identity.py compares them token for token.

Usage: python3 identity.py BASE_URL [--port P]
Prints one JSON line per prompt (id, token ids, sink usage), then {"summary": ...}.
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from probes import MODEL, PROMPTS, post

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from servebench import prompts as servebench_prompts  # noqa: E402


def run(base: str, item: dict) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": item["text"]}], "add_generation_prompt": True}
    if item["kwargs"]:
        body["chat_template_kwargs"] = item["kwargs"]
    ids = post(base, "/tokenize", body)["tokens"]
    response = post(base, "/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": item["max_tokens"],
                                              "temperature": 0, "logprobs": 0, "skip_special_tokens": False,
                                              "return_tokens_as_token_ids": True})
    choice = response["choices"][0]
    tokens = [int(t.split(":", 1)[1]) for t in choice["logprobs"]["tokens"]]
    return {"id": item["id"], "tokens": tokens, "sha": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16],
            "sink_rejection": (response.get("usage") or {}).get("sink_rejection")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if args.port:
        base = base.rsplit(":", 1)[0] + f":{args.port}"
    items = [{"id": p["id"], "text": p["text"], "kwargs": None, "max_tokens": 256}
             for p in servebench_prompts.default_prompts()[:128]]
    items += [{"id": f"p{i:02d}", "text": t, "kwargs": {"thinking": True, "reasoning_effort": "high"},
               "max_tokens": 300} for i, t in enumerate(PROMPTS)]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda item: run(base, item), items))
    for result in results:
        print(json.dumps(result), flush=True)
    print(json.dumps({"summary": {"prompts": len(results),
                                  "flagged": [r["id"] for r in results
                                              if (r["sink_rejection"] or {}).get("flagged_rows")],
                                  "sha": hashlib.sha256("".join(r["sha"] for r in results).encode()).hexdigest()[:16]}}),
          flush=True)


if __name__ == "__main__":
    main()
