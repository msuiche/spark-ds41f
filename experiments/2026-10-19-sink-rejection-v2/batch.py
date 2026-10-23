"""Gate item 3: the collapse prompts inside a concurrent batch.

The 16 probes (probes.py's prompts, thinking high, temperature 0) one at a time, then all 16 at
eight streams with staggered arrivals (one every 0.25 s), so Say hi. and Thanks! run beside the
other 14 in mixed batches. Each answer at eight streams must equal its one-stream answer and carry
the same sink flags.

Usage: python3 batch.py BASE_URL [--max-tokens N] [--stagger S] [--port P]
Prints one JSON line per prompt, then {"summary": ...}.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from probes import COLLAPSE_MARKERS, MODEL, PROMPTS, post


def ask(base: str, prompt: str, max_tokens: int) -> dict:
    ids = post(base, "/tokenize", {
        "model": MODEL, "messages": [{"role": "user", "content": prompt}], "add_generation_prompt": True,
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}})["tokens"]
    response = post(base, "/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": max_tokens,
                                              "temperature": 0, "skip_special_tokens": False})
    text = response["choices"][0]["text"]
    return {"text": text, "collapsed": any(m in text for m in COLLAPSE_MARKERS),
            "sink_rejection": (response.get("usage") or {}).get("sink_rejection")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--stagger", type=float, default=0.25)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if args.port:
        base = base.rsplit(":", 1)[0] + f":{args.port}"

    alone = [ask(base, p, args.max_tokens) for p in PROMPTS]

    def staggered(i: int) -> dict:
        time.sleep(i * args.stagger)
        return ask(base, PROMPTS[i], args.max_tokens)

    with ThreadPoolExecutor(8) as pool:
        batched = list(pool.map(staggered, range(len(PROMPTS))))
    for prompt, a, b in zip(PROMPTS, alone, batched):
        print(json.dumps({"prompt": prompt, "alone": a, "batched": b, "same_text": a["text"] == b["text"],
                          "same_flags": a["sink_rejection"] == b["sink_rejection"]}), flush=True)
    summary = {
        "prompts": len(PROMPTS),
        "same_text": sum(a["text"] == b["text"] for a, b in zip(alone, batched)),
        "same_flags": sum(a["sink_rejection"] == b["sink_rejection"] for a, b in zip(alone, batched)),
        "collapsed_alone": [p for p, a in zip(PROMPTS, alone) if a["collapsed"]],
        "collapsed_batched": [p for p, b in zip(PROMPTS, batched) if b["collapsed"]],
        "flagged_batched": [p for p, b in zip(PROMPTS, batched) if (b["sink_rejection"] or {}).get("flagged_rows")],
    }
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
