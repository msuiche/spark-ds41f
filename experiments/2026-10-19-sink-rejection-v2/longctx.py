"""Gate item 4: the collapse prompts after a long conversation.

A first turn holds a long document (Python standard-library source, servebench.prompts.source_text)
with a request to read it, a short assistant reply, then the collapse prompt as the last user turn,
thinking at effort high, temperature 0. Sizes 64K and 512K tokens; last turns `Say hi.` and
`Thanks!`. Run on r6x, the detector arm and the correction arm, it shows which contexts still
collapse on r6x, whether the detector flags them, and whether the correction answers cleanly with a
rejection deep in the compressed and sliding-window caches.

Usage: python3 longctx.py BASE_URL [--sizes 65536,524288] [--max-tokens N] [--port P]
Prints one JSON line per case, then {"summary": ...}.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from probes import COLLAPSE_MARKERS, MODEL, post

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from servebench import prompts as servebench_prompts  # noqa: E402

LAST_TURNS = ("Say hi.", "Thanks!")


def conversation(document: str, last: str) -> list[dict]:
    return [
        {"role": "user", "content": "Read this code carefully; I will ask about it later.\n\n" + document},
        {"role": "assistant", "content": "I've read the code. What would you like to know?"},
        {"role": "user", "content": last},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--sizes", default="65536,524288")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if args.port:
        base = base.rsplit(":", 1)[0] + f":{args.port}"

    rows = []
    for size in (int(s) for s in args.sizes.split(",")):
        document = servebench_prompts.source_text(size, seed=size)
        for last in LAST_TURNS:
            ids = post(base, "/tokenize", {
                "model": MODEL, "messages": conversation(document, last), "add_generation_prompt": True,
                "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}})["tokens"]
            start = time.time()
            response = post(base, "/v1/completions", {"model": MODEL, "prompt": ids,
                                                      "max_tokens": args.max_tokens, "temperature": 0,
                                                      "skip_special_tokens": False})
            text = response["choices"][0]["text"]
            row = {"size": size, "prompt_tokens": len(ids), "last": last, "seconds": round(time.time() - start, 1),
                   "text": text, "collapsed": any(m in text for m in COLLAPSE_MARKERS),
                   "sink_rejection": (response.get("usage") or {}).get("sink_rejection")}
            print(json.dumps(row), flush=True)
            rows.append(row)
    summary = {
        "cases": len(rows),
        "collapsed": [f"{r['size']}:{r['last']}" for r in rows if r["collapsed"]],
        "flagged": [f"{r['size']}:{r['last']}" for r in rows if (r["sink_rejection"] or {}).get("flagged_rows")],
        "rejections": {f"{r['size']}:{r['last']}": (r["sink_rejection"] or {}).get("draft_zero_rejections")
                       for r in rows if r["sink_rejection"]},
    }
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
