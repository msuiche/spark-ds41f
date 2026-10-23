"""Survey of high sink-column rows on broad text, for the detector's threshold.

Run against the detector-only survey arm (threshold 20,000; decoding is r6x's). Each request goes
through /v1/completions with the rendered prompt and per-token text (logprobs 0), at 8 streams.
Afterwards the scheduler's "Sink:" log lines since the start (each carries the largest
|residual[column]| among the request's flagged rows that step) are joined to the responses, with
the tokens around the flagged position.

Sets:
  default      servebench's 256 default prompts, the server's default sampling and reasoning effort
  extra        the other 256 template-topic pairs (first --limit), greedy
  probes       the thinking-collapse experiment's 16 prompts, greedy, thinking at effort high
  attribution  d027, d045 and x203 greedy (the rows checked on DeepSeek's reference), keeping the
               generated token ids

Since vLLM 5ac8364ec the Sink: line names the output position of the row holding the value (row r
reads output token first + r); each event is keyed on that row, whose input token the window
brackets. Older lines carry only the step's next output position.

Usage: python3 survey.py BASE_URL --set default|extra|probes [--limit N] [--max-tokens N]
Prints one JSON line per request with its events, then {"summary": ...} (lab.py keeps stdout).
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from servebench import prompts as servebench_prompts  # noqa: E402

MODEL = "deepseek-v4.1-flash"
CONTAINER = "dsv41-karmic-kraken"
PROBES = ["Say hi.", "Hello", "Thanks!", "Tell me a joke.", "What's 2+2?", "Who are you?",
          "Translate 'good morning' to French.", "Write one sentence about the sea.", "Good night!",
          "Thank you so much, that worked.", "Can you help me?", "Hi there, how are you?",
          "ok", "Great, thanks for the help!", "What day comes after Monday?", "Name a color."]
EVENT = re.compile(r"Sink: request (\S+), next output position (\d+): (\w+) row (\w+), (\d+) drafter "
                   r"row\(s\) flagged(?:, cut at row (\d+))?, value (\d+)"
                   r"(?: at output position (-?\d+) \(row (-?\d+)\))?")
BINS = (20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000)


def post(base: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read())


def prompt_set(name: str, limit: int | None) -> list[dict]:
    if name == "attribution":
        pool = {p["id"]: p["text"] for p in prompt_set("default", None) + prompt_set("extra", None)}
        return [{"id": i, "text": pool[i], "temperature": 0.0, "kwargs": None, "keep_ids": True}
                for i in ("d027", "d045", "x203")]
    if name == "probes":
        items = [{"id": f"p{i:02d}", "text": text, "temperature": 0.0,
                  "kwargs": {"thinking": True, "reasoning_effort": "high"}}
                 for i, text in enumerate(PROBES)]
    elif name == "default":
        items = [{**p, "temperature": None, "kwargs": None} for p in servebench_prompts.default_prompts()]
    else:
        used = {p["text"] for p in servebench_prompts.default_prompts()}
        items = [{"id": f"x{i:03d}", "text": text, "temperature": 0.0, "kwargs": None}
                 for i, text in enumerate(t.format(topic=topic) for topic in servebench_prompts.TOPICS
                                          for t in servebench_prompts.TEMPLATES)
                 if text not in used]
    return items[:limit] if limit else items


def run_one(base: str, item: dict, max_tokens: int) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": item["text"]}],
            "add_generation_prompt": True}
    if item["kwargs"]:
        body["chat_template_kwargs"] = item["kwargs"]
    ids = post(base, "/tokenize", body)["tokens"]
    request = {"model": MODEL, "prompt": ids, "max_tokens": max_tokens, "logprobs": 0,
               "skip_special_tokens": False}
    if item["temperature"] is not None:
        request["temperature"] = item["temperature"]
    if item.get("keep_ids"):
        request["return_tokens_as_token_ids"] = True
    response = post(base, "/v1/completions", request)
    choice = response["choices"][0]
    result = {"id": item["id"], "prompt": item["text"], "request_id": response["id"],
              "prompt_tokens": len(ids), "tokens": choice["logprobs"]["tokens"],
              "finish_reason": choice["finish_reason"]}
    if item.get("keep_ids"):
        result["prompt_ids"] = ids
        result["token_ids"] = [int(t.split(":", 1)[1]) for t in result["tokens"]]
        result["tokens"] = [post(base, "/detokenize", {"model": MODEL, "tokens": [t]})["prompt"]
                            for t in result["token_ids"]]
    return result


def scheduler_events(since: str) -> list[dict]:
    logs = subprocess.run(["docker", "logs", "--since", since, CONTAINER], capture_output=True, text=True)
    events = []
    for line in (logs.stdout + logs.stderr).splitlines():
        found = EVENT.search(line)
        if found:
            request, position, row, outcome, drafters, cut, value, row_position, value_row = found.groups()
            events.append({"request": request, "position": int(position), "row": row, "outcome": outcome,
                           "drafter_flags": int(drafters), "cut_row": None if cut is None else int(cut),
                           "value": int(value),
                           "row_position": None if row_position is None else int(row_position),
                           "value_row": None if value_row is None else int(value_row)})
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--set", choices=("default", "extra", "probes", "attribution"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 1))
    items = prompt_set(args.set, args.limit)
    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(lambda item: run_one(base, item, args.max_tokens), items))
    time.sleep(2)
    events = scheduler_events(since)

    by_request = {r["request_id"]: r for r in results}
    joined = []
    for event in events:
        result = next((r for rid, r in by_request.items() if event["request"].startswith(rid + "-")), None)
        if result is None:
            continue
        tokens = result["tokens"]
        q = event["row_position"]
        if q is None:  # an older line: the step's next output position
            p = event["position"]
            window = "".join(tokens[max(0, p - 8):p]) + "‖" + "".join(tokens[p:p + 6])
        elif q < 0:
            window = "[last prompt token]" + "".join(tokens[:6])
        else:
            window = "".join(tokens[max(0, q - 8):q]) + "[" + "".join(tokens[q:q + 1]) + "]" + "".join(tokens[q + 1:q + 6])
        event = {**event, "prompt_id": result["id"], "window": window}
        result.setdefault("events", []).append(event)
        joined.append(event)
    for result in results:
        tokens = result.pop("tokens")
        result["generated_tokens"] = len(tokens)
        print(json.dumps(result), flush=True)

    values = sorted((e["value"] for e in joined), reverse=True)
    summary = {
        "set": args.set,
        "requests": len(results),
        "generated_tokens": sum(r["generated_tokens"] for r in results),
        "events": len(joined),
        "unmatched_events": len(events) - len(joined),
        "events_at_or_above": {str(b): sum(v >= b for v in values) for b in BINS},
        "requests_with_an_event_at_or_above": {
            str(b): sum(any(e["value"] >= b for e in r.get("events", [])) for r in results) for b in BINS},
        "top": sorted(joined, key=lambda e: -e["value"])[:25],
    }
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
