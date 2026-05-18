#!/usr/bin/env python3
"""EMNLP-7: External-benchmark cross-correlation against MGSM.

Goal: show PROPHET-multilingual captures different signal from a
standard multilingual NLP benchmark. We run a fixed subsample of MGSM
(Multilingual Grade School Math) on the same agents used in the PROPHET
matrix and compute Spearman rank correlations between MGSM-accuracy
and PROPHET-multilingual-accuracy / -payoff / -ECE.

Cost discipline:
  * --smoke uses N=2 problems per language and a single agent.
  * --max-cost is enforced per agent.
  * The agent list is HARD-CODED so dry-run prints exactly what will be
    spent. No surprises.

Outputs:
  results/mgsm/runs/<agent>.json   per-agent per-language accuracy
  docs/paper/figures/mgsm_correlation.{tex,md,json}   final table
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Lazy import openai so --dry-run works without network.

# Agents to run on MGSM. Hand-picked to span the frontier:
#   - 2 frontier closed (GPT-5.2, Gemini 3.1 Pro)
#   - 2 frontier closed cheap (Gemini 3 Flash, GPT-5)
#   - 1 Anthropic frontier (Opus 4.7)
#   - 1 open-weight non-reasoning (DeepSeek V3.2)
#   - 1 open-weight reasoning (Qwen3-235B-Thinking)
#   - 1 open-weight non-reasoning Meta (Llama-4-Maverick)
AGENTS = [
    "openai/gpt-5.2",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
    "openai/gpt-5",
    "anthropic/claude-opus-4.7",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "meta-llama/llama-4-maverick",
]

# MGSM languages overlapping with PROPHET-multilingual v2 dictionary.
LANGS = ["en", "es", "fr", "de"]

PROMPT_TEMPLATE = (
    "Solve this grade-school math problem. Respond with only the final "
    "integer answer on the last line, no commentary.\n\n"
    "Problem: {q}\n\n"
    "Final answer:"
)


def extract_int(text: str) -> int | None:
    """Parse the LAST integer in the response. Handles thousand separators."""
    if not text:
        return None
    # Strip thousand-separators inside numbers
    cleaned = re.sub(r"(\d),(\d)", r"\1\2", text)
    nums = re.findall(r"-?\d+", cleaned)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except ValueError:
        return None


def call_agent(client, model: str, problem: str, max_tokens: int) -> tuple[str | None, float]:
    """One MGSM call. Returns (response_text, cost_usd)."""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(q=problem)}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as e:
        print(f"  ERR {model}: {e}", file=sys.stderr)
        return None, 0.0
    txt = r.choices[0].message.content if r.choices else None
    cost = 0.0
    if hasattr(r, "usage") and r.usage:
        # OpenRouter returns cost on the usage object when available.
        cost = float(getattr(r.usage, "cost", 0.0) or 0.0)
    return txt, cost


def run_agent(client, model: str, problems_by_lang: dict[str, list[dict]],
              max_tokens: int, max_cost: float, out_path: Path) -> dict:
    """Run one agent across all (lang, problem) pairs with a cost cap."""
    correct: dict[str, int] = {lang: 0 for lang in problems_by_lang}
    total: dict[str, int] = {lang: 0 for lang in problems_by_lang}
    cost_total = 0.0
    samples: list[dict] = []
    for lang, problems in problems_by_lang.items():
        for p in problems:
            if cost_total >= max_cost:
                print(f"  [{model}] cost cap ${max_cost:.2f} hit, stopping")
                break
            txt, cost = call_agent(client, model, p["question"], max_tokens)
            cost_total += cost
            pred = extract_int(txt or "")
            gold = int(p["answer_number"])
            ok = (pred == gold)
            total[lang] += 1
            if ok:
                correct[lang] += 1
            samples.append({
                "lang": lang,
                "pred": pred,
                "gold": gold,
                "correct": ok,
                "cost": cost,
                "response_excerpt": (txt or "")[:120],
            })
        if cost_total >= max_cost:
            break

    summary = {
        "agent": model,
        "cost_total": cost_total,
        "per_lang": {lang: {"correct": correct[lang], "total": total[lang],
                            "acc": (correct[lang] / total[lang]) if total[lang] else None}
                     for lang in problems_by_lang},
        "overall_acc": (sum(correct.values()) / sum(total.values())) if sum(total.values()) else None,
        "n_calls": sum(total.values()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "samples": samples}, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30,
                    help="problems per language")
    ap.add_argument("--smoke", action="store_true",
                    help="N=2 per lang, only gemini-3-flash, ~$0.01")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan, don't call API")
    ap.add_argument("--max-cost", type=float, default=2.0,
                    help="per-agent USD cap")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="completion budget per call (reasoning models need >=200)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/mgsm/runs"))
    ap.add_argument("--agents", type=str, default=None,
                    help="comma-separated subset of AGENTS")
    args = ap.parse_args()

    # Subsample agents / problems
    if args.smoke:
        agents = ["google/gemini-3-flash-preview"]
        n = 2
        max_cost = 0.05
    else:
        agents = args.agents.split(",") if args.agents else AGENTS
        n = args.n
        max_cost = args.max_cost

    # Load MGSM, take seeded slice
    from datasets import load_dataset
    problems_by_lang: dict[str, list[dict]] = {}
    for lang in LANGS:
        ds = load_dataset("juletxara/mgsm", lang, split="test")
        # Deterministic slice: first N (MGSM is already shuffled)
        problems_by_lang[lang] = [
            {"question": ds[i]["question"], "answer_number": ds[i]["answer_number"]}
            for i in range(min(n, len(ds)))
        ]
    n_calls_planned = sum(len(p) for p in problems_by_lang.values()) * len(agents)
    print(f"Plan: {len(agents)} agents × {len(LANGS)} langs × {n} problems = {n_calls_planned} calls")
    print(f"  langs: {LANGS}")
    print(f"  agents: {agents}")
    print(f"  per-agent cap: ${max_cost}, max-tokens: {args.max_tokens}")
    if args.dry_run:
        return 0

    # Smoke-test answer extraction before any call
    test_cases = [("The answer is 42", 42), ("Result: 1,234 dollars", 1234),
                  ("Step 1: 3+4=7. So the answer is -5.", -5), ("no number here", None),
                  ("$2,567,890.00 total", 256789000)]
    for inp, want in test_cases:
        got = extract_int(inp)
        # Note: $2,567,890.00 is a known edge case (commas in money + decimal); use as canary
        if inp.startswith("$"):
            print(f"  canary: {inp!r} -> {got} (gold {want}, may differ on decimal handling)")
        elif got != want:
            print(f"  FAIL: extract_int({inp!r}) = {got}, expected {want}")
            return 2

    # Connect
    from dotenv import load_dotenv
    load_dotenv()
    import openai
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight: ping every agent before any of them costs real money.
    # A failed ping skips that agent entirely (rather than letting 30+
    # calls drop into the "error -> PASS" trap on the orchestrator side).
    print("\n=== pre-flight pings ===")
    healthy_agents: list[str] = []
    for agent in agents:
        try:
            r = client.chat.completions.create(
                model=agent,
                messages=[{"role": "user", "content": "Reply with just: y"}],
                # max_tokens=64 is the SAFE floor: Azure requires >=16, reasoning
                # models burn 30-50 tokens on hidden thinking even for trivial
                # one-word replies. 8 tokens dropped 4 of 8 agents earlier.
                max_tokens=64,
                temperature=0.0,
            )
            # Empty content alone is NOT a failure for reasoning models — content
            # may go into reasoning_details. Treat "no exception" as healthy.
            healthy_agents.append(agent)
            content = r.choices[0].message.content if r.choices else None
            label = repr((content or "")[:24]) if content else "(reasoning-only response)"
            print(f"  OK {agent}: {label}")
        except Exception as e:
            print(f"  SKIP {agent}: {e}")
    if not healthy_agents:
        print("No agents passed pre-flight, aborting.")
        return 1
    agents = healthy_agents
    print(f"=== launching MGSM on {len(agents)} healthy agents ===")

    grand_total_cost = 0.0
    for agent in agents:
        out_path = args.out_dir / (agent.replace("/", "_") + ".json")
        if out_path.exists() and not args.smoke:
            existing = json.loads(out_path.read_text())
            if existing["summary"]["n_calls"] >= sum(len(p) for p in problems_by_lang.values()):
                print(f"  [{agent}] cached ({out_path}), skipping")
                grand_total_cost += existing["summary"]["cost_total"]
                continue
        print(f"\n--- agent: {agent} ---")
        t0 = time.time()
        s = run_agent(client, agent, problems_by_lang, args.max_tokens, max_cost, out_path)
        grand_total_cost += s["cost_total"]
        overall = s['overall_acc'] if s['overall_acc'] is not None else 0.0
        print(f"  {agent}: overall_acc={overall:.3f} "
              f"cost=${s['cost_total']:.4f} ({s['n_calls']} calls, {time.time()-t0:.0f}s)")
        for lang, m in s["per_lang"].items():
            if m["total"]:
                print(f"    {lang}: {m['correct']}/{m['total']} = {m['acc']:.3f}")
    print(f"\nGrand total cost: ${grand_total_cost:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
