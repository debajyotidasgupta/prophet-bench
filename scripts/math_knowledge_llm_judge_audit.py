#!/usr/bin/env python3
"""EMNLP-4: LLM-judge audit of the math and knowledge family verifiers.

The math/knowledge verifiers use last-numeric-token / canonical-substring
matching. A reviewer concern is that this is permissive: hedging answers
("the answer is probably 7 but could be 12") or near-misses ("Paris,
roughly") get accepted. We measure the empirical false-accept and
false-reject rates against a Gemini-3-Flash judge that grades the
response against the reference answer.

Sampling: stratified, ~150 responses across the matrix to cap cost.
Judge: openrouter:google/gemini-3-flash-preview (clean text output).

Writes: docs/paper/figures/math_knowledge_audit.{tex,md,json}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _llm_judge(prompt: str, reference: str, response: str, client) -> bool | None:
    q = (
        "An evaluator asked an AI assistant a question with a known reference "
        "answer. Did the assistant's response give the correct answer? Accept "
        "the response as correct iff it explicitly states an answer that "
        "matches the reference (up to trivial formatting). Reject if the "
        "response is wrong, evades the question, or hedges without committing.\n\n"
        f"QUESTION:\n{prompt}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"AI RESPONSE:\n{response}\n\n"
        "Reply with exactly 'CORRECT' or 'WRONG' on a single line."
    )
    try:
        resp = client.chat.completions.create(
            model="google/gemini-3-flash-preview",
            messages=[{"role": "user", "content": q}],
            max_tokens=24, temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip().upper()
        if "CORRECT" in text:
            return True
        if "WRONG" in text:
            return False
        return None
    except Exception as e:
        print(f"  judge error: {e}", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path,
                    default=Path("results/openrouter_matrix/runs"))
    ap.add_argument("--families", nargs="+", default=["math", "knowledge"])
    ap.add_argument("--max-per-family", type=int, default=80,
                    help="Cap LLM-judge calls per family (cost guardrail)")
    ap.add_argument("--out", type=Path,
                    default=Path("docs/paper/figures/math_knowledge_audit.tex"))
    args = ap.parse_args()

    sys.path.insert(0, "src")
    from prophet.families.math_family import MathFamily
    from prophet.families.knowledge_family import KnowledgeFamily
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        with open(".env") as f:
            for ln in f:
                if ln.startswith("OPENROUTER_API_KEY="):
                    api_key = ln.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found")
        return 1

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Regenerate task pool with reference answers
    ref_by_id: dict[str, dict] = {}
    fams = {"math": MathFamily, "knowledge": KnowledgeFamily}
    for fam_name, cls in fams.items():
        if fam_name not in args.families:
            continue
        for seed in (42, 7, 1729):
            for n in (20, 40, 100, 240):
                try:
                    tasks = cls().generate(n=n, seed=seed)
                    for t in tasks:
                        if t.task_id not in ref_by_id:
                            ref_by_id[t.task_id] = {
                                "prompt": t.prompt,
                                "reference": str(t.reference_answer),
                                "family": fam_name,
                            }
                except Exception:
                    pass
    print(f"Regenerated {len(ref_by_id)} task references across {args.families}")

    # Walk outcomes
    records_by_fam: dict[str, list[dict]] = defaultdict(list)
    for d in sorted(args.runs_dir.iterdir()):
        p = d / "outcomes.jsonl"
        if not p.exists():
            continue
        name = d.name.split("-", 2)[-1]
        if name.startswith("baseline:"):
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            fam = o.get("family")
            if fam not in args.families:
                continue
            if o.get("success") is None:
                continue
            ref = ref_by_id.get(o["task_id"])
            if not ref:
                continue
            resp = o.get("response", {})
            text = resp.get("text") or resp.get("answer") or ""
            records_by_fam[fam].append({
                "agent": name, "task_id": o["task_id"], "family": fam,
                "verifier_pass": bool(o["success"]),
                "prompt": ref["prompt"][:600],
                "reference": ref["reference"][:200],
                "response": text[:600],
            })

    # Cap per-family
    sampled: list[dict] = []
    for fam in args.families:
        rs = records_by_fam[fam]
        rs.sort(key=lambda r: r["task_id"])  # deterministic ordering
        if len(rs) > args.max_per_family:
            # Stratified sample by agent
            by_agent = defaultdict(list)
            for r in rs:
                by_agent[r["agent"]].append(r)
            per = max(1, args.max_per_family // max(len(by_agent), 1))
            taken: list[dict] = []
            for ag in by_agent:
                taken.extend(by_agent[ag][:per])
            sampled.extend(taken[: args.max_per_family])
        else:
            sampled.extend(rs)
    print(f"Sampled {len(sampled)} responses for LLM-judge audit")

    # Run judge
    judge_calls = 0
    for i, r in enumerate(sampled, 1):
        if i % 40 == 0:
            print(f"  {i}/{len(sampled)}...")
        verdict = _llm_judge(r["prompt"], r["reference"], r["response"], client)
        if verdict is None:
            continue
        r["judge_pass"] = verdict
        judge_calls += 1

    # Aggregate by family
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    for fam in args.families:
        fam_records = [r for r in sampled if r["family"] == fam and "judge_pass" in r]
        tp = fp = tn = fn = 0
        for r in fam_records:
            v, j = r["verifier_pass"], r["judge_pass"]
            if v and j: tp += 1
            elif not v and not j: tn += 1
            elif v and not j: fp += 1
            else: fn += 1
        total = len(fam_records)
        agree = (tp + tn) / max(total, 1)
        fa = fp / max(tp + fp, 1)
        fr = fn / max(tn + fn, 1)
        summary[fam] = {
            "n": total, "agree": agree, "false_accept": fa, "false_reject": fr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }
        print(f"\n=== {fam} ===")
        print(f"  N = {total}  agree = {agree*100:.1f}%  "
              f"FA = {fa*100:.1f}%  FR = {fr*100:.1f}%")
        print(f"  tp = {tp}, fp = {fp}, tn = {tn}, fn = {fn}")

    # Outputs
    (args.out.with_suffix(".json")).write_text(json.dumps({
        "judge": "openrouter:google/gemini-3-flash-preview",
        "judge_calls": judge_calls,
        "summary": summary,
    }, indent=2))
    md_lines = ["# Math + knowledge LLM-judge audit", "",
                "| Family | N | Agreement | False-accept | False-reject |",
                "|---|---|---|---|---|"]
    for fam, s in summary.items():
        md_lines.append(f"| {fam} | {s['n']} | {s['agree']*100:.1f}% | "
                        f"{s['false_accept']*100:.1f}% | {s['false_reject']*100:.1f}% |")
    (args.out.with_suffix(".md")).write_text("\n".join(md_lines))
    tex = [
        r"\begin{table}[t]\centering\small",
        rf"\caption{{LLM-judge audit of the math and knowledge family "
        rf"verifiers (mechanical last-token / canonical-substring rules). "
        rf"Judge: \texttt{{gemini-3-flash-preview}} on a "
        rf"correct-vs-wrong binary prompt with the regenerated reference "
        rf"answer. The false-accept rate bounds the over-acceptance of "
        rf"the substring rule; the false-reject rate bounds its "
        rf"over-strictness.}}",
        r"\label{tab:math-knowledge-audit}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Family & $N$ & Agreement & False-accept & False-reject \\",
        r"\midrule",
    ]
    for fam, s in summary.items():
        tex.append(
            f"{fam} & {s['n']} & {s['agree']*100:.1f}\\% & "
            f"{s['false_accept']*100:.1f}\\% & {s['false_reject']*100:.1f}\\% \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    args.out.write_text("\n".join(tex))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
