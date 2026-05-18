#!/usr/bin/env python3
"""Pre-flight $0.0001 ping for an OpenRouter agent.

Exits 0 if the agent answers a one-token call, else exits non-zero.
Used by orchestrator wrappers BEFORE launching a paid 40-task batch
to catch 402 / 401 / model-unavailable errors at $0.0001 instead of
silently filling the run with PASS outcomes.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: preflight_agent_ping.py <openrouter_model_id>", file=sys.stderr)
        return 2
    model = sys.argv[1].removeprefix("openrouter:")
    load_dotenv()
    import openai
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single character: y"}],
            max_tokens=8,
            temperature=0.0,
        )
    except Exception as e:
        print(f"preflight FAIL {model}: {e}", file=sys.stderr)
        return 1
    content = r.choices[0].message.content if r.choices else None
    cost = float(getattr(r.usage, "cost", 0.0) or 0.0) if r.usage else 0.0
    if not content:
        print(f"preflight FAIL {model}: empty response (reasoning model likely needs more max_tokens)",
              file=sys.stderr)
        return 1
    print(f"preflight OK {model}: {content[:32]!r}  (cost ${cost:.5f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
