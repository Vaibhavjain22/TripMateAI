"""
TripMateAI - Agent Evaluation Script
====================================

Evaluates the multi-agent travel planner on the golden dataset
(eval_dataset.json) using 4 metrics, all reported as percentages:

    1. Routing Accuracy        - deterministic (Jaccard overlap of agents)
    2. Guardrail Accuracy      - deterministic (allowed / blocked match)
    3. HITL Interrupt Fidelity - deterministic (pauses, then resumes)
    4. Constraint Adherence    - LLM-as-a-Judge (gpt-4o-mini checklist)

Every test case is traced to LangSmith as a parent run, and the metric
scores are attached to it as feedback, so results are visible both in
the console and in the LangSmith dashboard.

Usage:
    python evaluate.py                     # run all 20 cases
    python evaluate.py --ids TC-16 TC-17   # run selected cases only
    python evaluate.py --no-judge          # skip the LLM judge (cheaper)
    python evaluate.py --max-cost 0.30     # stop if OpenAI spend exceeds $0.30
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.run_helpers import get_current_run_tree, traceable
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

load_dotenv(ROOT_DIR / ".env")
load_dotenv(BASE_DIR / ".env")
load_dotenv()

EVAL_DIR = ROOT_DIR / "evaluation"
DATASET_PATH = EVAL_DIR / "eval_dataset.json" if (EVAL_DIR / "eval_dataset.json").exists() else BASE_DIR / "eval_dataset.json"
RESULTS_PATH = EVAL_DIR / "eval_results.json"

# gpt-4o-mini pricing (USD per 1M tokens)
INPUT_PRICE_PER_M = 0.15
OUTPUT_PRICE_PER_M = 0.60

# Keeps the judge prompt small and cheap
MAX_ANSWER_CHARS = 6000


# =========================================================
# Backend import (with retry while Supabase wakes up)
# =========================================================

def load_backend(retries: int = 6, wait_seconds: int = 20):
    """
    backend.py connects to PostgreSQL at import time. A resumed Supabase
    project can take a few minutes to accept connections, so retry.
    """
    for attempt in range(1, retries + 1):
        try:
            try:
                from src import backend  # noqa: WPS433
            except ImportError:
                import backend  # noqa: WPS433

            return backend.run_travel_agent, backend.resume_travel_agent
        except Exception as exc:
            # Drop the half-imported module so the next attempt is clean.
            sys.modules.pop("backend", None)
            sys.modules.pop("src.backend", None)
            print(
                f"[setup] Backend/database not ready "
                f"(attempt {attempt}/{retries}): {type(exc).__name__}: {exc}"
            )
            if attempt == retries:
                raise
            print(f"[setup] Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)


# =========================================================
# Cost tracking
# =========================================================

def usage_cost(usage: dict[str, Any]) -> tuple[int, int, float]:
    """Convert the usage-metadata callback output into tokens and USD."""
    input_tokens = 0
    output_tokens = 0

    for model_usage in usage.values():
        input_tokens += model_usage.get("input_tokens", 0)
        output_tokens += model_usage.get("output_tokens", 0)

    cost = (
        input_tokens / 1_000_000 * INPUT_PRICE_PER_M
        + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_M
    )
    return input_tokens, output_tokens, cost


# =========================================================
# Metric 1: Routing Accuracy (deterministic)
# =========================================================

def routing_accuracy(actual_agents: list[str], expected_agents: list[str]) -> float:
    """Jaccard overlap between selected and expected agents (0.0 - 1.0)."""
    actual = set(actual_agents or [])
    expected = set(expected_agents or [])

    if not actual and not expected:
        return 1.0

    return len(actual & expected) / len(actual | expected)


# =========================================================
# Metric 2: Guardrail Accuracy (deterministic)
# =========================================================

def guardrail_accuracy(actual_allowed: bool, expected_allowed: bool) -> float:
    return 1.0 if bool(actual_allowed) == bool(expected_allowed) else 0.0


# =========================================================
# Metric 3: HITL Interrupt Fidelity (deterministic)
# =========================================================

def hitl_fidelity(draft: dict[str, Any], final: dict[str, Any] | None) -> float:
    """
    A valid travel run must:
      1. pause at the human approval gate (requires_approval == True), and
      2. after resuming, finish with a non-empty final answer.
    """
    paused = bool(draft.get("requires_approval"))

    if final is None:
        return 0.0

    resumed = (
        not final.get("requires_approval")
        and bool(str(final.get("answer", "")).strip())
    )
    return 1.0 if paused and resumed else 0.0


# =========================================================
# Metric 4: Constraint Adherence (LLM-as-a-Judge)
# =========================================================

class ConstraintVerdict(BaseModel):
    origin_met: bool | None = Field(
        description="True if the plan uses the requested origin. Null if no origin was requested."
    )
    destination_met: bool | None = Field(
        description="True if the plan covers the requested destination. Null if none was requested."
    )
    duration_met: bool | None = Field(
        description="True if the plan matches the requested number of days. Null if no duration was requested."
    )
    budget_met: bool | None = Field(
        description="True if the estimated total respects the requested budget. Null if no budget was requested."
    )
    reason: str = Field(description="One short sentence explaining any failed constraint.")


JUDGE_SYSTEM_PROMPT = """You are a strict travel-plan auditor.
Check ONLY whether the travel plan respects the user's explicit constraints.
Rules:
- Evaluate only constraints listed as requested; return null for constraints marked NOT REQUESTED.
- origin_met: the plan departs from the requested city/airport.
- destination_met: the plan is for the requested destination(s).
- duration_met: the itinerary covers the requested number of days.
- budget_met: the plan's estimated total cost is within the requested budget,
  or the plan clearly and honestly states that the budget is not feasible.
Be strict and factual. Keep the reason to one sentence."""


def build_judge():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=150,
        api_key=os.getenv("OPENAI_API_KEY"),
    ).with_structured_output(ConstraintVerdict)


def constraint_adherence(
    judge,
    query: str,
    answer: str,
    expected: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Return the share of requested constraints satisfied (0.0 - 1.0)."""

    def describe(value: Any) -> str:
        return "NOT REQUESTED" if value in (None, "") else str(value)

    human_prompt = f"""User query:
{query}

Requested constraints:
- origin: {describe(expected.get('origin'))}
- destination: {describe(expected.get('destination'))}
- duration_days: {describe(expected.get('duration_days'))}
- budget: {describe(expected.get('budget'))}

Travel plan to audit:
{answer[:MAX_ANSWER_CHARS]}
"""

    verdict: ConstraintVerdict = judge.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ]
    )

    checks = {
        "origin": (expected.get("origin"), verdict.origin_met),
        "destination": (expected.get("destination"), verdict.destination_met),
        "duration": (expected.get("duration_days"), verdict.duration_met),
        "budget": (expected.get("budget"), verdict.budget_met),
    }

    # Score only constraints that were actually requested.
    applicable = {
        name: bool(result)
        for name, (requested, result) in checks.items()
        if requested not in (None, "")
    }

    score = (
        sum(applicable.values()) / len(applicable)
        if applicable
        else 1.0
    )

    return score, {"checks": applicable, "reason": verdict.reason}


# =========================================================
# Single test case (traced to LangSmith)
# =========================================================

@traceable(name="TripMateAI Eval Case", run_type="chain")
def run_case(
    case: dict[str, Any],
    run_travel_agent,
    resume_travel_agent,
    judge,
) -> dict[str, Any]:
    query = case["query"]
    expected_allowed = case["expected_guardrail"]

    draft = run_travel_agent(query)
    final = None

    scores: dict[str, float | None] = {
        "guardrail_accuracy": guardrail_accuracy(
            draft.get("guardrail_allowed", True), expected_allowed
        ),
        "routing_accuracy": None,
        "hitl_fidelity": None,
        "constraint_adherence": None,
    }
    judge_details: dict[str, Any] | None = None

    # Routing, HITL and constraints only apply to valid travel queries.
    if expected_allowed:
        scores["routing_accuracy"] = routing_accuracy(
            draft.get("selected_agents", []), case["expected_agents"]
        )

        if draft.get("requires_approval"):
            final = resume_travel_agent(
                thread_id=draft["thread_id"],
                approved=True,
                feedback="",
            )

        scores["hitl_fidelity"] = hitl_fidelity(draft, final)

        if judge is not None:
            answer = (final or draft).get("answer", "")
            if answer.strip() and case.get("expected_constraints"):
                score, judge_details = constraint_adherence(
                    judge, query, answer, case["expected_constraints"]
                )
                scores["constraint_adherence"] = score
            else:
                scores["constraint_adherence"] = 0.0
                judge_details = {"checks": {}, "reason": "Empty answer."}

    # Attach scores to this LangSmith run as feedback (best effort).
    run_tree = get_current_run_tree()
    run_id = str(run_tree.id) if run_tree else None
    session_id = (
        str(run_tree.session_id)
        if run_tree and getattr(run_tree, "session_id", None)
        else None
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "run_id": run_id,
        "session_id": session_id,
        "thread_id": draft.get("thread_id"),
        "selected_agents": draft.get("selected_agents", []),
        "expected_agents": case["expected_agents"],
        "guardrail_allowed": draft.get("guardrail_allowed", True),
        "scores": scores,
        "judge": judge_details,
    }


def push_feedback(
    client: Client | None,
    result: dict[str, Any],
    project_id: str | None = None,
) -> None:
    if client is None or not result.get("run_id"):
        return

    session_id = result.get("session_id") or project_id

    for key, score in result["scores"].items():
        if score is None:
            continue
        try:
            client.create_feedback(
                result["run_id"],
                key=key,
                score=score,
                session_id=session_id,
            )
        except Exception as exc:
            print(f"  [langsmith] Could not log feedback '{key}': {exc}")


# =========================================================
# Reporting
# =========================================================

def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: float | None) -> str:
    return "   n/a" if value is None else f"{value * 100:6.1f}%"


def print_report(results: list[dict[str, Any]], tokens: tuple[int, int, float]) -> dict[str, Any]:
    metric_names = {
        "routing_accuracy": "Supervisor Routing Accuracy",
        "guardrail_accuracy": "Safety Guardrail Accuracy",
        "hitl_fidelity": "HITL Interrupt Fidelity",
        "constraint_adherence": "Constraint Adherence (LLM Judge)",
    }

    summary: dict[str, float | None] = {}
    for key in metric_names:
        values = [
            r["scores"][key] for r in results if r["scores"].get(key) is not None
        ]
        summary[key] = average(values)

    available = [v for v in summary.values() if v is not None]
    overall = average(available)

    input_tokens, output_tokens, cost = tokens

    line = "=" * 62
    print(f"\n{line}")
    print("               TripMateAI Evaluation Results")
    print(line)
    print(f"Test cases run : {len(results)}")
    print(f"Tokens used    : {input_tokens:,} in / {output_tokens:,} out")
    print(f"OpenAI cost    : ~${cost:.5f} (agent runs + judge)")
    print("-" * 62)

    for index, (key, label) in enumerate(metric_names.items(), start=1):
        print(f"{index}. {label:<36}: {fmt(summary[key])}")

    print("-" * 62)
    print(f"{'OVERALL SYSTEM SCORE':<39}: {fmt(overall)}")
    print(line)

    failures = [
        r for r in results
        if any(score is not None and score < 1.0 for score in r["scores"].values())
    ]
    if failures:
        print("\nCases below 100% on at least one metric:")
        for r in failures:
            low = {
                k: f"{v * 100:.0f}%"
                for k, v in r["scores"].items()
                if v is not None and v < 1.0
            }
            reason = (r.get("judge") or {}).get("reason", "")
            print(f"  - {r['id']}: {low} {('| ' + reason) if reason else ''}")

    return {
        "metrics_percent": {
            k: (round(v * 100, 1) if v is not None else None)
            for k, v in summary.items()
        },
        "overall_percent": round(overall * 100, 1) if overall is not None else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 4),
    }


# =========================================================
# Main
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the TripMateAI agent.")
    parser.add_argument("--ids", nargs="*", help="Run only these test case IDs.")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge.")
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="Seconds to wait between cases (avoids API rate limits).",
    )
    parser.add_argument(
        "--max-cost", type=float, default=0.50,
        help="Stop the run if OpenAI spend exceeds this many USD.",
    )
    args = parser.parse_args()

    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if args.ids:
        wanted = set(args.ids)
        cases = [c for c in cases if c["id"] in wanted]

    if not cases:
        print("No matching test cases found.")
        return

    run_travel_agent, resume_travel_agent = load_backend()
    judge = None if args.no_judge else build_judge()

    try:
        ls_client = Client() if os.getenv("LANGSMITH_API_KEY") else None
    except Exception as exc:
        print(f"[langsmith] Client unavailable, scores stay local only: {exc}")
        ls_client = None

    project_id = None
    if ls_client:
        try:
            project = ls_client.read_project(
                project_name=os.getenv("LANGSMITH_PROJECT", "default")
            )
            project_id = str(project.id)
        except Exception as exc:
            print(f"[langsmith] Could not resolve project id: {exc}")

    results: list[dict[str, Any]] = []
    usage_totals: dict[str, Any] = {}

    print(f"\nRunning {len(cases)} test case(s)...\n")

    with get_usage_metadata_callback() as usage_cb:
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case['id']} ({case['category']})")

            try:
                result = run_case(case, run_travel_agent, resume_travel_agent, judge)
            except Exception as exc:
                print(f"  ERROR: {type(exc).__name__}: {exc}")
                result = {
                    "id": case["id"],
                    "category": case["category"],
                    "run_id": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "scores": {
                        "guardrail_accuracy": 0.0,
                        "routing_accuracy": 0.0 if case["expected_guardrail"] else None,
                        "hitl_fidelity": 0.0 if case["expected_guardrail"] else None,
                        "constraint_adherence": (
                            0.0 if case["expected_guardrail"] and judge else None
                        ),
                    },
                    "judge": None,
                }

            push_feedback(ls_client, result, project_id)
            results.append(result)

            scores_text = ", ".join(
                f"{k}={v * 100:.0f}%"
                for k, v in result["scores"].items()
                if v is not None
            )
            _, _, spent = usage_cost(usage_cb.usage_metadata)
            print(f"  -> {scores_text} | spent so far ~${spent:.5f}")

            if spent > args.max_cost:
                print(
                    f"\nStopping early: spend ~${spent:.4f} exceeded "
                    f"--max-cost ${args.max_cost:.2f}."
                )
                break

            if index < len(cases):
                time.sleep(args.delay)

        usage_totals = dict(usage_cb.usage_metadata)

    summary = print_report(results, usage_cost(usage_totals))

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "summary": summary,
                "cases": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nDetailed results saved to: {RESULTS_PATH.name}")
    if ls_client:
        project = os.getenv("LANGSMITH_PROJECT", "default")
        print(f"Traces and scores logged to LangSmith project: {project}")


if __name__ == "__main__":
    main()
