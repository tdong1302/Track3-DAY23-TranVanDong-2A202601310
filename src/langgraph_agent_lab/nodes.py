"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from .llm import get_llm
from .state import AgentState, Route, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Pydantic schema for structured classification output ─────────────
class ClassifyOutput(BaseModel):
    """Structured output for LLM classification."""

    route: str
    risk_level: str
    reasoning: str


# ─── implement ALL nodes below ────────────────────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    Uses .with_structured_output() to get a reliable enum classification.
    Priority: risky > tool > missing_info > error > simple
    """
    query = state.get("query", "")
    valid_routes = [r.value for r in Route if r not in (Route.DEAD_LETTER, Route.DONE)]

    system_prompt = (
        "You are a support-ticket routing classifier. "
        "Classify the user's support query into exactly one of these routes: "
        "simple, tool, missing_info, risky, error.\n\n"
        "Route definitions and priority (highest to lowest):\n"
        "1. risky   — Requests with side effects that require human approval "
        "(e.g., refunds, account deletion, financial changes, destructive ops).\n"
        "2. tool    — Requests that require looking up data or invoking a system tool "
        "(e.g., order status, account lookup).\n"
        "3. missing_info — The query is too vague or incomplete to action "
        "(e.g., 'fix it', 'help me', no specifics).\n"
        "4. error   — The query describes a system/technical error or failure "
        "(e.g., timeout, crash, system failure).\n"
        "5. simple  — A general FAQ question that can be answered directly "
        "(e.g., 'how do I reset password?').\n\n"
        "When a query matches multiple routes, choose the highest priority one.\n"
        "Set risk_level to 'high' for risky routes, 'low' otherwise.\n"
        "Provide a short reasoning (1-2 sentences)."
    )

    user_prompt = f"Support ticket: {query}"

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(ClassifyOutput)
        decision: ClassifyOutput = structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        # Validate route is in allowed set; fallback to simple if not
        route = decision.route if decision.route in valid_routes else "simple"
        risk_level = decision.risk_level if decision.risk_level in ("high", "low") else (
            "high" if route == "risky" else "low"
        )

        return {
            "route": route,
            "risk_level": risk_level,
            "messages": [f"classify:{route}"],
            "events": [
                make_event(
                    "classify",
                    "completed",
                    f"classified as {route}",
                    route=route,
                    risk_level=risk_level,
                    reasoning=decision.reasoning,
                )
            ],
        }

    except Exception as exc:  # noqa: BLE001
        # Controlled fallback: log error and default to error route for retry
        error_msg = f"LLM classification failed: {type(exc).__name__}: {exc}"
        return {
            "route": "error",
            "risk_level": "low",
            "errors": [error_msg],
            "messages": ["classify:fallback-error"],
            "events": [
                make_event(
                    "classify",
                    "failed",
                    "LLM classification failed; defaulting to error route",
                    fallback=True,
                    error=str(type(exc).__name__),
                )
            ],
        }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulates transient failures for error-route scenarios to test retry loops.
    - If route is "error" and attempt < 2: return error result
    - Otherwise: return a mock success result
    """
    route = state.get("route", "")
    attempt = state.get("attempt", 0)
    query = state.get("query", "")

    # Simulate transient failure for error route (first two attempts)
    if route == "error" and attempt < 2:
        result = f"ERROR: Transient failure executing tool for query '{query[:40]}' (attempt {attempt})"
        event_type = "failed"
        event_msg = "tool execution failed (transient error)"
    else:
        result = (
            f"Tool executed successfully for query '{query[:40]}'. "
            f"Result: [mock data retrieved at attempt {attempt}]"
        )
        event_type = "completed"
        event_msg = "tool executed successfully"

    return {
        "tool_results": [result],
        "events": [
            make_event(
                "tool",
                event_type,
                event_msg,
                attempt=attempt,
                route=route,
                success=(event_type == "completed"),
            )
        ],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Checks whether the latest tool result contains an ERROR substring.
    Returns evaluation_result: "needs_retry" or "success".
    """
    tool_results: list[str] = state.get("tool_results", []) or []

    if not tool_results:
        # No tool result to evaluate — treat as needs_retry
        return {
            "evaluation_result": "needs_retry",
            "events": [
                make_event(
                    "evaluate",
                    "needs_retry",
                    "no tool result found; scheduling retry",
                )
            ],
        }

    latest_result = tool_results[-1]

    # Heuristic: ERROR substring indicates failure
    if "ERROR" in latest_result.upper():
        verdict = "needs_retry"
        reason = "tool result contains ERROR indicator"
    else:
        verdict = "success"
        reason = "tool result looks successful"

    return {
        "evaluation_result": verdict,
        "events": [
            make_event(
                "evaluate",
                verdict,
                reason,
                verdict=verdict,
                latest_result_preview=latest_result[:80],
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    Grounded in: query, tool_results, approval decision (if risky route).
    """
    query = state.get("query", "")
    tool_results: list[str] = state.get("tool_results", []) or []
    approval: dict[str, Any] | None = state.get("approval")
    proposed_action: str | None = state.get("proposed_action")
    route = state.get("route", "")

    # Build context
    context_parts = [f"Original ticket: {query}"]

    if tool_results:
        context_parts.append("Tool results:\n" + "\n".join(f"- {r}" for r in tool_results))

    if route == "risky" and approval is not None:
        approved = approval.get("approved", False)
        comment = approval.get("comment", "")
        reviewer = approval.get("reviewer", "unknown")
        if approved:
            context_parts.append(
                f"Human approval: APPROVED by {reviewer}. "
                f"Proposed action was: {proposed_action or 'N/A'}. "
                f"Comment: {comment or 'none'}"
            )
        else:
            context_parts.append(
                f"Human approval: REJECTED by {reviewer}. "
                f"Comment: {comment or 'none'}. "
                "Do NOT claim the action was completed."
            )

    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful customer support agent. "
        "Answer the support ticket based ONLY on the provided context. "
        "Be concise, professional, and helpful. "
        "If context is limited, clearly state what you know and what requires further action. "
        "Never invent information not present in the context."
    )

    user_prompt = f"{context}\n\nProvide a helpful response to the customer."

    try:
        llm = get_llm()
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        final_answer = response.content if hasattr(response, "content") else str(response)

        return {
            "final_answer": final_answer,
            "events": [
                make_event(
                    "answer",
                    "completed",
                    "grounded answer generated",
                    route=route,
                    has_tool_results=bool(tool_results),
                    has_approval=approval is not None,
                )
            ],
        }

    except Exception as exc:  # noqa: BLE001
        error_msg = f"LLM answer generation failed: {type(exc).__name__}: {exc}"
        fallback_answer = (
            f"I was unable to generate a complete response due to a technical issue. "
            f"Your ticket regarding '{query[:60]}' has been received and will be reviewed."
        )
        return {
            "final_answer": fallback_answer,
            "errors": [error_msg],
            "events": [
                make_event(
                    "answer",
                    "failed",
                    "LLM answer generation failed; using fallback",
                    error=str(type(exc).__name__),
                )
            ],
        }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generates a specific, actionable clarification question.
    Also handles the rejected-approval path.
    """
    query = state.get("query", "")
    approval: dict[str, Any] | None = state.get("approval")
    proposed_action: str | None = state.get("proposed_action")

    # Determine context: rejected approval or missing info
    if approval is not None and not approval.get("approved", False):
        comment = approval.get("comment", "no comment")
        reviewer = approval.get("reviewer", "reviewer")
        question = (
            f"Your request to '{proposed_action or query[:60]}' was not approved "
            f"by {reviewer} (reason: {comment}). "
            "Could you clarify what alternative action you would like to take, "
            "or provide additional justification for the original request?"
        )
        reason = "approval rejected"
    else:
        # Missing info path: generate targeted question
        question = (
            f"To help with your request, I need more details. "
            f"Your message '{query[:80]}' is missing specific information. "
            "Could you please provide: (1) the specific issue or item involved, "
            "(2) any relevant account or order numbers, and "
            "(3) the exact outcome you expect?"
        )
        reason = "insufficient information in query"

    return {
        "pending_question": question,
        "final_answer": question,
        "events": [
            make_event(
                "clarify",
                "requested",
                "clarification requested",
                reason=reason,
                question_preview=question[:80],
            )
        ],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describes the proposed side-effect action. Does NOT execute anything.
    """
    query = state.get("query", "")
    risk_level = state.get("risk_level", "unknown")

    proposed_action = (
        f"Proposed action based on ticket: '{query[:120]}'. "
        f"Risk level: {risk_level}. "
        "This action involves potential side effects (e.g., data modification, "
        "financial transaction, account changes) and requires explicit human approval "
        "before execution."
    )

    return {
        "proposed_action": proposed_action,
        "events": [
            make_event(
                "risky_action",
                "proposed",
                "risky action prepared for approval",
                risk_level=risk_level,
                action_preview=proposed_action[:80],
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.
    """
    proposed_action = state.get("proposed_action", "")

    # Real HITL via interrupt (extension — disabled by default)
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        try:
            from langgraph.types import interrupt  # type: ignore[import]

            decision = interrupt(
                {
                    "question": "Approve this action?",
                    "proposed_action": proposed_action,
                }
            )
            approved = bool(decision.get("approved", False))
            reviewer = decision.get("reviewer", "human-reviewer")
            comment = decision.get("comment", "")
        except Exception:  # noqa: BLE001
            # If interrupt fails, fall through to mock approval
            approved = True
            reviewer = "mock-reviewer-fallback"
            comment = "interrupt failed; defaulted to mock approval"
    else:
        # Default: mock approval for CI/testing
        approved = True
        reviewer = "mock-reviewer"
        comment = "auto-approved for testing"

    approval = {
        "approved": approved,
        "reviewer": reviewer,
        "comment": comment,
    }

    return {
        "approval": approval,
        "events": [
            make_event(
                "approval",
                "observed",
                f"approval decision: {'approved' if approved else 'rejected'}",
                approved=approved,
                reviewer=reviewer,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increments attempt counter by exactly 1. Only this node may increment.
    """
    current_attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    new_attempt = current_attempt + 1

    error_msg = (
        f"Retry attempt {new_attempt}/{max_attempts}: "
        f"tool execution or evaluation failed at attempt {current_attempt}."
    )

    return {
        "attempt": new_attempt,
        "errors": [error_msg],
        "events": [
            make_event(
                "retry",
                "recorded",
                f"retry recorded (attempt {new_attempt} of {max_attempts})",
                new_attempt=new_attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    Sets final_answer to an escalation message and logs the dead-letter event.
    """
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    query = state.get("query", "")
    errors: list[str] = state.get("errors", []) or []
    latest_error = errors[-1] if errors else "unknown error"

    final_answer = (
        f"We were unable to complete your request: '{query[:80]}' "
        f"after {attempt} attempt(s) (limit: {max_attempts}). "
        "The issue has been escalated to our engineering team for manual review. "
        "Please reference this ticket for follow-up. We apologize for the inconvenience."
    )

    return {
        "final_answer": final_answer,
        "events": [
            make_event(
                "dead_letter",
                "exhausted",
                "max retries exceeded; ticket escalated",
                attempt=attempt,
                max_attempts=max_attempts,
                latest_error=latest_error[:120],
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
