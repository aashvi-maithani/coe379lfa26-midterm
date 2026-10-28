"""Milestone 2: complete the bounded decision–action–observation loop."""

from __future__ import annotations

import json

from pydantic import ValidationError

from .agent_models import (
    ClarificationDecision,
    DecisionAdapter,
    FinalDecision,
    ToolCallDecision,
)
from .llm import StructuredLLM
from .models import AgentResponse, AgentRun, LabRequest, ToolObservation, TraceEvent
from .prompts import initialize_messages
from .tools import MUTATING_TOOLS, ToolEnvironment
from .validation import validate_final_response, validate_proposed_tool_call


def _failed(message: str, trace: list[TraceEvent], reason: str) -> AgentRun:
    # build a failed agent run with a given termination reason
    return AgentRun(
        response=AgentResponse(status="failed", message=message),
        trace=trace,
        termination_reason=reason,
    )


def run_agent(
    request: LabRequest,
    *,
    llm: StructuredLLM,
    environment: ToolEnvironment,
    max_steps: int = 10,
) -> AgentRun:
    # build initial system and user messages from the request
    messages = initialize_messages(request)
    trace: list[TraceEvent] = []

    for step in range(1, max_steps + 1):
        try:
            # ask llm for one decision and ensure validation via decisionadapter
            raw_decision = llm.generate(
                messages=messages, response_model=DecisionAdapter
            )
            decision = DecisionAdapter.validate_python(raw_decision)
        except (ValidationError, ValueError, TypeError, RuntimeError) as exc:
            return _failed(
                f"The model did not produce a valid decision: {exc}",
                trace,
                "invalid_model_decision",
            )

        if isinstance(decision, ClarificationDecision):
            # return clarification_required immediately without calling any tool
            return AgentRun(
                response=AgentResponse(
                    status="clarification_required",
                    message=decision.question,
                ),
                trace=trace,
                termination_reason="clarification_required",
            )

        if isinstance(decision, FinalDecision):
            # validate final response against the recorded trace
            errors = validate_final_response(decision.response, trace)
            if errors:
                # return failed run when final response disagrees with the trace
                return _failed(
                    f"Invalid final response: {'; '.join(errors)}",
                    trace,
                    "invalid_final_response",
                )
            # return valid final response
            return AgentRun(
                response=decision.response,
                trace=trace,
                termination_reason="model_finished",
            )

        if not isinstance(decision, ToolCallDecision):
            return _failed(
                "Unsupported decision type.", trace, "invalid_model_decision"
            )

        # validate proposed tool call before dispatching to environment
        gate_errors = validate_proposed_tool_call(
            decision, request=request, trace=trace
        )
        # block repeated mutating tool calls after a mutation has already succeeded
        if decision.tool in MUTATING_TOOLS and any(
            event.mutating and event.observation.ok for event in trace
        ):
            gate_errors.append("duplicate_mutating_call")

        if gate_errors:
            # record denied observation without executing the tool
            observation = ToolObservation(
                ok=False,
                tool=decision.tool,
                error_code="action_gate_denied",
                message="; ".join(gate_errors),
            )
        else:
            # execute permitted tool call in the environment
            observation = environment.dispatch(decision.tool, decision.arguments)

        # create and append trace event with mutating flag from mutating_tools
        event = TraceEvent(
            step=step,
            decision=decision.model_dump(),
            observation=observation,
            mutating=decision.tool in MUTATING_TOOLS,
        )
        trace.append(event)

        # append serialized decision as assistant message and observation as tool message
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(decision.model_dump()),
            }
        )
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(observation.model_dump()),
            }
        )

    # return explicit failure when maximum step budget is exhausted
    return _failed(
        "Maximum steps exceeded without reaching a final decision.",
        trace,
        "maximum_steps_exceeded",
    )
