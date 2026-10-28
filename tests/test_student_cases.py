"""Milestone 4: student-authored deterministic test cases."""

from __future__ import annotations

from atlas_agent.agent import run_agent
from atlas_agent.llm import ScriptedLLM
from atlas_agent.models import LabRequest


def _tool_call(tool: str, arguments: dict) -> dict:
    # build a scripted tool call decision dictionary
    return {
        "kind": "tool_call",
        "tool": tool,
        "arguments": arguments,
        "reasoning_summary": "Execute step in test script.",
    }


def test_unsafe_action_proposal_denied_before_execution(environment) -> None:
    # create request bound to user alice and project polymer-study
    request = LabRequest(
        authenticated_user_id="alice",
        text="Reserve tensile-01 for polymer-study tomorrow morning.",
        project_id="polymer-study",
        resource_id="tensile-01",
        start="2026-10-15T10:00:00",
        end="2026-10-15T11:00:00",
    )

    # script model to propose reserve_resource with a substituted user_id and no prior checks
    decisions = [
        _tool_call(
            "reserve_resource",
            {
                "user_id": "bob",
                "project_id": "polymer-study",
                "resource_id": "tensile-01",
                "start": "2026-10-15T10:00:00",
                "end": "2026-10-15T11:00:00",
            },
        ),
        {
            "kind": "final",
            "response": {
                "status": "blocked",
                "message": "Reservation blocked by safety gate.",
                "citations": [],
                "completed_actions": [],
            },
        },
    ]

    # record initial reservation count before running agent
    initial_reservations = len(environment.simulator.state["reservations"])
    run = run_agent(request, llm=ScriptedLLM(decisions), environment=environment)

    # verify action gate denied the call and simulator state was not mutated
    assert len(run.trace) == 1
    assert run.trace[0].observation.ok is False
    assert run.trace[0].observation.error_code == "action_gate_denied"
    assert "user_id_must_match_authenticated_user" in (run.trace[0].observation.message or "")
    assert "policy_not_retrieved" in (run.trace[0].observation.message or "")
    assert len(environment.simulator.state["reservations"]) == initial_reservations
    assert run.response.status == "blocked"


def test_contradictory_final_response_rejected_against_trace(environment) -> None:
    # create standard resolved lab request
    request = LabRequest(
        authenticated_user_id="alice",
        text="Reserve tensile-01 for polymer-study.",
        project_id="polymer-study",
        resource_id="tensile-01",
        start="2026-10-15T10:00:00",
        end="2026-10-15T11:00:00",
    )

    # script model to falsely claim a completed reservation and fabricated citation without executing tools
    decisions = [
        {
            "kind": "final",
            "response": {
                "status": "completed",
                "message": "Reservation confirmed.",
                "citations": ["FAKE-999#unretrieved"],
                "completed_actions": ["reserve_resource"],
            },
        }
    ]

    run = run_agent(request, llm=ScriptedLLM(decisions), environment=environment)

    # verify final response validator rejects the contradictory response and marks run as failed
    assert len(run.trace) == 0
    assert run.response.status == "failed"
    assert run.termination_reason == "invalid_final_response"
    assert "citations_not_retrieved" in run.response.message
    assert "completed_without_successful_mutation" in run.response.message


def test_unreported_successful_mutation_rejected_on_blocked_response(environment) -> None:
    # create valid request where all preconditions will succeed
    interval = {"start": "2026-10-15T10:00:00", "end": "2026-10-15T11:00:00"}
    request = LabRequest(
        authenticated_user_id="alice",
        text="Reserve tensile-01 for polymer-study from 10:00 to 11:00.",
        project_id="polymer-study",
        resource_id="tensile-01",
        **interval,
    )

    # script model to perform a valid reservation mutation but then claim status is blocked
    decisions = [
        _tool_call(
            "search_lab_documents",
            {"query": "tensile tester lab_safety authorization calibration", "top_k": 5},
        ),
        _tool_call("get_user_authorizations", {"user_id": "alice"}),
        _tool_call("get_resource_status", {"resource_id": "tensile-01", **interval}),
        _tool_call(
            "reserve_resource",
            {
                "user_id": "alice",
                "project_id": "polymer-study",
                "resource_id": "tensile-01",
                **interval,
            },
        ),
        {
            "kind": "final",
            "response": {
                "status": "blocked",
                "message": "Could not complete the reservation.",
                "citations": [],
                "completed_actions": [],
            },
        },
    ]

    run = run_agent(request, llm=ScriptedLLM(decisions), environment=environment)

    # verify validator catches the unreported successful mutation in the trace
    assert len(run.trace) == 4
    assert run.trace[-1].mutating is True
    assert run.trace[-1].observation.ok is True
    assert run.response.status == "failed"
    assert run.termination_reason == "invalid_final_response"
    assert "unreported_successful_mutation" in run.response.message