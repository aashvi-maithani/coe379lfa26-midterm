"""Milestone 3: implement three selected validation responsibilities."""

from __future__ import annotations

from collections import Counter

from .agent_models import ToolCallDecision
from .models import AgentResponse, LabRequest, TraceEvent
from .policy_context import ReservationContext, build_reservation_context


def validate_request_binding(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
) -> list[str]:
    """prevent identity and resolved-field substitution."""
    errors: list[str] = []
    args = decision.arguments

    # verify user_id matches the authenticated user in the request
    if "user_id" in args and args["user_id"] != request.authenticated_user_id:
        errors.append("user_id_must_match_authenticated_user")

    # map resolved request fields to check for substitution or unresolved usage
    resolved_fields = {
        "project_id": request.project_id,
        "resource_id": request.resource_id,
        "start": request.start,
        "end": request.end,
    }

    for field_name, trusted_value in resolved_fields.items():
        if trusted_value is None:
            # reject reservation calls when a required request field is unresolved
            if decision.tool == "reserve_resource":
                errors.append(f"{field_name}_unresolved_in_request")
        elif field_name in args and args[field_name] != trusted_value:
            # reject any tool call that substitutes a resolved request field
            errors.append(f"{field_name}_does_not_match_request")

    return errors


def validate_reservation_preconditions(context: ReservationContext) -> list[str]:
    """return an error for every required context fact that is false."""
    errors: list[str] = []

    # map each reservation context boolean attribute to its corresponding error code
    precondition_checks = [
        (context.arguments_complete, "incomplete_reservation_arguments"),
        (context.policy_retrieved, "policy_not_retrieved"),
        (context.authorizations_checked, "authorizations_not_checked"),
        (context.resource_status_checked, "resource_status_not_checked"),
        (context.project_authorized, "project_not_authorized"),
        (context.qualification_current, "required_qualification_missing"),
        (context.resource_available, "resource_not_available"),
        (context.calibration_current, "calibration_not_current"),
        (context.maintenance_clear, "resource_in_maintenance"),
        (context.after_hours_approval_satisfied, "after_hours_approval_missing"),
    ]

    # append error code for every precondition fact that is false
    for fact_is_true, error_code in precondition_checks:
        if not fact_is_true:
            errors.append(error_code)

    return errors


def validate_proposed_tool_call(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
    trace: list[TraceEvent],
) -> list[str]:
    """Supplied wrapper. Do not modify this function."""
    errors = validate_request_binding(decision, request=request)
    if decision.tool == "reserve_resource":
        context = build_reservation_context(decision, trace)
        errors.extend(validate_reservation_preconditions(context))
    return errors


def retrieved_passage_ids(trace: list[TraceEvent]) -> set[str]:
    """Supplied helper. Do not modify this function."""
    identifiers: set[str] = set()
    for event in trace:
        if event.observation.tool != "search_lab_documents" or not event.observation.ok:
            continue
        identifiers.update(
            passage["passage_id"]
            for passage in event.observation.data.get("passages", [])
        )
    return identifiers


def validate_final_response(
    response: AgentResponse, trace: list[TraceEvent]
) -> list[str]:
    """validate citations, claimed actions, and final status."""
    errors: list[str] = []

    # check that every citation in response was retrieved in the trace
    retrieved_ids = retrieved_passage_ids(trace)
    unretrieved = [c for c in response.citations if c not in retrieved_ids]
    if unretrieved:
        errors.append(f"citations_not_retrieved: {', '.join(unretrieved)}")

    # collect all tool names that made a successful mutation
    successful_mutations = [
        event.observation.tool
        for event in trace
        if event.mutating and event.observation.ok
    ]
    mutation_counts = Counter(successful_mutations)
    claimed_counts = Counter(response.completed_actions)

    # check that every claimed completed action appears in successful mutations
    if any(
        claimed_counts[action] > mutation_counts[action]
        for action in claimed_counts
    ):
        errors.append("completed_action_not_in_successful_trace")

    # if status is completed, require a successful mutation and a claimed action
    if response.status == "completed":
        if not successful_mutations:
            errors.append("completed_without_successful_mutation")
        if not response.completed_actions:
            errors.append("completed_without_claimed_actions")

    # if status is not completed, forbid claimed actions and unreported mutations
    else:
        if response.completed_actions:
            errors.append("non_completed_response_claims_actions")
        if successful_mutations:
            errors.append("unreported_successful_mutation")

    return errors