# Day 08 Lab Report

## 1. Team / student

- Name: 2A202600635 Lê Dương Hiếu
- Repo/commit: Head (Local Lab Repository)
- Date: 2026-06-29

## 2. Architecture

The system utilizes a structured **LangGraph StateGraph** to process support tickets. The workflow graph comprises 11 distinct nodes and conditional routing logic:

- **intake**: Normalizes and pre-processes input queries.
- **classify**: Leverages LLM structured output to categorize intent into one of five routes: `simple`, `tool`, `missing_info`, `risky`, or `error`.
- **tool**: Executes backend actions, including simulating transient timeouts/failures for testing robustness.
- **evaluate**: Evaluates tool output quality (using LLM-as-judge with regex fallback).
- **answer**: LLM generates the final response grounded in user query, tool results, and approvals.
- **clarify**: Requests missing/vague information from the user instead of hallucinating.
- **risky_action**: Prepares sensitive operations (refunds, deletions) for approval.
- **approval**: Represents a human-in-the-loop checkpoint. In production mode (`LANGGRAPH_INTERRUPT=true`), this yields execution via `interrupt()` and awaits resumption.
- **retry**: Increments retry counts and logs errors during transient failures.
- **dead_letter**: Gracefully terminates execution when maximum retries are exhausted.
- **finalize**: Performs audit logging before finishing.

## 3. State schema

The state is managed using `AgentState`, a typed dictionary defining the current context:

| Field             | Reducer   | Why                                                            |
| ----------------- | --------- | -------------------------------------------------------------- |
| thread_id         | overwrite | Identifies the execution flow session.                         |
| scenario_id       | overwrite | Identifies the grading scenario ID.                            |
| query             | overwrite | The original ticket query.                                     |
| route             | overwrite | The current route target classified by the LLM.                |
| risk_level        | overwrite | The risk level ('high' or 'low').                              |
| attempt           | overwrite | The current retry attempt counter.                             |
| max_attempts      | overwrite | The maximum permitted retry attempts.                          |
| final_answer      | overwrite | Holds the final customer-facing output message.                |
| evaluation_result | overwrite | Latest tool execution evaluation ('success' or 'needs_retry'). |
| pending_question  | overwrite | Latest clarification request question.                         |
| proposed_action   | overwrite | Description of the risky operation needing verification.       |
| approval          | overwrite | Record of the human approval decision.                         |
| messages          | append    | audit conversation/events                                      |
| tool_results      | append    | Accumulates results of tool invocations.                       |
| errors            | append    | Log of errors encountered.                                     |
| events            | append    | audit conversation/events                                      |

## 4. Scenario results

**Summary Metrics:**

- **Total Scenarios:** 7
- **Success Rate:** 100.00%
- **Average Nodes Visited:** 6.4
- **Total Retries:** 3
- **Total Interrupts:** 2

| Scenario        | Expected route | Actual route | Success | Retries | Interrupts |
| --------------- | -------------- | ------------ | ------: | ------: | ---------: |
| S01_simple      | simple         | simple       |     Yes |       0 |          0 |
| S02_tool        | tool           | tool         |     Yes |       0 |          0 |
| S03_missing     | missing_info   | missing_info |     Yes |       0 |          0 |
| S04_risky       | risky          | risky        |     Yes |       0 |          1 |
| S05_error       | error          | error        |     Yes |       2 |          0 |
| S06_delete      | risky          | risky        |     Yes |       0 |          1 |
| S07_dead_letter | error          | error        |     Yes |       1 |          0 |

## 5. Failure analysis

1. **Retry or tool failure (S05 & S07):** Transient tool errors are routed to the `retry` node, incrementing the attempt counter. S05 recovers on the second retry (within the `max_attempts=3` limit) and successfully transitions to `answer`. S07 exhausts retries immediately (`max_attempts=1`) and is safely routed to the `dead_letter` node, ensuring bounded loops.
2. **Risky action without approval (S04 & S06):** Destructive actions (deletions/refunds) are classified as `risky` and require approval. The workflow routes them to the `approval` node. If `LANGGRAPH_INTERRUPT` is active, it enforces a hard gate by triggering an `interrupt()` which requires manual checkpointer resumption.

## 6. Persistence / recovery evidence

We implemented a SQLite-backed checkpointer (`SqliteSaver`). By supplying a unique `thread_id` in the graph execution configuration, the checkpointer persists state checkpoints across executions. If execution is interrupted by `interrupt()`, the state is saved to the SQLite DB (`outputs/checkpoints.db`) under WAL mode and can be successfully resumed when the user updates the thread with the approval decision.

## 7. Extension work

- **SQLite Checkpointer (`SqliteSaver`):** Completed SQLite-backed persistence layer in `persistence.py` with multi-threading compatibility (`check_same_thread=False`) and WAL journaling enabled.
- **LLM-as-Judge Evaluator:** Implemented an LLM-as-judge inside `evaluate_node` to assess tool result quality, with automatic substring checking fallback in case API keys are omitted.
- **Robust Local Run Compatibility:** Standardized LLM invocations with keyword/heuristic fallbacks, allowing the test suite and scenario execution script to run and validate successfully even without set API keys.

## 8. Improvement plan

If we had one more day, we would:

1. Productionize the real-time UI by building a Streamlit dashboard showing active tickets and permitting reviewers to review, comment, and click Approve/Reject directly.
2. Add comprehensive distributed tracing using langsmith to observe agent execution latency and costs.
