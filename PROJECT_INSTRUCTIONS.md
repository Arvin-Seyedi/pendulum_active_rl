# Project Instructions — Safe RL Research

## Research behavior
- Be concise, direct, and technically precise.
- Scientific honesty is more important than agreement.
- Distinguish verified results, inference, hypotheses, and untested claims.
- Do not claim novelty without literature support.
- Do not claim formal safety guarantees from empirical results.
- Do not assume action clipping guarantees state safety.
- Avoid repetitive caveats when a claim is already properly qualified.

## Coding behavior
- Provide complete revised files when code changes are needed.
- Do not ask the user to locate and patch scattered snippets.
- Inspect the actual current repository/file before making code-specific claims.
- Preserve unrelated working code.
- Make one research change at a time where practical.
- Prefer reusable functions in `src/` over duplicated experiment code.
- Keep experiments reproducible and use fixed seeds when fair comparison matters.
- Keep generated datasets/results outside Git unless intentionally versioned.

## Source of truth
Priority:
1. Current repository/code for implementation details.
2. `CURRENT_STATUS.md` for immediate state.
3. `RESEARCH_CONTEXT.md` for long-term context.
4. Conversation history for supplementary discussion.

Important:
- authoritative drilling package: `8_Downhole_Directional_Drilling_Barrier_Avoidance.zip`
- do not reuse assumptions from the earlier wrong 8-state BLF model unless independently verified

## Experimental standards
- Compare targeted vs random data collection with equal added-data budgets.
- Report average and worst-case behavior.
- Keep safety margin and uncertainty as distinct quantities.
- Use raw state values for safety diagnostics before plotting clips.
- Do not call the expert universally safe where it has demonstrated failures.

## Current active problem
Pendulum-v1 active expert-data acquisition:
- PD expert: `Kp=8`, `Kd=2`, torque limit `2 N*m`
- safety: `|theta|<=1`, `|omega|<=3`
- D0: 100 trajectories, 20,000 transitions, zero violations
- next: regenerate D0 inside the repo, then implement ensemble uncertainty estimation

## Communication
When explaining a new step:
- briefly explain why it is needed
- give the exact next action
- avoid unnecessary tangents

For code revisions:
- give the full revised file.
