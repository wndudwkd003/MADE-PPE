# utils/analysis_utils.py


from typing import Any

def get_proposal(final_state: dict[str, Any]):
    proposals = []
    for key, value in final_state.items():
        if not key.startswith("proposal_"):
            continue

        if value["flag"] is not True:
            continue

        proposals.append((key, value))

    return proposals
