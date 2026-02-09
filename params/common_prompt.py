# params/common_prompt.py

from params.prompt_params import StageEnum, PromptPack

class CommonPrompt:
    STRICT_CONSTRAINTS = [
        "1) STRICT LABELING: For final JSON output fields (WorkEnvironment, HazardFactor, PPEItem), use ONLY the allowed labels listed below.",
        "2) OBSERVATION vs. REQUIREMENT LOGIC:",
        "   - OBSERVATION (WorkEnv, Hazard, Wearing): Rely STRICTLY on visual evidence. Do not hallucinate hazards or worn items not in the image.",
        "   - REQUIREMENT (Compliance): Determine 'Required PPE' based on the safety needs of the identified hazard. "
        "If a hazard exists, you MUST list the necessary PPE as required, EVEN IF the worker is not wearing it.",
        "3) SCHEMA PROPOSALS: Proposers and Judges can suggest new labels or mappings, but these MUST be placed in the justification text or the 'proposals' list.",
        "4) NO HALLUCINATION: If a body part (e.g., hands, feet) is not visible in the image, do NOT assume or invent PPE status.",
        "5) UPPER BOUND & NECESSITY: The defined mappings are the MAXIMUM CANDIDATE LIST (Upper Bound). Do NOT blind-copy all mapped items."
    ]

    VISIBILITY_RULE = "- VISIBILITY RULE: If hands or feet are out of frame or obscured, state 'not visible' in reasoning and do NOT set worn=true based on a guess."

    SCHEMA_PROPOSAL_RULE = (
        "- SCHEMA PROPOSAL: If you see a new WorkEnvironment, Hazard, or PPE that fits this image but is NOT in the 'Allowed labels', "
        "you MUST describe it in your reasoning. Suggest it as a new label or a new mapping rule.\n"
        "- FORMAT: Use only current allowed labels for the JSON prediction, but advocate for changes in your text."
    )


    STAGE_GOALS = {
        StageEnum.WORK_ENVIRONMENT: "Choose 1 WorkEnvironment key.",
        StageEnum.HAZARD: "List all applicable HazardFactor keys.",
        StageEnum.COMPLIANCE: "Decide the FINAL required PPE list (PPEItem keys).",
        StageEnum.WEARING: "Decide worn=true/false ONLY for PPE visible on body parts shown in the image.",
        StageEnum.IMPROPER_WEARING: "For visible and worn=true items, decide if they are worn incorrectly."
    }

    PROPOSAL_POLICY_TEXT = (
        "Proposal policy (for label-set / mapping-table changes):\n"
        "\n"
        "- You must output proposals as a list field named: proposals\n"
        "- If no changes are needed, set proposals to an empty list: []\n"
        "\n"
        "Each item in proposals must match ProposalOut:\n"
        "- flag: must be true for every proposal item in the list\n"
        "- kind: choose EXACTLY ONE of the following: label | mapping\n"
        "- subject depends on kind:\n"
        "  - kind=label: work_environment | hazard | ppe\n"
        "  - kind=mapping: we_to_hazard | we_hazard_to_ppe\n"
        "- type: choose EXACTLY ONE of the following: add | remove | modify\n"
        "- target_from / target_to rules:\n"
        "  - add: target_from=\"\" and target_to=\"NEW_KEY\"\n"
        "  - remove: target_from=\"EXISTING_KEY\" and target_to=\"\"\n"
        "  - modify: target_from=\"EXISTING_KEY\" and target_to=\"NEW_KEY\"\n"
        "- scope rules (only for kind=mapping):\n"
        "  - subject=we_to_hazard: scope.work_environment set, scope.hazard=\"\"\n"
        "  - subject=we_hazard_to_ppe: scope.work_environment set, scope.hazard set\n"
        "- proposal: short, evidence-based reason\n"
        "- IMPORTANT: If you add a new label, also add a connecting mapping proposal."
    )

    SYSTEM_PROMPT = (
        "You are an industrial safety labeling assistant for Personal Protective Equipment (PPE) compliance.\n"
        "You will be given an image and a state object from previous stages.\n"
        "Follow constraints and role instructions strictly.\n"
        "Keep reasons short and evidence-based.\n"
    )

