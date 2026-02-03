# utils/agent_utils.py

from params.params import AgentEnum


def get_agent(agent: AgentEnum):
    if agent == AgentEnum.MADE:
        from core.made import MADE

        return MADE

    if agent == AgentEnum.SINGLE_ONESHOT:
        from core.single_oneshot import SingleOneShot

        return SingleOneShot

    if agent == AgentEnum.SINGLE_STEP:
        from core.single_step import SingleStep

        return SingleStep

    raise ValueError(f"Unknown agent: {agent}")
