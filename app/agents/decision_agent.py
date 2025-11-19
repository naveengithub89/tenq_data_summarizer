from __future__ import annotations

from pydantic_ai import Agent, RunContext

from app.agents.dependencies import AgentDependencies
from app.config.settings import get_settings
from app.agents.models import DecisionOutput, TenQInsights

settings = get_settings()

decision_agent = Agent(
    settings.llm_model, 
    deps_type=AgentDependencies,
    output_type=DecisionOutput,
    instructions=(
        "You are an equity analyst asked to provide a high-level Buy/Sell/Hold style "
        "view purely from the latest 10-Q. "
        "Be conservative, highlight uncertainties, and clearly explain that this is "
        "not investment advice. Prefer HOLD when information is incomplete or ambiguous."
    ),
)


@decision_agent.tool
def pass_insights(
    ctx: RunContext[AgentDependencies],
    insights: TenQInsights,
) -> TenQInsights:
    """
    No-op tool that just passes the structured insights into the model context.
    """
    return insights
