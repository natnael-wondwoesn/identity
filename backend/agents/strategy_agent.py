from typing import Dict, Any, List
from datetime import datetime, timedelta
import json

from .base_agent import BaseAgent, AgentState
from langchain_core.language_models import BaseLanguageModel


class StrategyAgent(BaseAgent):
    """
    Strategy Agent synthesizes research findings and analysis results
    to develop comprehensive market research strategies and actionable recommendations.
    """

    def __init__(self, llm: BaseLanguageModel):
        super().__init__(llm, "StrategyAgent")
        self.strategy_frameworks = self._init_strategy_frameworks()

    def _init_strategy_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """Initialize strategic analysis frameworks"""
        return {
            "market_entry": {
                "components": [
                    "market_attractiveness",
                    "competitive_position",
                    "entry_barriers",
                    "timing",
                ],
                "applicable_to": ["new_market_research", "expansion_planning"],
                "deliverables": [
                    "entry_strategy",
                    "go_to_market_plan",
                    "risk_assessment",
                ],
            },
            "competitive_positioning": {
                "components": [
                    "competitive_analysis",
                    "differentiation",
                    "value_proposition",
                    "market_gaps",
                ],
                "applicable_to": ["product_strategy", "market_positioning"],
                "deliverables": [
                    "positioning_strategy",
                    "competitive_response",
                    "market_opportunities",
                ],
            },
            "growth_strategy": {
                "components": [
                    "market_trends",
                    "customer_insights",
                    "growth_vectors",
                    "investment_priorities",
                ],
                "applicable_to": ["business_planning", "investment_decisions"],
                "deliverables": [
                    "growth_roadmap",
                    "investment_recommendations",
                    "performance_metrics",
                ],
            },
            "risk_assessment": {
                "components": [
                    "market_risks",
                    "competitive_threats",
                    "regulatory_changes",
                    "economic_factors",
                ],
                "applicable_to": ["strategic_planning", "due_diligence"],
                "deliverables": [
                    "risk_matrix",
                    "mitigation_strategies",
                    "scenario_planning",
                ],
            },
        }

    async def execute(self, state: AgentState) -> AgentState:
        """Execute strategic analysis and generate comprehensive recommendations"""

        self.log_action(
            "Starting strategic analysis",
            {
                "research_query": state.research_query,
                "analysis_available": bool(state.analysis_results),
                "sources_count": len(state.sources),
            },
        )

        if not state.analysis_results:
            self.log_action("No analysis results available for strategy development")
            state.errors.append(
                "No analysis results available for strategy development"
            )
            return state

        try:
            # Determine appropriate strategy framework
            strategy_framework = await self._select_strategy_framework(state)

            # Synthesize insights from research and analysis
            synthesized_insights = await self._synthesize_insights(state)

            # Develop strategic recommendations
            strategic_recommendations = await self._develop_recommendations(
                state, synthesized_insights
            )

            # Create implementation roadmap
            implementation_plan = await self._create_implementation_plan(
                strategic_recommendations, state
            )

            # Generate comprehensive strategy document
            strategy_document = await self._generate_strategy_document(
                strategy_framework,
                synthesized_insights,
                strategic_recommendations,
                implementation_plan,
            )

            # Create executive summary
            executive_summary = await self._create_executive_summary(
                state, strategic_recommendations, implementation_plan
            )

            # Update state with strategy results
            updates = {
                "strategy_results": {
                    "framework_used": strategy_framework,
                    "synthesized_insights": synthesized_insights,
                    "recommendations": strategic_recommendations,
                    "implementation_plan": implementation_plan,
                    "strategy_document": strategy_document,
                    "executive_summary": executive_summary,
                    "strategy_timestamp": datetime.now().isoformat(),
                    "confidence_rating": self._calculate_strategy_confidence(state),
                }
            }

            state = self.update_state(state, updates)

            self.log_action(
                "Strategic analysis completed",
                {
                    "framework": strategy_framework["name"],
                    "recommendations_count": len(strategic_recommendations),
                    "confidence_rating": updates["strategy_results"][
                        "confidence_rating"
                    ],
                },
            )

        except Exception as e:
            error_msg = f"Strategy Agent failed: {str(e)}"
            self.logger.error(error_msg)
            state.errors.append(error_msg)

        return state

    async def _select_strategy_framework(self, state: AgentState) -> Dict[str, Any]:
        """Select the most appropriate strategy framework based on research context"""

        prompt = f"""
        Select the most appropriate strategic framework for this market research:
        
        Research Query: {state.research_query}
        Industry: {state.industry or 'General'}
        Available Analysis: {list(state.analysis_results.get('results', {}).keys()) if state.analysis_results else []}
        
        Available Frameworks:
        1. Market Entry - For new market opportunities and expansion planning
        2. Competitive Positioning - For competitive analysis and differentiation
        3. Growth Strategy - For business growth and investment decisions
        4. Risk Assessment - For risk analysis and scenario planning
        
        Consider:
        - The nature of the research query
        - Available data and analysis results
        - Strategic objectives implied by the query
        
        Recommend the most suitable framework and explain why.
        """

        framework_response = await self.generate_response(prompt)

        # Determine framework based on response (simplified selection)
        selected_framework = "growth_strategy"  # Default

        if "market entry" in framework_response.lower():
            selected_framework = "market_entry"
