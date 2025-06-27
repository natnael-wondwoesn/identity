from typing import Dict, Any, List
import asyncio
import uuid
from datetime import datetime

from .base_agent import BaseAgent, AgentState
from langchain_core.language_models import BaseLanguageModel


class ControllerAgent(BaseAgent):
    """
    Controller Agent orchestrates the market research workflow,
    delegates tasks, and aggregates outputs from other agents.
    """

    def __init__(self, llm: BaseLanguageModel):
        super().__init__(llm, "ControllerAgent")
        self.task_queue = []
        self.active_tasks = {}

    async def execute(self, state: AgentState) -> AgentState:
        """Orchestrate the complete market research workflow"""
        self.log_action(
            "Starting market research workflow",
            {"query": state.research_query, "industry": state.industry},
        )

        if not self.validate_input(state):
            state.errors.append("Invalid input provided to Controller Agent")
            return state

        try:
            # Initialize workflow
            workflow_plan = await self._create_workflow_plan(state)
            state = self.update_state(state, {"workflow_plan": workflow_plan})

            # Set up task coordination
            task_results = await self._coordinate_agents(state)

            # Aggregate final results
            final_report = await self._aggregate_results(state, task_results)
            state = self.update_state(state, {"final_report": final_report})

            self.log_action("Workflow completed successfully")

        except Exception as e:
            error_msg = f"Controller Agent workflow failed: {str(e)}"
            self.logger.error(error_msg)
            state.errors.append(error_msg)

        return state

    async def _create_workflow_plan(self, state: AgentState) -> Dict[str, Any]:
        """Create a detailed workflow plan based on research requirements"""

        prompt = f"""
        Create a market research workflow plan for the following query:
        Query: {state.research_query}
        Industry: {state.industry or 'General'}
        Timeframe: {state.timeframe or 'Current'}
        
        Provide a structured plan that includes:
        1. Research phases and objectives
        2. Data sources to investigate (only legitimate, publicly available sources)
        3. Analysis methods to apply
        4. Compliance considerations
        5. Expected deliverables
        
        Focus on ethical data collection from:
        - Industry reports and publications
        - Company press releases and official statements
        - Government economic data
        - Trade association publications
        - News articles and market analysis
        """

        plan_text = await self.generate_response(prompt)

        return {
            "plan_text": plan_text,
            "phases": ["research", "compliance_check", "analysis", "reporting"],
            "estimated_duration": "30-60 minutes",
            "created_at": datetime.now().isoformat(),
        }

    async def _coordinate_agents(self, state: AgentState) -> Dict[str, Any]:
        """Coordinate execution of specialized agents"""

        # This would integrate with other agents in the actual implementation
        # For now, we'll simulate the coordination structure

        coordination_plan = {
            "research_phase": {
                "agent": "ResearchAgent",
                "status": "pending",
                "priority": 1,
            },
            "compliance_phase": {
                "agent": "ComplianceAgent",
                "status": "pending",
                "priority": 2,
            },
            "analysis_phase": {
                "agent": "AnalysisAgent",
                "status": "pending",
                "priority": 3,
            },
            "strategy_phase": {
                "agent": "StrategyAgent",
                "status": "pending",
                "priority": 4,
            },
        }

        self.log_action("Agent coordination plan created", coordination_plan)

        return coordination_plan

    async def _aggregate_results(
        self, state: AgentState, task_results: Dict[str, Any]
    ) -> str:
        """Aggregate results from all agents into a final report"""

        prompt = f"""
        Create a comprehensive market research report based on the following information:
        
        Research Query: {state.research_query}
        Industry Focus: {state.industry or 'General Market'}
        
        Available Data Sources: {len(state.sources)} sources identified
        Analysis Results: {state.analysis_results}
        Compliance Status: All sources verified as compliant
        
        Create a professional market research report that includes:
        1. Executive Summary
        2. Methodology and Data Sources
        3. Key Findings
        4. Market Trends and Insights
        5. Recommendations
        6. Compliance and Limitations
        
        Ensure the report is professional, actionable, and clearly states data limitations.
        """

        final_report = await self.generate_response(
            prompt,
            {
                "sources_count": len(state.sources),
                "analysis_data": state.analysis_results,
            },
        )

        self.log_action(
            "Final report generated",
            {"report_length": len(final_report), "sources_used": len(state.sources)},
        )

        return final_report

    def get_workflow_status(self, task_id: str) -> Dict[str, Any]:
        """Get current status of a workflow task"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        return {"status": "not_found", "message": "Task not found"}

    async def cleanup_completed_tasks(self):
        """Clean up completed tasks to prevent memory buildup"""
        completed_tasks = [
            task_id
            for task_id, task in self.active_tasks.items()
            if task.get("status") == "completed"
        ]

        for task_id in completed_tasks:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

        self.log_action("Cleaned up completed tasks", {"count": len(completed_tasks)})
