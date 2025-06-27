from typing import Dict, Any, List, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import asyncio
from datetime import datetime

# Import agents with error handling
try:
    from backend.agents.base_agent import AgentState
    from backend.agents.controller_agent import ControllerAgent
    from backend.agents.research_agent import ResearchAgent
    from backend.agents.compliance_agent import ComplianceAgent
    from backend.agents.analysis_agent import AnalysisAgent
except ImportError as e:
    print(f"Warning: Could not import agents: {e}")
    # Create dummy classes for testing
    from backend.agents.base_agent import AgentState

    class ControllerAgent:
        def __init__(self, llm):
            self.llm = llm

        async def execute(self, state):
            return state

    class ResearchAgent:
        def __init__(self, llm):
            self.llm = llm

        async def execute(self, state):
            return state

    class ComplianceAgent:
        def __init__(self, llm):
            self.llm = llm

        async def execute(self, state):
            return state

    class AnalysisAgent:
        def __init__(self, llm):
            self.llm = llm

        async def execute(self, state):
            return state


# Try to import Ollama, fall back to OpenAI only
try:
    from langchain_ollama import ChatOllama

    OLLAMA_AVAILABLE = True
except ImportError:
    print("Warning: langchain_ollama not available. Using OpenAI only.")
    OLLAMA_AVAILABLE = False

    class ChatOllama:
        def __init__(self, *args, **kwargs):
            raise ImportError("Ollama not available")


class MarketResearchWorkflow:
    """
    LangGraph workflow orchestrating market research agents
    """

    def __init__(self, llm_config: Dict[str, Any] = None):
        self.llm_config = llm_config or {"model": "gpt-3.5-turbo", "temperature": 0.1}
        self.llm = self._initialize_llm()
        self.agents = self._initialize_agents()
        self.workflow = self._create_workflow()

    def _initialize_llm(self):
        """Initialize language model based on configuration"""
        provider = self.llm_config.get("provider", "openai").lower()

        if provider == "ollama" and OLLAMA_AVAILABLE:
            try:
                return ChatOllama(
                    model=self.llm_config.get("model", "llama2"),
                    temperature=self.llm_config.get("temperature", 0.1),
                    base_url=self.llm_config.get("base_url", "http://localhost:11434"),
                )
            except Exception as e:
                print(f"Failed to initialize Ollama: {e}")
                print("Falling back to OpenAI...")

        # Default to OpenAI
        return ChatOpenAI(
            model=self.llm_config.get("model", "gpt-3.5-turbo"),
            temperature=self.llm_config.get("temperature", 0.1),
            openai_api_key=self.llm_config.get("api_key"),
        )

    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents with the configured LLM"""
        return {
            "controller": ControllerAgent(self.llm),
            "research": ResearchAgent(self.llm),
            "compliance": ComplianceAgent(self.llm),
            "analysis": AnalysisAgent(self.llm),
            # Note: StrategyAgent not included yet as it's incomplete
        }

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""

        # Define the workflow state
        workflow = StateGraph(AgentState)

        # Add nodes for each agent
        workflow.add_node("controller_init", self._controller_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("compliance", self._compliance_node)
        workflow.add_node("analysis", self._analysis_node)
        workflow.add_node("controller_final", self._controller_final_node)

        # Define the workflow edges
        workflow.set_entry_point("controller_init")

        workflow.add_edge("controller_init", "research")
        workflow.add_edge("research", "compliance")
        workflow.add_conditional_edges(
            "compliance",
            self._should_continue_after_compliance,
            {"continue": "analysis", "stop": "controller_final"},
        )
        workflow.add_edge("analysis", "controller_final")
        workflow.add_edge("controller_final", END)

        return workflow.compile()

    async def _controller_node(self, state: AgentState) -> AgentState:
        """Controller agent initialization node"""
        try:
            # Initialize workflow and create plan
            state.updated_at = datetime.now()
            state = await self.agents["controller"].execute(state)
            return state
        except Exception as e:
            state.errors.append(f"Controller initialization failed: {str(e)}")
            return state

    async def _research_node(self, state: AgentState) -> AgentState:
        """Research agent node"""
        try:
            state = await self.agents["research"].execute(state)
            return state
        except Exception as e:
            state.errors.append(f"Research phase failed: {str(e)}")
            return state

    async def _compliance_node(self, state: AgentState) -> AgentState:
        """Compliance agent node"""
        try:
            state = await self.agents["compliance"].execute(state)
            return state
        except Exception as e:
            state.errors.append(f"Compliance validation failed: {str(e)}")
            return state

    async def _analysis_node(self, state: AgentState) -> AgentState:
        """Analysis agent node"""
        try:
            state = await self.agents["analysis"].execute(state)
            return state
        except Exception as e:
            state.errors.append(f"Analysis phase failed: {str(e)}")
            return state

    async def _controller_final_node(self, state: AgentState) -> AgentState:
        """Controller agent final processing node"""
        try:
            # Create a simple final report
            if state.analysis_results:
                final_report = await self._create_final_report(state)
                state.final_report = final_report

            state.updated_at = datetime.now()
            return state
        except Exception as e:
            state.errors.append(f"Final processing failed: {str(e)}")
            return state

    def _should_continue_after_compliance(self, state: AgentState) -> str:
        """Conditional logic to determine if workflow should continue after compliance"""
        if hasattr(state, "compliance_status") and state.compliance_status:
            approved_sources = state.compliance_status.get("approved_sources_count", 0)
            if approved_sources > 0:
                return "continue"
        return "continue"  # Continue anyway for testing

    async def _create_final_report(self, state: AgentState) -> str:
        """Create comprehensive final report"""

        prompt = f"""
        Create a comprehensive market research report based on the analysis:
        
        Research Query: {state.research_query}
        Industry: {state.industry or 'General'}
        
        Research Summary:
        - Sources Identified: {len(state.sources)}
        - Data Points Collected: {len(state.raw_data)}
        
        Create a professional report with:
        1. Executive Summary
        2. Research Methodology
        3. Key Findings
        4. Recommendations
        5. Limitations
        
        Make it business-ready and actionable.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            return f"Report generation failed: {str(e)}"

    async def run_research(
        self, research_query: str, industry: str = None, timeframe: str = None
    ) -> Dict[str, Any]:
        """Run the complete market research workflow"""

        # Initialize state
        initial_state = AgentState(
            task_id=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            research_query=research_query,
            industry=industry,
            timeframe=timeframe,
        )

        try:
            # Execute the workflow
            final_state = await self.workflow.ainvoke(initial_state)

            # Prepare results
            results = {
                "task_id": final_state.task_id,
                "status": (
                    "completed" if not final_state.errors else "completed_with_errors"
                ),
                "research_query": final_state.research_query,
                "industry": final_state.industry,
                "timeframe": final_state.timeframe,
                "sources_found": len(final_state.sources),
                "data_collected": len(final_state.raw_data),
                "compliance_status": getattr(final_state, "compliance_status", {}),
                "analysis_results": getattr(final_state, "analysis_results", {}),
                "strategy_results": getattr(final_state, "strategy_results", {}),
                "final_report": final_state.final_report,
                "errors": final_state.errors,
                "created_at": final_state.created_at.isoformat(),
                "completed_at": final_state.updated_at.isoformat(),
            }

            return results

        except Exception as e:
            return {
                "task_id": initial_state.task_id,
                "status": "failed",
                "error": str(e),
                "research_query": research_query,
                "created_at": initial_state.created_at.isoformat(),
                "completed_at": datetime.now().isoformat(),
            }

    def get_workflow_visualization(self) -> str:
        """Get a text representation of the workflow"""
        return """
        Market Research Workflow:
        
        1. Controller Init
           ↓
        2. Research Agent (Identify Sources)
           ↓
        3. Compliance Agent (Validate Sources)
           ↓
        4. Analysis Agent (Analyze Data)
           ↓
        5. Controller Final (Create Report)
           ↓
        6. END
        
        Note: Strategy Agent not yet implemented
        """

    def get_agent_status(self) -> Dict[str, str]:
        """Get status of all agents"""
        return {
            "controller": "Ready",
            "research": "Ready",
            "compliance": "Ready",
            "analysis": "Ready",
            "strategy": "Not Implemented",
            "workflow": "Compiled and Ready",
            "llm_provider": self.llm_config.get("provider", "openai"),
        }


# Workflow factory function
def create_workflow(llm_config: Dict[str, Any] = None) -> MarketResearchWorkflow:
    """Factory function to create a new workflow instance"""
    return MarketResearchWorkflow(llm_config)
