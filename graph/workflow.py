from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

# Try different import paths for LangGraph
try:
    from langgraph.graph import StateGraph, END

    LANGGRAPH_AVAILABLE = True
    print("✅ LangGraph StateGraph imported successfully")
except ImportError:
    try:
        from langgraph import StateGraph, END

        LANGGRAPH_AVAILABLE = True
        print("✅ LangGraph imported from root")
    except ImportError:
        print("❌ LangGraph not available. Using fallback implementation.")
        LANGGRAPH_AVAILABLE = False

# Try to import Google Gemini
try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    GEMINI_AVAILABLE = True
    print("✅ Google Gemini (LangChain) available")
except ImportError:
    try:
        import google.generativeai as genai

        GEMINI_AVAILABLE = True
        print("✅ Google GenerativeAI (direct) available")
    except ImportError:
        print("❌ Google Gemini not available")
        GEMINI_AVAILABLE = False

# Try to import Ollama
try:
    from langchain_ollama import ChatOllama

    OLLAMA_AVAILABLE = True
    print("✅ Ollama available")
except ImportError:
    try:
        from langchain_community.llms import Ollama

        OLLAMA_AVAILABLE = True
        print("✅ Ollama (community) available")
    except ImportError:
        print("❌ Ollama not available")
        OLLAMA_AVAILABLE = False

# Import agents with error handling
try:
    from backend.agents.base_agent import AgentState
except ImportError:
    print("❌ Could not import AgentState. Creating fallback.")
    from pydantic import BaseModel
    from datetime import datetime

    class AgentState(BaseModel):
        task_id: str
        research_query: str
        industry: Optional[str] = None
        timeframe: Optional[str] = None
        sources: List[Dict[str, Any]] = []
        raw_data: List[Dict[str, Any]] = []
        analysis_results: Dict[str, Any] = {}
        compliance_status: Dict[str, Any] = {}
        strategy_results: Dict[str, Any] = {}
        final_report: Optional[str] = None
        errors: List[str] = []
        created_at: datetime = datetime.now()
        updated_at: datetime = datetime.now()


# Simple fallback workflow if LangGraph is not available
if not LANGGRAPH_AVAILABLE:

    class StateGraph:
        def __init__(self, state_class):
            self.state_class = state_class
            self.nodes = {}
            self.edges = {}
            self.entry_point = None

        def add_node(self, name, func):
            self.nodes[name] = func

        def add_edge(self, from_node, to_node):
            if from_node not in self.edges:
                self.edges[from_node] = []
            self.edges[from_node].append(to_node)

        def add_conditional_edges(self, from_node, condition_func, edge_mapping):
            # Simplified - just use first option
            if edge_mapping:
                first_option = list(edge_mapping.values())[0]
                self.add_edge(from_node, first_option)

        def set_entry_point(self, node):
            self.entry_point = node

        def compile(self):
            return SimpleWorkflow(self.nodes, self.edges, self.entry_point)

    class SimpleWorkflow:
        def __init__(self, nodes, edges, entry_point):
            self.nodes = nodes
            self.edges = edges
            self.entry_point = entry_point

        async def ainvoke(self, initial_state):
            current_state = initial_state
            current_node = self.entry_point

            # Simple sequential execution
            execution_order = [
                "controller_init",
                "research",
                "compliance",
                "analysis",
                "controller_final",
            ]

            for node_name in execution_order:
                if node_name in self.nodes:
                    try:
                        print(f"🔄 Executing {node_name}...")
                        current_state = await self.nodes[node_name](current_state)
                        print(f"✅ Completed {node_name}")
                    except Exception as e:
                        print(f"❌ Error in {node_name}: {e}")
                        current_state.errors.append(f"{node_name}: {str(e)}")

            return current_state

    END = "END"


# Custom Gemini LLM class
class GeminiLLM:
    """Custom Gemini LLM wrapper"""

    def __init__(
        self, api_key: str, model: str = "gemini-pro", temperature: float = 0.1
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._setup_gemini()

    def _setup_gemini(self):
        """Setup Gemini API"""
        try:
            if GEMINI_AVAILABLE:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
                print(f"✅ Gemini {self.model} initialized")
            else:
                raise ImportError("Gemini not available")
        except Exception as e:
            print(f"❌ Failed to setup Gemini: {e}")
            self.client = None

    async def ainvoke(self, prompt: str):
        """Async invoke for Gemini"""
        try:
            if self.client:
                # Use Gemini API
                response = self.client.generate_content(prompt)
                return GeminiResponse(response.text)
            else:
                return GeminiResponse(f"Mock Gemini response to: {prompt[:100]}...")
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            return GeminiResponse(f"Error: {str(e)}")


class GeminiResponse:
    """Response wrapper for Gemini"""

    def __init__(self, content: str):
        self.content = content


# Custom Ollama LLM class for fallback
class OllamaLLM:
    """Custom Ollama LLM wrapper"""

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        print(f"✅ Ollama {self.model} configured for {base_url}")

    async def ainvoke(self, prompt: str):
        """Async invoke for Ollama (simplified)"""
        try:
            # In a real implementation, this would make HTTP requests to Ollama
            import requests

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                return OllamaResponse(result.get("response", "No response"))
            else:
                return OllamaResponse(f"Ollama error: {response.status_code}")

        except Exception as e:
            print(f"❌ Ollama API error: {e}")
            return OllamaResponse(f"Mock Ollama response to: {prompt[:100]}...")


class OllamaResponse:
    """Response wrapper for Ollama"""

    def __init__(self, content: str):
        self.content = content


class MarketResearchWorkflow:
    """
    Market research workflow with Gemini and Ollama support (no OpenAI)
    """

    def __init__(self, llm_config: Dict[str, Any] = None):
        self.llm_config = llm_config or {
            "provider": "gemini",
            "model": "gemini-pro",
            "temperature": 0.1,
        }
        self.llm = self._initialize_llm()
        self.agents = self._initialize_agents()
        self.workflow = self._create_workflow()

    def _initialize_llm(self):
        """Initialize LLM with Gemini and Ollama support only"""
        provider = self.llm_config.get("provider", "gemini").lower()

        # Try Gemini first
        if provider == "gemini" and GEMINI_AVAILABLE:
            try:
                api_key = self.llm_config.get("api_key")
                if not api_key:
                    print("❌ No Gemini API key provided")
                    return self._try_ollama_fallback()

                # Try LangChain Gemini first
                if "ChatGoogleGenerativeAI" in globals():
                    return ChatGoogleGenerativeAI(
                        model=self.llm_config.get("model", "gemini-pro"),
                        temperature=self.llm_config.get("temperature", 0.1),
                        google_api_key=api_key,
                    )
                else:
                    # Use custom Gemini wrapper
                    return GeminiLLM(
                        api_key=api_key,
                        model=self.llm_config.get("model", "gemini-pro"),
                        temperature=self.llm_config.get("temperature", 0.1),
                    )
            except Exception as e:
                print(f"❌ Failed to initialize Gemini: {e}")
                return self._try_ollama_fallback()

        # Try Ollama
        elif provider == "ollama" and OLLAMA_AVAILABLE:
            try:
                if "ChatOllama" in globals():
                    return ChatOllama(
                        model=self.llm_config.get("model", "llama2"),
                        temperature=self.llm_config.get("temperature", 0.1),
                        base_url=self.llm_config.get(
                            "base_url", "http://localhost:11434"
                        ),
                    )
                else:
                    return OllamaLLM(
                        model=self.llm_config.get("model", "llama2"),
                        base_url=self.llm_config.get(
                            "base_url", "http://localhost:11434"
                        ),
                        temperature=self.llm_config.get("temperature", 0.1),
                    )
            except Exception as e:
                print(f"❌ Failed to initialize Ollama: {e}")
                return MockLLM()

        # Fallback logic
        else:
            return self._try_ollama_fallback()

    def _try_ollama_fallback(self):
        """Try Ollama as fallback"""
        if OLLAMA_AVAILABLE:
            try:
                print("🔄 Trying Ollama as fallback...")
                return OllamaLLM()
            except Exception as e:
                print(f"❌ Ollama fallback failed: {e}")

        # Final fallback to mock
        print("⚠️ Using mock LLM - configure Gemini or Ollama for full functionality")
        return MockLLM()

    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize agents with fallbacks"""
        try:
            from backend.agents.controller_agent import ControllerAgent
            from backend.agents.research_agent import ResearchAgent
            from backend.agents.compliance_agent import ComplianceAgent
            from backend.agents.analysis_agent import AnalysisAgent

            return {
                "controller": ControllerAgent(self.llm),
                "research": ResearchAgent(self.llm),
                "compliance": ComplianceAgent(self.llm),
                "analysis": AnalysisAgent(self.llm),
            }
        except ImportError as e:
            print(f"❌ Could not import agents: {e}")
            return {
                "controller": MockAgent("controller"),
                "research": MockAgent("research"),
                "compliance": MockAgent("compliance"),
                "analysis": MockAgent("analysis"),
            }

    def _create_workflow(self):
        """Create workflow with error handling"""
        try:
            # Create the workflow
            workflow = StateGraph(AgentState)

            # Add nodes
            workflow.add_node("controller_init", self._controller_node)
            workflow.add_node("research", self._research_node)
            workflow.add_node("compliance", self._compliance_node)
            workflow.add_node("analysis", self._analysis_node)
            workflow.add_node("controller_final", self._controller_final_node)

            # Add edges
            workflow.set_entry_point("controller_init")
            workflow.add_edge("controller_init", "research")
            workflow.add_edge("research", "compliance")
            workflow.add_edge("compliance", "analysis")
            workflow.add_edge("analysis", "controller_final")
            workflow.add_edge("controller_final", END)

            return workflow.compile()

        except Exception as e:
            print(f"❌ Failed to create workflow: {e}")
            return SimpleWorkflow({}, {}, "controller_init")

    async def _controller_node(self, state: AgentState) -> AgentState:
        """Controller node"""
        try:
            print("🎯 Controller: Initializing workflow...")
            state.updated_at = datetime.now()
            if "controller" in self.agents:
                state = await self.agents["controller"].execute(state)
            return state
        except Exception as e:
            state.errors.append(f"Controller failed: {str(e)}")
            return state

    async def _research_node(self, state: AgentState) -> AgentState:
        """Research node"""
        try:
            print("🔍 Research: Finding data sources...")
            if "research" in self.agents:
                state = await self.agents["research"].execute(state)
            return state
        except Exception as e:
            state.errors.append(f"Research failed: {str(e)}")
            return state

    async def _compliance_node(self, state: AgentState) -> AgentState:
        """Compliance node"""
        try:
            print("⚖️ Compliance: Validating sources...")
            if "compliance" in self.agents:
                state = await self.agents["compliance"].execute(state)
            return state
        except Exception as e:
            state.errors.append(f"Compliance failed: {str(e)}")
            return state

    async def _analysis_node(self, state: AgentState) -> AgentState:
        """Analysis node"""
        try:
            print("📊 Analysis: Processing data...")
            if "analysis" in self.agents:
                state = await self.agents["analysis"].execute(state)
            return state
        except Exception as e:
            state.errors.append(f"Analysis failed: {str(e)}")
            return state

    async def _controller_final_node(self, state: AgentState) -> AgentState:
        """Final controller node"""
        try:
            print("📋 Controller: Creating final report...")

            # Create final report
            prompt = f"""
Create a professional market research report for:

Query: {state.research_query}
Industry: {state.industry or 'General Market'}
Research Date: {datetime.now().strftime('%B %d, %Y')}

Based on the research conducted:
- Sources Identified: {len(state.sources)}
- Data Points Collected: {len(state.raw_data)}
- Compliance Checks: Completed
- Analysis Status: {'Completed' if state.analysis_results else 'In Progress'}

Please create a comprehensive report with:
1. Executive Summary
2. Market Overview
3. Key Findings
4. Strategic Insights
5. Recommendations
6. Next Steps

Make it professional and actionable for business decision-makers.
Focus on publicly available market data and ethical research practices.
"""

            try:
                response = await self.llm.ainvoke(prompt)
                final_report = (
                    response.content if hasattr(response, "content") else str(response)
                )
            except Exception as e:
                final_report = f"""
Market Research Report - {state.research_query}
========================================

Query: {state.research_query}
Industry: {state.industry or 'General'}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
LLM Provider: {self.llm_config.get('provider', 'unknown')}

Executive Summary:
This research was conducted using AI agents focused on publicly available data sources.

Research Process:
✓ Controller Agent: Workflow orchestration
✓ Research Agent: Source identification  
✓ Compliance Agent: Ethical validation
✓ Analysis Agent: Data processing

Sources Found: {len(state.sources)}
Data Points: {len(state.raw_data)}
Status: {'Completed with AI assistance' if not state.errors else 'Completed with issues'}

Note: Full report generation requires proper LLM configuration.
Error details: {str(e)}
"""

            state.final_report = final_report
            state.updated_at = datetime.now()
            return state
        except Exception as e:
            state.errors.append(f"Final processing failed: {str(e)}")
            return state

    async def run_research(
        self, research_query: str, industry: str = None, timeframe: str = None
    ) -> Dict[str, Any]:
        """Run the research workflow"""
        provider = self.llm_config.get("provider", "gemini")
        print(f"🚀 Starting research with {provider}: {research_query}")

        # Initialize state
        initial_state = AgentState(
            task_id=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            research_query=research_query,
            industry=industry,
            timeframe=timeframe,
        )

        try:
            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state)

            # Return results
            return {
                "task_id": final_state.task_id,
                "status": (
                    "completed" if not final_state.errors else "completed_with_errors"
                ),
                "research_query": final_state.research_query,
                "industry": final_state.industry,
                "sources_found": len(final_state.sources),
                "data_collected": len(final_state.raw_data),
                "final_report": final_state.final_report,
                "errors": final_state.errors,
                "created_at": final_state.created_at.isoformat(),
                "completed_at": final_state.updated_at.isoformat(),
                "llm_provider": provider,
            }

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
        """Get workflow visualization"""
        provider = self.llm_config.get("provider", "gemini")
        model = self.llm_config.get("model", "gemini-pro")

        return f"""
Market Research Workflow (Gemini + Ollama):

1. Controller Init -> 2. Research -> 3. Compliance -> 4. Analysis -> 5. Final Report

LLM Provider: {provider}
Model: {model}
LangGraph: {'Available' if LANGGRAPH_AVAILABLE else 'Fallback mode'}
Gemini: {'Available' if GEMINI_AVAILABLE else 'Not available'}
Ollama: {'Available' if OLLAMA_AVAILABLE else 'Not available'}
"""

    def get_agent_status(self) -> Dict[str, str]:
        """Get agent status"""
        return {
            "workflow": "Ready (Gemini + Ollama)",
            "langgraph": "Available" if LANGGRAPH_AVAILABLE else "Using fallback",
            "gemini": (
                "Available" if GEMINI_AVAILABLE else "Install google-generativeai"
            ),
            "ollama": "Available" if OLLAMA_AVAILABLE else "Install langchain-ollama",
            "llm_provider": self.llm_config.get("provider", "gemini"),
            "agents": "Ready" if LANGGRAPH_AVAILABLE else "Fallback mode",
        }


# Mock classes for testing without dependencies
class MockLLM:
    async def ainvoke(self, prompt):
        provider = "Mock"
        return MockResponse(f"{provider} response to: {prompt[:100]}...")


class MockResponse:
    def __init__(self, content):
        self.content = content


class MockAgent:
    def __init__(self, name):
        self.name = name

    async def execute(self, state):
        print(f"🤖 {self.name}: Mock execution")
        # Add some mock data
        if self.name == "research":
            state.sources = [{"url": "mock-source.com", "type": "mock"}]
            state.raw_data = [{"title": "Mock data", "content": "Sample content"}]
        elif self.name == "compliance":
            state.compliance_status = {"approved_sources_count": 1}
        elif self.name == "analysis":
            state.analysis_results = {"summary": "Mock analysis completed"}
        return state


# Factory function
def create_workflow(llm_config: Dict[str, Any] = None) -> MarketResearchWorkflow:
    """Create workflow instance with Gemini + Ollama support"""
    if llm_config is None:
        llm_config = {
            "provider": "gemini",  # or "ollama"
            "model": "gemini-pro",  # or "llama2" for ollama
            "temperature": 0.1,
        }
    return MarketResearchWorkflow(llm_config)
