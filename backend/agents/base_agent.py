from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging
from langchain.schema import BaseMessage
from langchain_core.language_models import BaseLanguageModel

logger = logging.getLogger(__name__)

class AgentState(BaseModel):
    """Shared state between agents"""
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

class BaseAgent(ABC):
    """Base class for all market research agents"""
    
    def __init__(self, llm: BaseLanguageModel, agent_name: str):
        self.llm = llm
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")
        
    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """Execute the agent's primary function"""
        pass
    
    def log_action(self, action: str, details: Dict[str, Any] = None):
        """Log agent actions for audit trail"""
        log_entry = {
            "agent": self.agent_name,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.logger.info(f"Agent {self.agent_name}: {action}", extra=log_entry)
    
    def validate_input(self, state: AgentState) -> bool:
        """Validate input state before processing"""
        if not state.research_query:
            self.logger.error("Research query is required")
            return False
        return True
    
    def update_state(self, state: AgentState, updates: Dict[str, Any]) -> AgentState:
        """Update agent state with new information"""
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
        state.updated_at = datetime.now()
        return state
    
    async def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate LLM response with context"""
        try:
            if context:
                prompt = f"Context: {context}\n\nQuery: {prompt}"
            
            response = await self.llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            return f"Error generating response: {str(e)}"
