from typing import Dict, Any, List
import asyncio
import requests
import feedparser
from datetime import datetime, timedelta
import re

from .base_agent import BaseAgent, AgentState
from langchain_core.language_models import BaseLanguageModel


class ResearchAgent(BaseAgent):
    """
    Research Agent identifies and evaluates legitimate public data sources
    for market research, focusing on industry reports, company data, and market trends.
    """

    def __init__(self, llm: BaseLanguageModel):
        super().__init__(llm, "ResearchAgent")
        self.legitimate_sources = self._init_legitimate_sources()

    def _init_legitimate_sources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize list of legitimate data sources for market research"""
        return {
            "government_data": {
                "sources": [
                    "https://www.census.gov/econ/",
                    "https://www.bls.gov/data/",
                    "https://fred.stlouisfed.org/",
                    "https://www.sec.gov/edgar.shtml",
                ],
                "type": "government",
                "compliance_level": "high",
                "description": "Official government economic and business data",
            },
            "industry_reports": {
                "sources": [
                    "https://www.statista.com/",
                    "https://www.ibisworld.com/",
                    "https://www.marketresearch.com/",
                ],
                "type": "research",
                "compliance_level": "high",
                "description": "Professional market research and industry analysis",
            },
            "news_feeds": {
                "sources": [
                    "https://feeds.reuters.com/reuters/businessNews",
                    "https://feeds.bloomberg.com/markets/news.rss",
                    "https://rss.cnn.com/rss/money_latest.rss",
                ],
                "type": "news",
                "compliance_level": "high",
                "description": "Business and market news from reputable sources",
            },
            "company_data": {
                "sources": ["press_releases", "annual_reports", "investor_relations"],
                "type": "corporate",
                "compliance_level": "high",
                "description": "Official company communications and filings",
            },
        }

    async def execute(self, state: AgentState) -> AgentState:
        """Execute research phase to identify and collect data from legitimate sources"""

        self.log_action(
            "Starting research phase",
            {"query": state.research_query, "industry": state.industry},
        )

        if not self.validate_input(state):
            state.errors.append("Invalid input for Research Agent")
            return state

        try:
            # Generate research strategy
            research_strategy = await self._generate_research_strategy(state)

            # Identify relevant sources
            relevant_sources = await self._identify_sources(state, research_strategy)

            # Collect sample data from sources
            collected_data = await self._collect_sample_data(relevant_sources)

            # Update state with findings
            updates = {
                "sources": relevant_sources,
                "raw_data": collected_data,
                "research_strategy": research_strategy,
            }

            state = self.update_state(state, updates)
            self.log_action(
                "Research phase completed",
                {
                    "sources_found": len(relevant_sources),
                    "data_points": len(collected_data),
                },
            )

        except Exception as e:
            error_msg = f"Research Agent failed: {str(e)}"
            self.logger.error(error_msg)
            state.errors.append(error_msg)

        return state

    async def _generate_research_strategy(self, state: AgentState) -> Dict[str, Any]:
        """Generate a focused research strategy based on the query"""

        prompt = f"""
        Create a research strategy for the following market research query:
        Query: {state.research_query}
        Industry: {state.industry or 'General'}
        
        Provide a structured research approach that identifies:
        1. Key research objectives
        2. Primary information needs
        3. Relevant data categories (financial, operational, market trends, etc.)
        4. Appropriate timeframes for data collection
        5. Success metrics for the research
        
        Focus only on publicly available, legitimate business information sources.
        Avoid any personal data collection or privacy-sensitive approaches.
        """

        strategy_text = await self.generate_response(prompt)

        return {
            "strategy_text": strategy_text,
            "objectives": self._extract_objectives(strategy_text),
            "data_categories": [
                "financial",
                "market_trends",
                "industry_analysis",
                "competitive_landscape",
            ],
            "timeframe": state.timeframe or "last_12_months",
        }

    def _extract_objectives(self, strategy_text: str) -> List[str]:
        """Extract key objectives from strategy text"""
        # Simple extraction - in production, would use more sophisticated NLP
        objectives = []
        lines = strategy_text.split("\n")
        for line in lines:
            if any(
                keyword in line.lower()
                for keyword in ["objective", "goal", "aim", "target"]
            ):
                clean_line = re.sub(r"^\d+\.?\s*", "", line.strip())
                if clean_line:
                    objectives.append(clean_line)
        return objectives[:5]  # Limit to top 5 objectives

    async def _identify_sources(
        self, state: AgentState, strategy: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify relevant sources based on research strategy"""

        relevant_sources = []

        for category, source_info in self.legitimate_sources.items():
            relevance_score = await self._score_source_relevance(
                state.research_query, state.industry, source_info
            )

            if relevance_score > 0.6:  # Threshold for relevance
                for source in source_info["sources"]:
                    source_entry = {
                        "url": source,
                        "category": category,
                        "type": source_info["type"],
                        "compliance_level": source_info["compliance_level"],
                        "relevance_score": relevance_score,
                        "description": source_info["description"],
                        "identified_at": datetime.now().isoformat(),
                    }
                    relevant_sources.append(source_entry)

        # Sort by relevance score
        relevant_sources.sort(key=lambda x: x["relevance_score"], reverse=True)

        self.log_action(
            "Sources identified",
            {
                "total_sources": len(relevant_sources),
                "high_relevance": len(
                    [s for s in relevant_sources if s["relevance_score"] > 0.8]
                ),
            },
        )

        return relevant_sources[:15]  # Limit to top 15 sources

    async def _score_source_relevance(
        self, query: str, industry: str, source_info: Dict
    ) -> float:
        """Score how relevant a source is to the research query"""

        prompt = f"""
        Rate the relevance of this data source for the given research query on a scale of 0.0 to 1.0:
        
        Research Query: {query}
        Industry: {industry or 'General'}
        
        Data Source: {source_info['description']}
        Source Type: {source_info['type']}
        
        Consider:
        - How well the source aligns with the research objectives
        - The reliability and authority of the source type
        - The likelihood of finding relevant data
        
        Respond with only a number between 0.0 and 1.0.
        """

        try:
            response = await self.generate_response(prompt)
            # Extract numeric score from response
            score_match = re.search(r"([0-1]\.?\d*)", response)
            if score_match:
                return min(float(score_match.group(1)), 1.0)
            return 0.5  # Default moderate relevance
        except:
            return 0.5

    async def _collect_sample_data(
        self, sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collect sample data from identified sources for validation"""

        collected_data = []

        for source in sources[:5]:  # Sample from top 5 sources
            try:
                if source["type"] == "news" and source["url"].endswith(".rss"):
                    # Handle RSS feeds
                    sample_data = await self._collect_rss_sample(source)
                elif source["type"] == "government":
                    # Handle government data sources
                    sample_data = await self._collect_government_sample(source)
                else:
                    # Generic data source handling
                    sample_data = await self._collect_generic_sample(source)

                if sample_data:
                    collected_data.extend(sample_data)

            except Exception as e:
                self.logger.warning(
                    f"Failed to collect sample from {source['url']}: {str(e)}"
                )
                continue

        self.log_action("Sample data collected", {"samples": len(collected_data)})
        return collected_data

    async def _collect_rss_sample(self, source: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect sample data from RSS feeds"""
        try:
            feed = feedparser.parse(source["url"])
            samples = []

            for entry in feed.entries[:5]:  # Top 5 entries
                sample = {
                    "source": source["url"],
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", "")[:200],  # Limit summary length
                    "published": entry.get("published", ""),
                    "link": entry.get("link", ""),
                    "collected_at": datetime.now().isoformat(),
                    "data_type": "news_article",
                }
                samples.append(sample)

            return samples
        except Exception as e:
            self.logger.error(f"RSS collection failed: {str(e)}")
            return []

    async def _collect_government_sample(
        self, source: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Collect sample data from government sources"""
        # In a real implementation, this would use specific APIs
        # For now, return structured sample data
        return [
            {
                "source": source["url"],
                "data_type": "government_economic",
                "sample_available": True,
                "api_access": "requires_key",
                "collected_at": datetime.now().isoformat(),
                "note": "Government data source identified - API integration needed",
            }
        ]

    async def _collect_generic_sample(
        self, source: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Collect sample from generic sources"""
        return [
            {
                "source": source["url"],
                "data_type": source["type"],
                "status": "source_identified",
                "requires_subscription": (
                    True
                    if "statista" in source["url"] or "ibisworld" in source["url"]
                    else False
                ),
                "collected_at": datetime.now().isoformat(),
            }
        ]
