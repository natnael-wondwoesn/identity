from typing import Dict, Any, List
import asyncio
import requests
import feedparser
from datetime import datetime, timedelta
import re

from .base_agent import BaseAgent, AgentState
from langchain_core.language_models import BaseLanguageModel

# Import crawl4ai for web scraping
try:
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
    from crawl4ai.extraction_strategy import LLMExtractionStrategy

    CRAWL4AI_AVAILABLE = True
    print("✅ Crawl4AI available")
except ImportError:
    CRAWL4AI_AVAILABLE = False
    print("❌ Crawl4AI not available - falling back to basic scraping")


class ResearchAgent(BaseAgent):
    """
    Hybrid Research Agent that combines web scraping with crawl4ai and traditional research.
    User provides a URL and query, the agent scrapes the site, extracts relevant data,
    and then performs research based on the scraped content.
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
        """Execute hybrid research: scrape target URL first, then research based on scraped data"""

        self.log_action(
            "Starting hybrid research phase",
            {
                "query": state.research_query,
                "target_url": getattr(state, "target_url", None),
                "industry": state.industry,
            },
        )

        if not self.validate_input(state):
            state.errors.append("Invalid input for Research Agent")
            return state

        try:
            scraped_data = None

            # First, scrape the target URL if provided
            if hasattr(state, "target_url") and state.target_url:
                scraped_data = await self._scrape_target_url(
                    state.target_url, state.research_query
                )

            # Generate research strategy based on scraped data and query
            research_strategy = await self._generate_research_strategy(
                state, scraped_data
            )

            # If we have scraped data, use it as primary source, otherwise use traditional sources
            if scraped_data:
                # Use scraped data as the primary source
                relevant_sources = [
                    {
                        "url": state.target_url,
                        "category": "scraped_content",
                        "type": "web_scraping",
                        "compliance_level": "medium",
                        "relevance_score": 1.0,
                        "description": f"Scraped content from {state.target_url}",
                        "identified_at": datetime.now().isoformat(),
                        "content": scraped_data,
                    }
                ]

                # Add traditional sources as supplementary
                traditional_sources = await self._identify_sources(
                    state, research_strategy
                )
                relevant_sources.extend(
                    traditional_sources[:5]
                )  # Limit traditional sources

                collected_data = [scraped_data]
                # Collect limited sample data from traditional sources
                traditional_sample_data = await self._collect_sample_data(
                    traditional_sources[:3]
                )
                collected_data.extend(traditional_sample_data)
            else:
                # Fall back to traditional research approach
                relevant_sources = await self._identify_sources(
                    state, research_strategy
                )
                collected_data = await self._collect_sample_data(relevant_sources)

            # Update state with findings
            updates = {
                "sources": relevant_sources,
                "raw_data": collected_data,
                "research_strategy": research_strategy,
                "scraped_content": scraped_data,
            }

            state = self.update_state(state, updates)
            self.log_action(
                "Hybrid research phase completed",
                {
                    "sources_found": len(relevant_sources),
                    "data_points": len(collected_data),
                    "scraped_url": getattr(state, "target_url", None),
                    "has_scraped_data": scraped_data is not None,
                },
            )

        except Exception as e:
            error_msg = f"Hybrid Research Agent failed: {str(e)}"
            self.logger.error(error_msg)
            state.errors.append(error_msg)

        return state

    async def _scrape_target_url(self, url: str, query: str) -> Dict[str, Any]:
        """Scrape the target URL using crawl4ai with intelligent extraction"""

        if not CRAWL4AI_AVAILABLE:
            self.log_action(
                "Crawl4AI not available, using fallback scraping", {"url": url}
            )
            return await self._fallback_scrape(url)

        try:
            self.log_action("Starting web scraping", {"url": url, "query": query})

            # Create extraction strategy based on the research query (if available)
            extraction_strategy = None
            try:
                extraction_strategy = LLMExtractionStrategy(
                    provider="openai",  # This can be configured
                    api_token="your-api-key",  # This should come from config
                    instruction=f"""
                    Extract relevant information from this webpage that relates to the research query: "{query}"
                    
                    Focus on:
                    1. Key business information, statistics, and data points
                    2. Market trends and insights
                    3. Company information, products, services
                    4. Financial data or performance metrics
                    5. Industry analysis or competitive information
                    
                    Structure the extracted data in a clear, organized format with:
                    - Main topics and themes
                    - Key statistics and numbers
                    - Important quotes or statements
                    - Contact information or company details
                    - Any relevant dates or timeframes
                    
                    Ignore navigation elements, ads, and irrelevant content.
                    """,
                )
            except Exception as e:
                self.log_action(
                    f"Could not create LLM extraction strategy: {str(e)}", {"url": url}
                )
                extraction_strategy = None

            # Configure the crawler
            crawler_config = {
                "headless": True,
                "verbose": True,
                "word_count_threshold": 10,
                "remove_overlay_elements": True,
                "simulate_user": True,
                "magic": True,  # Enable smart extraction
            }

            # Add extraction strategy only if available
            if extraction_strategy:
                crawler_config["extraction_strategy"] = extraction_strategy

            config = CrawlerRunConfig(**crawler_config)

            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url, config=config)

                if result.success:
                    extracted_content = (
                        result.extracted_content
                        if hasattr(result, "extracted_content")
                        else None
                    )

                    scraped_data = {
                        "url": url,
                        "title": result.metadata.get("title", ""),
                        "description": result.metadata.get("description", ""),
                        "content": result.cleaned_html or result.markdown,
                        "extracted_content": extracted_content,
                        "metadata": result.metadata,
                        "scraped_at": datetime.now().isoformat(),
                        "word_count": (
                            len(result.cleaned_html.split())
                            if result.cleaned_html
                            else 0
                        ),
                        "query_context": query,
                    }

                    self.log_action(
                        "Web scraping completed successfully",
                        {
                            "url": url,
                            "word_count": scraped_data["word_count"],
                            "has_extracted_content": extracted_content is not None,
                        },
                    )

                    return scraped_data
                else:
                    raise Exception(
                        f"Crawl4AI failed to scrape {url}: {result.error_message}"
                    )

        except Exception as e:
            self.log_action(f"Crawl4AI scraping failed: {str(e)}", {"url": url})
            # Fall back to basic scraping
            return await self._fallback_scrape(url)

    async def _fallback_scrape(self, url: str) -> Dict[str, Any]:
        """Fallback web scraping using requests and BeautifulSoup"""
        try:
            import requests
            from bs4 import BeautifulSoup

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text content
            text_content = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = " ".join(chunk for chunk in chunks if chunk)

            return {
                "url": url,
                "title": soup.title.string if soup.title else "",
                "content": text_content,
                "scraped_at": datetime.now().isoformat(),
                "word_count": len(text_content.split()),
                "method": "fallback_scraping",
            }

        except Exception as e:
            self.log_action(f"Fallback scraping failed: {str(e)}", {"url": url})
            return {
                "url": url,
                "error": str(e),
                "scraped_at": datetime.now().isoformat(),
                "method": "failed_scraping",
            }

    async def _generate_research_strategy(
        self, state: AgentState, scraped_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate a focused research strategy based on the query and scraped data"""

        # Include scraped data context in the prompt if available
        scraped_context = ""
        if scraped_data and not scraped_data.get("error"):
            scraped_context = f"""
            
        SCRAPED DATA CONTEXT:
        URL: {scraped_data.get('url', 'N/A')}
        Title: {scraped_data.get('title', 'N/A')}
        Content Summary: {scraped_data.get('content', '')[:500]}...
        
        Use this scraped data as the primary source and focus the research strategy on:
        1. Analyzing and expanding on the information found in the scraped content
        2. Finding complementary data that supports or challenges the scraped findings
        3. Identifying gaps in the scraped data that need additional research
        4. Verifying claims or data points found in the scraped content
        """

        prompt = f"""
        Create a research strategy for the following market research query:
        Query: {state.research_query}
        Industry: {state.industry or 'General'}
        Target URL: {getattr(state, 'target_url', 'None')}
        {scraped_context}
        
        Provide a structured research approach that identifies:
        1. Key research objectives (prioritizing analysis of scraped content if available)
        2. Primary information needs and data gaps
        3. Relevant data categories (financial, operational, market trends, etc.)
        4. Appropriate timeframes for data collection
        5. Success metrics for the research
        6. How to validate and cross-reference the scraped data
        
        Focus on publicly available, legitimate business information sources.
        Avoid any personal data collection or privacy-sensitive approaches.
        """

        strategy_text = await self.generate_response(prompt)

        return {
            "strategy_text": strategy_text,
            "objectives": self._extract_objectives(strategy_text),
            "data_categories": [
                "scraped_content_analysis",
                "financial",
                "market_trends",
                "industry_analysis",
                "competitive_landscape",
            ],
            "timeframe": state.timeframe or "last_12_months",
            "has_scraped_data": scraped_data is not None
            and not scraped_data.get("error"),
            "primary_source": (
                getattr(state, "target_url", None)
                if scraped_data
                else "traditional_sources"
            ),
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
