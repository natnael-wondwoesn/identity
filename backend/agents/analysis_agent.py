from typing import Dict, Any, List
import json
from datetime import datetime, timedelta
import re
from collections import defaultdict
import statistics

from .base_agent import BaseAgent, AgentState
from langchain_core.language_models import BaseLanguageModel


class AnalysisAgent(BaseAgent):
    """
    Analysis Agent processes collected market data to extract insights,
    identify trends, and generate actionable market intelligence.
    """

    def __init__(self, llm: BaseLanguageModel):
        super().__init__(llm, "AnalysisAgent")
        self.analysis_methods = self._init_analysis_methods()

    def _init_analysis_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available analysis methods and their configurations"""
        return {
            "trend_analysis": {
                "description": "Identify market trends and patterns over time",
                "applicable_to": ["financial_data", "market_metrics", "news_sentiment"],
                "output_type": "trends_report",
            },
            "competitive_analysis": {
                "description": "Analyze competitive landscape and positioning",
                "applicable_to": ["company_data", "industry_reports", "market_share"],
                "output_type": "competitive_report",
            },
            "sentiment_analysis": {
                "description": "Analyze market sentiment from news and reports",
                "applicable_to": ["news_articles", "press_releases", "analyst_reports"],
                "output_type": "sentiment_report",
            },
            "financial_analysis": {
                "description": "Analyze financial metrics and performance",
                "applicable_to": [
                    "financial_statements",
                    "market_data",
                    "economic_indicators",
                ],
                "output_type": "financial_report",
            },
            "market_sizing": {
                "description": "Estimate market size and growth potential",
                "applicable_to": [
                    "industry_data",
                    "government_statistics",
                    "research_reports",
                ],
                "output_type": "market_size_report",
            },
        }

    async def execute(self, state: AgentState) -> AgentState:
        """Execute comprehensive analysis of collected market data"""

        self.log_action(
            "Starting market data analysis",
            {
                "data_points": len(state.raw_data),
                "sources": len(state.sources),
                "query": state.research_query,
            },
        )

        if not state.raw_data and not state.sources:
            self.log_action("No data available for analysis")
            state.errors.append("No data available for analysis")
            return state

        try:
            # Determine appropriate analysis methods
            analysis_plan = await self._create_analysis_plan(state)

            # Execute data preprocessing
            processed_data = await self._preprocess_data(state.raw_data)

            # Perform different types of analysis
            analysis_results = {}

            if "trend_analysis" in analysis_plan["methods"]:
                analysis_results["trends"] = await self._perform_trend_analysis(
                    processed_data, state
                )

            if "sentiment_analysis" in analysis_plan["methods"]:
                analysis_results["sentiment"] = await self._perform_sentiment_analysis(
                    processed_data, state
                )

            if "competitive_analysis" in analysis_plan["methods"]:
                analysis_results["competitive"] = (
                    await self._perform_competitive_analysis(processed_data, state)
                )

            if "market_sizing" in analysis_plan["methods"]:
                analysis_results["market_size"] = await self._perform_market_sizing(
                    processed_data, state
                )

            # Generate insights and recommendations
            insights = await self._generate_insights(analysis_results, state)

            # Create comprehensive analysis summary
            analysis_summary = await self._create_analysis_summary(
                analysis_results, insights
            )

            # Update state with analysis results
            updates = {
                "analysis_results": {
                    "analysis_plan": analysis_plan,
                    "processed_data_summary": self._summarize_processed_data(
                        processed_data
                    ),
                    "results": analysis_results,
                    "insights": insights,
                    "summary": analysis_summary,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "data_quality_score": self._calculate_data_quality_score(
                        processed_data
                    ),
                }
            }

            state = self.update_state(state, updates)

            self.log_action(
                "Analysis completed successfully",
                {
                    "analysis_methods_used": len(analysis_results),
                    "insights_generated": len(insights),
                    "data_quality": updates["analysis_results"]["data_quality_score"],
                },
            )

        except Exception as e:
            error_msg = f"Analysis Agent failed: {str(e)}"
            self.logger.error(error_msg)
            state.errors.append(error_msg)

        return state

    async def _create_analysis_plan(self, state: AgentState) -> Dict[str, Any]:
        """Create an analysis plan based on available data and research objectives"""

        prompt = f"""
        Create an analysis plan for this market research project:
        
        Research Query: {state.research_query}
        Industry: {state.industry or 'General'}
        Available Data Types: {[item.get('data_type', 'unknown') for item in state.raw_data[:5]]}
        Number of Sources: {len(state.sources)}
        
        Based on the available data, recommend which analysis methods would be most valuable:
        
        Available Methods:
        1. Trend Analysis - Market trends and patterns over time
        2. Sentiment Analysis - Market sentiment from news and reports  
        3. Competitive Analysis - Competitive landscape analysis
        4. Financial Analysis - Financial metrics and performance
        5. Market Sizing - Market size and growth estimates
        
        Provide:
        1. Recommended analysis methods (list the most relevant ones)
        2. Priority order for analysis
        3. Expected insights from each method
        4. Data requirements for each method
        """

        plan_response = await self.generate_response(prompt)

        # Extract recommended methods (simplified extraction)
        recommended_methods = []
        for method in self.analysis_methods.keys():
            if method.replace("_", " ") in plan_response.lower():
                recommended_methods.append(method)

        # Ensure at least basic analysis if none detected
        if not recommended_methods:
            recommended_methods = ["trend_analysis", "sentiment_analysis"]

        return {
            "methods": recommended_methods,
            "plan_text": plan_response,
            "priority_order": recommended_methods,  # In order of recommendation
            "created_at": datetime.now().isoformat(),
        }

    async def _preprocess_data(
        self, raw_data: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Preprocess and categorize raw data for analysis"""

        processed_data = defaultdict(list)

        for item in raw_data:
            data_type = item.get("data_type", "unknown")

            # Clean and standardize data
            cleaned_item = {
                "original": item,
                "text_content": self._extract_text_content(item),
                "timestamp": self._extract_timestamp(item),
                "source": item.get("source", "unknown"),
                "data_type": data_type,
                "processed_at": datetime.now().isoformat(),
            }

            # Add sentiment indicators if text is available
            if cleaned_item["text_content"]:
                cleaned_item["estimated_sentiment"] = self._estimate_sentiment(
                    cleaned_item["text_content"]
                )

            processed_data[data_type].append(cleaned_item)

        # Convert defaultdict to regular dict for JSON serialization
        return dict(processed_data)

    def _extract_text_content(self, item: Dict[str, Any]) -> str:
        """Extract text content from data item"""
        text_fields = ["title", "summary", "content", "description", "text"]
        text_content = []

        for field in text_fields:
            if field in item and item[field]:
                text_content.append(str(item[field]))

        return " ".join(text_content)[:1000]  # Limit length

    def _extract_timestamp(self, item: Dict[str, Any]) -> str:
        """Extract timestamp from data item"""
        timestamp_fields = ["published", "date", "timestamp", "collected_at"]

        for field in timestamp_fields:
            if field in item and item[field]:
                return str(item[field])

        return datetime.now().isoformat()

    def _estimate_sentiment(self, text: str) -> str:
        """Simple sentiment estimation (in production, use proper NLP models)"""
        positive_words = [
            "good",
            "great",
            "excellent",
            "positive",
            "growth",
            "increase",
            "profit",
            "success",
        ]
        negative_words = [
            "bad",
            "poor",
            "decline",
            "decrease",
            "loss",
            "failure",
            "risk",
            "concern",
        ]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    async def _perform_trend_analysis(
        self, processed_data: Dict[str, List], state: AgentState
    ) -> Dict[str, Any]:
        """Perform trend analysis on time-series data"""

        prompt = f"""
        Analyze trends in this market research data:
        
        Research Query: {state.research_query}
        Data Categories: {list(processed_data.keys())}
        Total Data Points: {sum(len(items) for items in processed_data.values())}
        
        Sample Data: {str(list(processed_data.values())[0][:3] if processed_data else 'No data available')}
        
        Identify:
        1. Key trends in the market/industry
        2. Patterns in the data over time
        3. Growth indicators
        4. Cyclical patterns
        5. Emerging themes
        
        Provide specific, actionable insights about market trends.
        """

        trend_analysis = await self.generate_response(prompt)

        return {
            "analysis": trend_analysis,
            "data_points_analyzed": sum(
                len(items) for items in processed_data.values()
            ),
            "trend_indicators": self._extract_trend_indicators(processed_data),
            "confidence_level": self._calculate_confidence_level(processed_data),
            "analysis_type": "trend_analysis",
        }

    async def _perform_sentiment_analysis(
        self, processed_data: Dict[str, List], state: AgentState
    ) -> Dict[str, Any]:
        """Perform sentiment analysis on textual data"""

        # Collect sentiment data
        sentiment_data = []
        for data_type, items in processed_data.items():
            for item in items:
                if item.get("estimated_sentiment"):
                    sentiment_data.append(item["estimated_sentiment"])

        if not sentiment_data:
            return {"error": "No text data available for sentiment analysis"}

        # Calculate sentiment distribution
        sentiment_counts = defaultdict(int)
        for sentiment in sentiment_data:
            sentiment_counts[sentiment] += 1

        total_items = len(sentiment_data)
        sentiment_distribution = {
            sentiment: (count / total_items) * 100
            for sentiment, count in sentiment_counts.items()
        }

        prompt = f"""
        Analyze market sentiment based on this data:
        
        Research Query: {state.research_query}
        
        Sentiment Distribution:
        - Positive: {sentiment_distribution.get('positive', 0):.1f}%
        - Negative: {sentiment_distribution.get('negative', 0):.1f}%
        - Neutral: {sentiment_distribution.get('neutral', 0):.1f}%
        
        Total sources analyzed: {total_items}
        
        Provide insights about:
        1. Overall market sentiment
        2. Implications for the industry/market
        3. Risk factors indicated by sentiment
        4. Opportunities suggested by positive sentiment
        """

        sentiment_analysis = await self.generate_response(prompt)

        return {
            "analysis": sentiment_analysis,
            "sentiment_distribution": sentiment_distribution,
            "total_items_analyzed": total_items,
            "dominant_sentiment": max(
                sentiment_distribution, key=sentiment_distribution.get
            ),
            "analysis_type": "sentiment_analysis",
        }

    async def _perform_competitive_analysis(
        self, processed_data: Dict[str, List], state: AgentState
    ) -> Dict[str, Any]:
        """Perform competitive landscape analysis"""

        prompt = f"""
        Perform competitive analysis based on available market data:
        
        Research Query: {state.research_query}
        Industry: {state.industry or 'General'}
        Data Sources: {len(state.sources)} sources
        
        Available data types: {list(processed_data.keys())}
        
        Analyze:
        1. Competitive landscape overview
        2. Key market players (based on available data)
        3. Market positioning opportunities
        4. Competitive advantages/disadvantages
        5. Market entry barriers
        6. Strategic recommendations
        
        Focus on actionable competitive intelligence.
        """

        competitive_analysis = await self.generate_response(prompt)

        return {
            "analysis": competitive_analysis,
            "data_sources_used": len(state.sources),
            "market_players_identified": self._count_entities_mentioned(
                competitive_analysis
            ),
            "analysis_type": "competitive_analysis",
        }

    async def _perform_market_sizing(
        self, processed_data: Dict[str, List], state: AgentState
    ) -> Dict[str, Any]:
        """Perform market sizing analysis"""

        prompt = f"""
        Estimate market size and growth potential:
        
        Research Query: {state.research_query}
        Industry: {state.industry or 'General'}
        Available Data: {list(processed_data.keys())}
        
        Based on the available information, provide:
        1. Market size estimates (if data supports it)
        2. Growth trends and projections
        3. Market segments analysis
        4. Geographic distribution (if applicable)
        5. Key growth drivers
        6. Market maturity assessment
        
        Note: Base estimates only on available data and clearly state limitations.
        """

        market_sizing = await self.generate_response(prompt)

        return {
            "analysis": market_sizing,
            "data_confidence": "medium" if len(processed_data) > 5 else "low",
            "analysis_type": "market_sizing",
            "limitations": "Estimates based on limited public data sources",
        }

    async def _generate_insights(
        self, analysis_results: Dict[str, Any], state: AgentState
    ) -> List[Dict[str, Any]]:
        """Generate actionable insights from analysis results"""

        insights = []

        for analysis_type, results in analysis_results.items():
            if isinstance(results, dict) and "analysis" in results:
                insight = {
                    "type": analysis_type,
                    "key_finding": self._extract_key_finding(results["analysis"]),
                    "confidence_level": results.get("confidence_level", "medium"),
                    "actionability": self._assess_actionability(results["analysis"]),
                    "source_analysis": analysis_type,
                }
                insights.append(insight)

        return insights

    def _extract_key_finding(self, analysis_text: str) -> str:
        """Extract the most important finding from analysis text"""
        # Simple extraction - in production, use more sophisticated NLP
        sentences = analysis_text.split(".")
        for sentence in sentences[:3]:  # Check first 3 sentences
            if any(
                keyword in sentence.lower()
                for keyword in ["key", "important", "significant", "main"]
            ):
                return sentence.strip()

        return sentences[0].strip() if sentences else "No key finding extracted"

    def _assess_actionability(self, analysis_text: str) -> str:
        """Assess how actionable the analysis results are"""
        actionable_indicators = [
            "recommend",
            "should",
            "opportunity",
            "strategy",
            "action",
        ]

        if any(
            indicator in analysis_text.lower() for indicator in actionable_indicators
        ):
            return "high"
        elif "trend" in analysis_text.lower() or "pattern" in analysis_text.lower():
            return "medium"
        else:
            return "low"

    async def _create_analysis_summary(
        self, analysis_results: Dict[str, Any], insights: List[Dict[str, Any]]
    ) -> str:
        """Create a comprehensive summary of all analysis results"""

        prompt = f"""
        Create an executive summary of this market research analysis:
        
        Analysis Types Completed: {list(analysis_results.keys())}
        Total Insights Generated: {len(insights)}
        
        Key Insights:
        {chr(10).join([f"- {insight['key_finding']}" for insight in insights[:5]])}
        
        Create a concise executive summary that includes:
        1. Overview of analysis scope
        2. Key findings and insights
        3. Strategic recommendations
        4. Data limitations and confidence levels
        5. Next steps for further research
        
        Make it business-focused and actionable.
        """

        return await self.generate_response(prompt)

    def _summarize_processed_data(
        self, processed_data: Dict[str, List]
    ) -> Dict[str, Any]:
        """Create a summary of processed data for reporting"""
        return {
            "total_categories": len(processed_data),
            "category_breakdown": {
                category: len(items) for category, items in processed_data.items()
            },
            "total_data_points": sum(len(items) for items in processed_data.values()),
            "processing_timestamp": datetime.now().isoformat(),
        }

    def _calculate_data_quality_score(self, processed_data: Dict[str, List]) -> float:
        """Calculate a data quality score based on completeness and variety"""
        if not processed_data:
            return 0.0

        # Factor in data variety
        variety_score = min(len(processed_data) / 5.0, 1.0)  # Max score at 5 categories

        # Factor in data volume
        total_items = sum(len(items) for items in processed_data.values())
        volume_score = min(total_items / 20.0, 1.0)  # Max score at 20 items

        # Factor in text content availability
        text_items = sum(
            1
            for items in processed_data.values()
            for item in items
            if item.get("text_content")
        )
        text_score = min(text_items / total_items, 1.0) if total_items > 0 else 0.0

        # Weighted average
        quality_score = variety_score * 0.3 + volume_score * 0.4 + text_score * 0.3
        return round(quality_score, 2)

    def _extract_trend_indicators(self, processed_data: Dict[str, List]) -> List[str]:
        """Extract trend indicators from processed data"""
        indicators = []

        # Simple trend detection based on data patterns
        for category, items in processed_data.items():
            if len(items) > 3:
                indicators.append(
                    f"Sufficient data available for {category} trend analysis"
                )

        return indicators

    def _calculate_confidence_level(self, processed_data: Dict[str, List]) -> str:
        """Calculate confidence level based on data quality and quantity"""
        total_items = sum(len(items) for items in processed_data.values())

        if total_items >= 15:
            return "high"
        elif total_items >= 8:
            return "medium"
        else:
            return "low"

    def _count_entities_mentioned(self, text: str) -> int:
        """Count potential company/entity mentions in text"""
        # Simple entity counting - in production, use proper NER
        entity_patterns = [r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", r"\b[A-Z]{2,}\b"]
        entities = set()

        for pattern in entity_patterns:
            matches = re.findall(pattern, text)
            entities.update(matches)

        return len(entities)
