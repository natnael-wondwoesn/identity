from typing import Dict, Any, List
import re
from datetime import datetime
from urllib.parse import urlparse

from .base_agent import BaseAgent, AgentState
from langchain_core.language_models import BaseLanguageModel


class ComplianceAgent(BaseAgent):
    """
    Compliance Agent ensures all data sources and collection methods
    comply with ethical guidelines, legal requirements, and platform terms of service.
    """

    def __init__(self, llm: BaseLanguageModel):
        super().__init__(llm, "ComplianceAgent")
        self.compliance_rules = self._init_compliance_rules()
        self.approved_domains = self._init_approved_domains()

    def _init_compliance_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize compliance rules and guidelines"""
        return {
            "data_privacy": {
                "rules": [
                    "No collection of personal identifiable information (PII)",
                    "No behavioral tracking of individuals",
                    "No unauthorized access to private accounts",
                    "Respect robots.txt and terms of service",
                ],
                "severity": "critical",
            },
            "legal_compliance": {
                "rules": [
                    "GDPR compliance for EU data",
                    "CCPA compliance for California data",
                    "Respect copyright and intellectual property",
                    "Follow platform-specific API terms",
                ],
                "severity": "critical",
            },
            "ethical_guidelines": {
                "rules": [
                    "Use only publicly available information",
                    "Transparent data collection purposes",
                    "No deceptive data gathering methods",
                    "Respect rate limits and fair usage",
                ],
                "severity": "high",
            },
            "technical_standards": {
                "rules": [
                    "Implement proper error handling",
                    "Use secure data transmission",
                    "Log all data collection activities",
                    "Maintain data minimization principles",
                ],
                "severity": "medium",
            },
        }

    def _init_approved_domains(self) -> List[str]:
        """Initialize list of pre-approved domains for data collection"""
        return [
            "gov",  # Government domains
            "edu",  # Educational institutions
            "census.gov",
            "bls.gov",
            "fred.stlouisfed.org",
            "sec.gov",  # US Government
            "reuters.com",
            "bloomberg.com",
            "wsj.com",  # Financial news
            "statista.com",
            "ibisworld.com",  # Market research firms
            "sec.gov",
            "edgar.sec.gov",  # Securities filings
        ]

    async def execute(self, state: AgentState) -> AgentState:
        """Execute compliance validation for all identified sources and methods"""

        self.log_action(
            "Starting compliance validation",
            {"sources_to_check": len(state.sources), "query": state.research_query},
        )

        try:
            # Validate research query for compliance
            query_compliance = await self._validate_research_query(state.research_query)

            # Validate each data source
            source_validations = await self._validate_sources(state.sources)

            # Check data collection methods
            method_compliance = await self._validate_collection_methods(state)

            # Generate compliance report
            compliance_report = await self._generate_compliance_report(
                query_compliance, source_validations, method_compliance
            )

            # Filter approved sources
            approved_sources = self._filter_approved_sources(
                state.sources, source_validations
            )

            # Update state with compliance results
            updates = {
                "compliance_status": {
                    "query_compliance": query_compliance,
                    "source_validations": source_validations,
                    "method_compliance": method_compliance,
                    "compliance_report": compliance_report,
                    "total_sources_checked": len(state.sources),
                    "approved_sources_count": len(approved_sources),
                    "validation_timestamp": datetime.now().isoformat(),
                },
                "sources": approved_sources,  # Replace with only approved sources
            }

            state = self.update_state(state, updates)

            self.log_action(
                "Compliance validation completed",
                {
                    "approved_sources": len(approved_sources),
                    "rejected_sources": len(state.sources) - len(approved_sources),
                    "overall_compliance": (
                        "APPROVED" if approved_sources else "REJECTED"
                    ),
                },
            )

        except Exception as e:
            error_msg = f"Compliance validation failed: {str(e)}"
            self.logger.error(error_msg)
            state.errors.append(error_msg)

        return state

    async def _validate_research_query(self, query: str) -> Dict[str, Any]:
        """Validate that the research query is compliant and ethical"""

        prompt = f"""
        Analyze this market research query for compliance and ethical concerns:
        
        Query: "{query}"
        
        Check for:
        1. Does it seek personal or private information about individuals?
        2. Does it involve any deceptive or harmful practices?
        3. Is it focused on legitimate business/market research?
        4. Are there any privacy or legal red flags?
        
        Respond with:
        - APPROVED: Query is compliant and ethical
        - NEEDS_MODIFICATION: Query needs adjustments (specify what)
        - REJECTED: Query violates compliance rules (explain why)
        
        Include reasoning for your decision.
        """

        response = await self.generate_response(prompt)

        # Parse the response to determine approval status
        status = "NEEDS_REVIEW"
        if "APPROVED" in response.upper():
            status = "APPROVED"
        elif "REJECTED" in response.upper():
            status = "REJECTED"
        elif "NEEDS_MODIFICATION" in response.upper():
            status = "NEEDS_MODIFICATION"

        return {
            "status": status,
            "analysis": response,
            "validated_at": datetime.now().isoformat(),
            "compliant": status == "APPROVED",
        }

    async def _validate_sources(
        self, sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate each data source for compliance"""

        validations = []

        for source in sources:
            validation = await self._validate_single_source(source)
            validations.append(validation)

        return validations

    async def _validate_single_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single data source"""

        validation_result = {
            "source_url": source.get("url", ""),
            "source_type": source.get("type", ""),
            "domain_approved": False,
            "terms_compliant": False,
            "privacy_compliant": False,
            "overall_status": "PENDING",
            "issues": [],
            "recommendations": [],
        }

        # Check domain approval
        if source.get("url"):
            domain = urlparse(source["url"]).netloc.lower()
            validation_result["domain_approved"] = any(
                approved_domain in domain for approved_domain in self.approved_domains
            )

        # Check source type compliance
        if source.get("type") in ["government", "news", "research"]:
            validation_result["terms_compliant"] = True
            validation_result["privacy_compliant"] = True

        # Check compliance level
        if source.get("compliance_level") == "high":
            validation_result["privacy_compliant"] = True

        # Determine overall status
        if (
            validation_result["domain_approved"]
            and validation_result["terms_compliant"]
            and validation_result["privacy_compliant"]
        ):
            validation_result["overall_status"] = "APPROVED"
        else:
            validation_result["overall_status"] = "REJECTED"
            if not validation_result["domain_approved"]:
                validation_result["issues"].append("Domain not in approved list")
            if not validation_result["terms_compliant"]:
                validation_result["issues"].append("Terms of service concerns")
            if not validation_result["privacy_compliant"]:
                validation_result["issues"].append("Privacy compliance issues")

        # Add recommendations for improvement
        if validation_result["overall_status"] == "REJECTED":
            validation_result["recommendations"].append(
                "Use official APIs where available"
            )
            validation_result["recommendations"].append(
                "Verify terms of service compliance"
            )
            validation_result["recommendations"].append(
                "Consider alternative approved sources"
            )

        return validation_result

    async def _validate_collection_methods(self, state: AgentState) -> Dict[str, Any]:
        """Validate data collection methods for compliance"""

        methods_analysis = {
            "rate_limiting": "REQUIRED",
            "user_agent_identification": "REQUIRED",
            "robots_txt_compliance": "REQUIRED",
            "api_usage": "PREFERRED",
            "data_minimization": "REQUIRED",
            "purpose_limitation": "REQUIRED",
            "retention_policy": "REQUIRED",
            "overall_compliance": "APPROVED",
        }

        # Check if any high-risk collection methods are being used
        risk_factors = []

        # Add risk assessment based on query and sources
        if len(state.sources) > 20:
            risk_factors.append("High volume of sources may indicate scraping approach")

        if any(
            "social" in source.get("description", "").lower()
            for source in state.sources
        ):
            risk_factors.append(
                "Social media sources require extra privacy precautions"
            )

        methods_analysis["risk_factors"] = risk_factors
        methods_analysis["mitigation_required"] = len(risk_factors) > 0

        return methods_analysis

    def _filter_approved_sources(
        self, sources: List[Dict[str, Any]], validations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter sources to only include approved ones"""

        approved_sources = []

        for source, validation in zip(sources, validations):
            if validation.get("overall_status") == "APPROVED":
                # Add compliance metadata to source
                source["compliance_validated"] = True
                source["validation_timestamp"] = datetime.now().isoformat()
                source["compliance_score"] = self._calculate_compliance_score(
                    validation
                )
                approved_sources.append(source)

        return approved_sources

    def _calculate_compliance_score(self, validation: Dict[str, Any]) -> float:
        """Calculate a compliance score for a validated source"""
        score = 0.0

        if validation.get("domain_approved"):
            score += 0.4
        if validation.get("terms_compliant"):
            score += 0.3
        if validation.get("privacy_compliant"):
            score += 0.3

        return round(score, 2)

    async def _generate_compliance_report(
        self,
        query_compliance: Dict[str, Any],
        source_validations: List[Dict[str, Any]],
        method_compliance: Dict[str, Any],
    ) -> str:
        """Generate a comprehensive compliance report"""

        approved_sources = len(
            [v for v in source_validations if v.get("overall_status") == "APPROVED"]
        )
        rejected_sources = len(source_validations) - approved_sources

        prompt = f"""
        Generate a compliance report for this market research project:
        
        Query Compliance: {query_compliance.get('status', 'UNKNOWN')}
        Sources Approved: {approved_sources}
        Sources Rejected: {rejected_sources}
        Method Compliance: {method_compliance.get('overall_compliance', 'PENDING')}
        
        Include:
        1. Executive Summary of compliance status
        2. Key compliance achievements
        3. Areas of concern (if any)
        4. Recommendations for implementation
        5. Risk mitigation strategies
        6. Data handling requirements
        
        Make it professional and actionable for a development team.
        """

        report = await self.generate_response(prompt)

        # Add structured data to report
        structured_summary = f"""
        
        COMPLIANCE SUMMARY:
        ==================
        Query Status: {query_compliance.get('status', 'UNKNOWN')}
        Approved Sources: {approved_sources}/{len(source_validations)}
        Overall Risk Level: {'LOW' if approved_sources > rejected_sources else 'MEDIUM'}
        Validation Date: {datetime.now().isoformat()}
        
        APPROVED DOMAINS:
        {', '.join(self.approved_domains[:10])}...
        
        """

        return report + structured_summary

    def get_compliance_guidelines(self) -> Dict[str, Any]:
        """Return current compliance guidelines for reference"""
        return {
            "rules": self.compliance_rules,
            "approved_domains": self.approved_domains,
            "last_updated": datetime.now().isoformat(),
            "version": "1.0",
        }

    async def validate_realtime_request(
        self, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a real-time data collection request"""

        validation = {"approved": False, "issues": [], "recommendations": []}

        # Check request URL
        if "url" in request_data:
            domain = urlparse(request_data["url"]).netloc.lower()
            if not any(approved in domain for approved in self.approved_domains):
                validation["issues"].append(f"Domain {domain} not in approved list")

        # Check request method
        if request_data.get("method", "").upper() in ["POST", "PUT", "DELETE"]:
            validation["issues"].append("Only GET requests allowed for data collection")

        # Check for personal data indicators
        personal_data_indicators = ["email", "phone", "address", "ssn", "personal"]
        if any(
            indicator in str(request_data).lower()
            for indicator in personal_data_indicators
        ):
            validation["issues"].append("Request may target personal data")

        validation["approved"] = len(validation["issues"]) == 0

        if not validation["approved"]:
            validation["recommendations"].extend(
                [
                    "Use approved domains only",
                    "Limit to GET requests for public data",
                    "Focus on aggregated, non-personal information",
                ]
            )

        return validation
