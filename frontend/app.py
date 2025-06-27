import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Hybrid Web Scraping & Research System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #cce7ff;
        border: 1px solid #99d6ff;
        color: #004085;
    }
</style>
""",
    unsafe_allow_html=True,
)


def check_api_health() -> bool:
    """Check if the API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def start_research_task(
    query: str, target_url: str = None, industry: str = None, timeframe: str = "current"
) -> Optional[str]:
    """Start a new research task"""
    try:
        payload = {
            "query": query,
            "target_url": target_url if target_url else None,
            "industry": industry if industry else None,
            "timeframe": timeframe,
        }
        response = requests.post(
            f"{API_BASE_URL}/research/start", json=payload, timeout=10
        )
        if response.status_code == 200:
            return response.json()["task_id"]
        else:
            st.error(f"Failed to start research: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error starting research: {str(e)}")
        return None


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status"""
    try:
        response = requests.get(f"{API_BASE_URL}/research/status/{task_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_task_results(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task results"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/research/results/{task_id}", timeout=10
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 202:
            return {"status": "not_ready"}
        return None
    except:
        return None


def get_recent_tasks() -> list:
    """Get recent tasks"""
    try:
        response = requests.get(f"{API_BASE_URL}/research/tasks?limit=10", timeout=5)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []


def display_progress_bar(status: str, progress: Dict[str, Any] = None):
    """Display progress bar based on task status"""
    progress_mapping = {
        "queued": 10,
        "running": 50,
        "completed": 100,
        "completed_with_errors": 100,
        "failed": 0,
        "cancelled": 0,
    }

    percentage = progress_mapping.get(status, 0)
    if progress and "percentage" in progress:
        percentage = progress["percentage"]

    # Color based on status
    color = "normal"
    if status == "failed":
        color = "red"
    elif status == "completed_with_errors":
        color = "orange"
    elif status == "completed":
        color = "green"

    st.progress(percentage / 100)

    stage = progress.get("stage", status) if progress else status
    st.write(
        f"Status: {status.replace('_', ' ').title()} - {stage.replace('_', ' ').title()}"
    )


def display_analysis_results(results: Dict[str, Any]):
    """Display analysis results in a structured format"""

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Sources Found", results.get("sources_found", 0))
    with col2:
        st.metric(
            "Data Points",
            results.get("analysis_results", {})
            .get("processed_data_summary", {})
            .get("total_data_points", 0),
        )
    with col3:
        compliance = results.get("compliance_status", {})
        approved = compliance.get("approved_sources_count", 0)
        total = compliance.get("total_sources_checked", 0)
        st.metric("Compliance Rate", f"{approved}/{total}" if total > 0 else "N/A")
    with col4:
        analysis_data = results.get("analysis_results", {})
        quality_score = analysis_data.get("data_quality_score", 0)
        st.metric("Data Quality", f"{quality_score:.2f}" if quality_score else "N/A")

    # Analysis Results Tabs
    if results.get("analysis_results"):
        analysis_data = results["analysis_results"]

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Analysis Summary", "🔍 Insights", "📈 Trends", "⚖️ Compliance"]
        )

        with tab1:
            st.subheader("Analysis Summary")
            if analysis_data.get("summary"):
                st.write(analysis_data["summary"])

            # Data breakdown
            if analysis_data.get("processed_data_summary"):
                data_summary = analysis_data["processed_data_summary"]
                st.subheader("Data Breakdown")

                if data_summary.get("category_breakdown"):
                    df = pd.DataFrame(
                        list(data_summary["category_breakdown"].items()),
                        columns=["Category", "Count"],
                    )
                    fig = px.bar(df, x="Category", y="Count", title="Data by Category")
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Key Insights")
            if analysis_data.get("insights"):
                for i, insight in enumerate(analysis_data["insights"], 1):
                    with st.expander(
                        f"Insight {i}: {insight.get('type', 'Unknown').replace('_', ' ').title()}"
                    ):
                        st.write(
                            f"**Key Finding:** {insight.get('key_finding', 'N/A')}"
                        )
                        st.write(
                            f"**Confidence Level:** {insight.get('confidence_level', 'N/A')}"
                        )
                        st.write(
                            f"**Actionability:** {insight.get('actionability', 'N/A')}"
                        )

        with tab3:
            st.subheader("Trend Analysis")
            analysis_results = analysis_data.get("results", {})

            if analysis_results.get("trends"):
                trends = analysis_results["trends"]
                st.write(trends.get("analysis", "No trend analysis available"))

                # Display trend indicators if available
                if trends.get("trend_indicators"):
                    st.subheader("Trend Indicators")
                    for indicator in trends["trend_indicators"]:
                        st.write(f"• {indicator}")

            if analysis_results.get("sentiment"):
                sentiment = analysis_results["sentiment"]
                st.subheader("Identity Sentiment")

                if sentiment.get("sentiment_distribution"):
                    # Create sentiment pie chart
                    sentiment_data = sentiment["sentiment_distribution"]
                    fig = px.pie(
                        values=list(sentiment_data.values()),
                        names=list(sentiment_data.keys()),
                        title="Identity Sentiment Distribution",
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("Compliance Report")
            compliance = results.get("compliance_status", {})

            if compliance:
                st.write(
                    f"**Total Sources Checked:** {compliance.get('total_sources_checked', 0)}"
                )
                st.write(
                    f"**Approved Sources:** {compliance.get('approved_sources_count', 0)}"
                )

                if compliance.get("compliance_report"):
                    st.subheader("Detailed Compliance Report")
                    st.text_area(
                        "Compliance Details",
                        compliance["compliance_report"],
                        height=300,
                    )


def display_strategy_results(strategy_results: Dict[str, Any]):
    """Display strategy results"""
    if not strategy_results:
        st.info("No strategy results available")
        return

    st.subheader("Strategic Analysis")

    # Executive Summary
    if strategy_results.get("executive_summary"):
        st.subheader("Executive Summary")
        st.write(strategy_results["executive_summary"])

    # Recommendations
    if strategy_results.get("recommendations"):
        st.subheader("Strategic Recommendations")
        recommendations = strategy_results["recommendations"]
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"Recommendation {i}"):
                st.write(rec)

    # Implementation Plan
    if strategy_results.get("implementation_plan"):
        st.subheader("Implementation Plan")
        st.write(strategy_results["implementation_plan"])


def main():
    """Main Streamlit application"""

    # Header
    st.markdown(
        '<h1 class="main-header">🔍 Hybrid Web Scraping & Research System</h1>',
        unsafe_allow_html=True,
    )

    # Check API health
    if not check_api_health():
        st.markdown(
            '<div class="error-box">⚠️ API is not accessible. Please ensure the FastAPI backend is running on http://localhost:8000</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")

        # API Status
        st.success("✅ API Connected")

        # Navigation
        page = st.selectbox(
            "Navigate to:",
            ["New Research", "Task Monitor", "Recent Results", "System Info"],
        )

        st.markdown("---")

        # Settings
        st.subheader("Settings")
        auto_refresh = st.checkbox("Auto-refresh task status", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 10)

    # Main content based on navigation
    if page == "New Research":
        st.header("🔍 Start New Hybrid Web Scraping & Research")

        with st.form("research_form"):
            st.subheader("Research Parameters")

            # URL Input for Hybrid Scraping
            target_url = st.text_input(
                "Target Website URL (Optional)",
                placeholder="e.g., https://example.com/article",
                help="Enter a URL to scrape and analyze. If provided, the system will scrape this site and analyze it based on your query. If left empty, traditional research sources will be used.",
            )

            query = st.text_area(
                "Research Query*",
                placeholder="e.g., Analyze the behavioral patterns of people in this post, or extract key financial data from this company page.",
                help="Describe what analysis or research you want to conduct. If you provided a URL above, this query will guide what information is extracted from the scraped content.",
            )

            col1, col2 = st.columns(2)
            with col1:
                industry = st.selectbox(
                    "Industry (Optional)",
                    [
                        "",
                        "Technology",
                        "Healthcare",
                        "Finance",
                        "Automotive",
                        "Retail",
                        "Energy",
                        "Other",
                    ],
                )

            with col2:
                timeframe = st.selectbox(
                    "Time Frame",
                    ["current", "last_6_months", "last_12_months", "last_2_years"],
                )

            submitted = st.form_submit_button(
                "🚀 Start Hybrid Research", type="primary"
            )

            if submitted:
                if not query.strip():
                    st.error("Please enter a research query")
                else:
                    with st.spinner("Starting hybrid research task..."):
                        task_id = start_research_task(
                            query, target_url, industry, timeframe
                        )

                        if task_id:
                            st.success(f"Research task started! Task ID: {task_id}")
                            st.session_state.current_task_id = task_id
                            st.session_state.task_start_time = time.time()
                            st.rerun()

        # Display current task progress if any
        if "current_task_id" in st.session_state:
            task_id = st.session_state.current_task_id

            st.markdown("---")
            st.subheader(f"Current Task Progress: {task_id}")

            status_data = get_task_status(task_id)
            if status_data:
                display_progress_bar(status_data["status"], status_data.get("progress"))

                # Auto-refresh
                if auto_refresh and status_data["status"] not in [
                    "completed",
                    "failed",
                    "cancelled",
                ]:
                    time.sleep(refresh_interval)
                    st.rerun()

                # Show results if completed
                if status_data["status"] in ["completed", "completed_with_errors"]:
                    results = get_task_results(task_id)
                    if results and results.get("status") != "not_ready":
                        st.success("✅ Research completed!")

                        # Display results
                        display_analysis_results(results)

                        # Display strategy results
                        if results.get("strategy_results"):
                            display_strategy_results(results["strategy_results"])

                        # Display final report
                        if results.get("final_report"):
                            st.subheader("📄 Final Report")
                            st.text_area(
                                "Complete Research Report",
                                results["final_report"],
                                height=400,
                            )

                            # Download button
                            st.download_button(
                                "📥 Download Report",
                                results["final_report"],
                                file_name=f"Identity_research_{task_id[:8]}.txt",
                                mime="text/plain",
                            )

    elif page == "Task Monitor":
        st.header("📋 Task Monitor")

        # Get recent tasks
        tasks = get_recent_tasks()

        if not tasks:
            st.info("No recent tasks found")
        else:
            for task in tasks:
                with st.expander(f"Task {task['task_id'][:8]} - {task['status']}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Status:** {task['status']}")
                        st.write(f"**Created:** {task['created_at']}")
                        st.write(f"**Updated:** {task['updated_at']}")

                    with col2:
                        if task.get("progress"):
                            display_progress_bar(task["status"], task["progress"])

                        if task["status"] in ["completed", "completed_with_errors"]:
                            if st.button(
                                f"View Results", key=f"view_{task['task_id']}"
                            ):
                                st.session_state.view_task_id = task["task_id"]
                                st.rerun()

    elif page == "Recent Results":
        st.header("📊 Recent Results")

        # Check if viewing specific task
        if "view_task_id" in st.session_state:
            task_id = st.session_state.view_task_id
            results = get_task_results(task_id)

            if results and results.get("status") != "not_ready":
                st.subheader(f"Results for Task: {task_id}")

                # Basic info
                st.write(f"**Query:** {results.get('research_query', 'N/A')}")
                st.write(f"**Industry:** {results.get('industry', 'N/A')}")
                st.write(f"**Completed:** {results.get('completed_at', 'N/A')}")

                # Results
                display_analysis_results(results)

                if results.get("strategy_results"):
                    display_strategy_results(results["strategy_results"])

                if st.button("← Back to Task List"):
                    del st.session_state.view_task_id
                    st.rerun()
            else:
                st.error("Could not load results for this task")
        else:
            # Show summary of recent completed tasks
            tasks = [
                t
                for t in get_recent_tasks()
                if t["status"] in ["completed", "completed_with_errors"]
            ]

            if not tasks:
                st.info("No completed tasks found")
            else:
                for task in tasks:
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])

                        with col1:
                            st.write(f"**Task:** {task['task_id'][:8]}")
                            st.write(f"**Completed:** {task['updated_at']}")

                        with col2:
                            status_color = (
                                "🟢" if task["status"] == "completed" else "🟡"
                            )
                            st.write(f"{status_color} {task['status']}")

                        with col3:
                            if st.button("View", key=f"view_recent_{task['task_id']}"):
                                st.session_state.view_task_id = task["task_id"]
                                st.rerun()

                        st.markdown("---")

    elif page == "System Info":
        st.header("ℹ️ System Information")

        try:
            # Get workflow info
            response = requests.get(f"{API_BASE_URL}/workflow/info")
            if response.status_code == 200:
                workflow_info = response.json()

                st.subheader("Workflow Configuration")
                config = workflow_info.get("configuration", {})
                st.write(f"**LLM Provider:** {config.get('llm_provider', 'N/A')}")
                st.write(f"**Model:** {config.get('llm_model', 'N/A')}")
                st.write(f"**Temperature:** {config.get('temperature', 'N/A')}")

                st.subheader("Agent Status")
                agents = workflow_info.get("agents", {})
                for agent, status in agents.items():
                    status_icon = "✅" if status == "Ready" else "❌"
                    st.write(f"{status_icon} **{agent.title()}:** {status}")

                st.subheader("Workflow Visualization")
                st.text(workflow_info.get("visualization", "Not available"))

            # API Health
            health_response = requests.get(f"{API_BASE_URL}/health")
            if health_response.status_code == 200:
                health = health_response.json()

                st.subheader("System Health")
                st.write(
                    f"**API Status:** {'✅ Healthy' if health.get('api') == 'healthy' else '❌ Unhealthy'}"
                )
                st.write(
                    f"**Workflow Status:** {'✅ Healthy' if health.get('workflow') == 'healthy' else '❌ Unhealthy'}"
                )
                st.write(f"**Active Tasks:** {health.get('active_tasks', 0)}")
                st.write(f"**Last Check:** {health.get('timestamp', 'N/A')}")

        except Exception as e:
            st.error(f"Could not retrieve system information: {str(e)}")


if __name__ == "__main__":
    main()
