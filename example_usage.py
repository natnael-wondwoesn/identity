#!/usr/bin/env python3
"""
Example usage of the Hybrid Web Scraping & Research System

This script demonstrates how to use the system for different types of research:
1. Pure web scraping with analysis
2. Hybrid scraping + traditional research
3. Traditional research only
"""

import asyncio
import requests
import json
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"


def check_api_status():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except:
        return False


def start_research_task(query, target_url=None, industry=None, timeframe="current"):
    """Start a new research task"""
    payload = {
        "query": query,
        "target_url": target_url,
        "industry": industry,
        "timeframe": timeframe,
    }

    try:
        response = requests.post(f"{API_BASE_URL}/research/start", json=payload)
        if response.status_code == 200:
            return response.json()["task_id"]
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def wait_for_completion(task_id, timeout=300):
    """Wait for task completion and return results"""
    import time

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Check status
            status_response = requests.get(f"{API_BASE_URL}/research/status/{task_id}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"Status: {status_data['status']}")

                if status_data["status"] in [
                    "completed",
                    "completed_with_errors",
                    "failed",
                ]:
                    # Get results
                    results_response = requests.get(
                        f"{API_BASE_URL}/research/results/{task_id}"
                    )
                    if results_response.status_code == 200:
                        return results_response.json()
                    else:
                        print(f"Could not get results: {results_response.text}")
                        return None

            time.sleep(5)  # Wait 5 seconds before checking again

        except Exception as e:
            print(f"Error checking status: {e}")
            time.sleep(5)

    print("Timeout waiting for completion")
    return None


def example_1_pure_web_scraping():
    """Example 1: Pure web scraping and analysis"""
    print("\n=== Example 1: Pure Web Scraping ===")
    print("Scraping a company website and analyzing their products/services")

    query = "Extract all product information, pricing, and company details from this website"
    target_url = "https://example-company.com"  # Replace with actual URL

    print(f"Query: {query}")
    print(f"Target URL: {target_url}")

    task_id = start_research_task(query, target_url)
    if task_id:
        print(f"Task started: {task_id}")
        results = wait_for_completion(task_id)
        if results:
            print("✅ Research completed!")
            print(f"Sources found: {results.get('sources_found', 0)}")
            if results.get("final_report"):
                print("\nFinal Report Preview:")
                print(results["final_report"][:500] + "...")
        else:
            print("❌ Research failed or timed out")
    else:
        print("❌ Failed to start task")


def example_2_hybrid_research():
    """Example 2: Hybrid scraping + traditional research"""
    print("\n=== Example 2: Hybrid Research ===")
    print("Scraping a company website and combining with market research")

    query = "Analyze this company's competitive position in the technology market"
    target_url = "https://tech-company.com"  # Replace with actual URL
    industry = "Technology"

    print(f"Query: {query}")
    print(f"Target URL: {target_url}")
    print(f"Industry: {industry}")

    task_id = start_research_task(query, target_url, industry)
    if task_id:
        print(f"Task started: {task_id}")
        results = wait_for_completion(task_id)
        if results:
            print("✅ Hybrid research completed!")
            print(f"Sources found: {results.get('sources_found', 0)}")
            if results.get("final_report"):
                print("\nFinal Report Preview:")
                print(results["final_report"][:500] + "...")
        else:
            print("❌ Research failed or timed out")
    else:
        print("❌ Failed to start task")


def example_3_traditional_research():
    """Example 3: Traditional research only (no URL)"""
    print("\n=== Example 3: Traditional Research ===")
    print("Traditional market research without web scraping")

    query = "Market trends in renewable energy sector for 2024"
    industry = "Energy"
    timeframe = "last_12_months"

    print(f"Query: {query}")
    print(f"Industry: {industry}")
    print(f"Timeframe: {timeframe}")

    task_id = start_research_task(query, None, industry, timeframe)
    if task_id:
        print(f"Task started: {task_id}")
        results = wait_for_completion(task_id)
        if results:
            print("✅ Traditional research completed!")
            print(f"Sources found: {results.get('sources_found', 0)}")
            if results.get("final_report"):
                print("\nFinal Report Preview:")
                print(results["final_report"][:500] + "...")
        else:
            print("❌ Research failed or timed out")
    else:
        print("❌ Failed to start task")


def main():
    """Main function to run examples"""
    print("🔍 Hybrid Web Scraping & Research System - Examples")
    print("=" * 60)

    # Check if API is running
    if not check_api_status():
        print("❌ API is not running. Please start the backend first:")
        print("   cd backend")
        print("   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        return

    print("✅ API is running")

    # Run examples
    try:
        # Uncomment the examples you want to run:

        # example_1_pure_web_scraping()
        # example_2_hybrid_research()
        example_3_traditional_research()  # This one works without external URLs

        print("\n" + "=" * 60)
        print("Examples completed! Check the Streamlit frontend for more details:")
        print("http://localhost:8501")

    except KeyboardInterrupt:
        print("\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")


if __name__ == "__main__":
    main()
