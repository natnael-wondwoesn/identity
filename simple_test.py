#!/usr/bin/env python3
"""
Simple test of the Hybrid Web Scraping & Research System
"""

import requests
import time
import json

API_BASE_URL = "http://localhost:8000"


def test_system():
    print("🔍 Testing Hybrid Web Scraping & Research System")
    print("=" * 50)

    # Test 1: Check API health
    print("1. Testing API health...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API Status: {health['api']}")
            print(f"✅ Workflow Status: {health['workflow']}")
            print(f"✅ Agents: {health.get('agents', {}).get('workflow', 'Unknown')}")
        else:
            print(f"❌ API not healthy: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ API not accessible: {e}")
        return

    # Test 2: Simple traditional research (no URL)
    print("\n2. Testing traditional research...")
    payload = {
        "query": "Recent developments in artificial intelligence",
        "industry": "Technology",
        "timeframe": "current",
    }

    try:
        # Start research task
        response = requests.post(
            f"{API_BASE_URL}/research/start", json=payload, timeout=10
        )

        if response.status_code == 200:
            task_data = response.json()
            task_id = task_data["task_id"]
            print(f"✅ Task started: {task_id}")

            # Monitor task progress
            max_wait = 60  # Wait up to 60 seconds
            wait_time = 0

            while wait_time < max_wait:
                time.sleep(5)
                wait_time += 5

                # Check status
                status_response = requests.get(
                    f"{API_BASE_URL}/research/status/{task_id}"
                )
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"   Status: {status_data['status']} (waited {wait_time}s)")

                    if status_data["status"] in ["completed", "completed_with_errors"]:
                        print("✅ Task completed! Getting results...")

                        # Try to get results
                        results_response = requests.get(
                            f"{API_BASE_URL}/research/results/{task_id}"
                        )
                        if results_response.status_code == 200:
                            results = results_response.json()
                            print(f"✅ Results retrieved!")
                            print(f"   Research Query: {results.get('research_query')}")
                            print(
                                f"   Sources Found: {results.get('sources_found', 0)}"
                            )
                            print(f"   Status: {results.get('status')}")

                            if results.get("final_report"):
                                print(f"\n📄 Report Preview:")
                                preview = results["final_report"][:300]
                                print(f"   {preview}...")

                            return True
                        else:
                            print(
                                f"⚠️ Results not ready: {results_response.status_code}"
                            )
                            continue

                    elif status_data["status"] == "failed":
                        print(f"❌ Task failed")
                        if status_data.get("errors"):
                            print(f"   Errors: {status_data['errors']}")
                        return False

                else:
                    print(f"⚠️ Could not check status: {status_response.status_code}")

            print(f"⏰ Task timed out after {max_wait} seconds")
            return False

        else:
            print(f"❌ Failed to start task: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_system()
    print("\n" + "=" * 50)
    if success:
        print("🎉 System test PASSED! The hybrid research system is working.")
    else:
        print("❌ System test FAILED. Check the logs above for details.")
