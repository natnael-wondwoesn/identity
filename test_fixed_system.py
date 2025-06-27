#!/usr/bin/env python3
"""
Simple test of the fixed system
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_system():
    try:
        print("Testing fixed system...")
        
        # Test imports
        from graph.workflow import create_workflow
        print("+ Imports working")
        
        # Test workflow creation
        llm_config = {
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "temperature": 0.1
        }
        workflow = create_workflow(llm_config)
        print("+ Workflow created")
        
        # Test simple research
        result = await workflow.run_research(
            research_query="Test query about market trends",
            industry="Technology"
        )
        
        print(f"+ Research completed: {result['status']}")
        if result.get('final_report'):
            print(f"Report preview: {result['final_report'][:200]}...")
        
        return True
        
    except Exception as e:
        print(f"X Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_system())
    print(f"\n{'System test PASSED!' if success else 'System test FAILED!'}")
