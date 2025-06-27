#!/usr/bin/env python3
"""
Windows-Compatible System Diagnostic and Fix Script
Fixed encoding issues for Windows
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("   X Python 3.9+ required")
        return False
    else:
        print("   + Python version OK")
        return True


def check_environment_file():
    """Check and fix .env file"""
    print("\nChecking .env file...")

    env_path = Path(".env")
    env_example_path = Path(".env.example")

    if not env_path.exists():
        print("   X .env file not found")
        if env_example_path.exists():
            print("   > Copying .env.example to .env")
            import shutil

            shutil.copy(env_example_path, env_path)
        else:
            print("   > Creating basic .env file")
            create_basic_env()

    # Check for API key
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            env_content = f.read()

        # Find API key lines
        api_key_lines = [
            line
            for line in env_content.split("\n")
            if line.startswith(("GOOGLE_API_KEY=", "GEMINI_API_KEY="))
        ]

        valid_key_found = False
        for line in api_key_lines:
            if "=" in line:
                key_value = line.split("=", 1)[1].strip()
                if key_value and key_value not in [
                    "your-gemini-api-key-here",
                    "your-api-key-here",
                ]:
                    valid_key_found = True
                    break

        if valid_key_found:
            print("   + Gemini API key found")
            # Test if key looks valid (should start with AIza)
            if key_value.startswith("AIza"):
                print("   + API key format looks correct")
            else:
                print("   ! API key format may be incorrect (should start with 'AIza')")
        else:
            print("   ! No valid Gemini API key found")
            print("     Please add your key: GOOGLE_API_KEY=your-actual-key-here")
            print("     Get your key from: https://aistudio.google.com/app/apikey")
            return False

    except Exception as e:
        print(f"   X Error reading .env: {e}")
        return False

    return True


def create_basic_env():
    """Create basic .env file with fixed settings"""
    content = """# Fixed Configuration for Market Research System
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
LLM_TEMPERATURE=0.1

# Add your Gemini API key here - Get it from: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your-gemini-api-key-here

# API Configuration  
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=sqlite:///./data/market_research.db

# Logging
LOG_LEVEL=INFO
"""

    with open(".env", "w", encoding="utf-8") as f:
        f.write(content)
    print("   + Created basic .env file")


def check_dependencies():
    """Check Python dependencies"""
    print("\nChecking dependencies...")

    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        print("   X requirements.txt not found")
        return False

    # Check key packages
    key_packages = [
        "fastapi",
        "streamlit",
        "langgraph",
        "google-generativeai",
        "pydantic",
    ]

    missing = []
    for package in key_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   + {package}")
        except ImportError:
            print(f"   X {package} missing")
            missing.append(package)

    if missing:
        print(f"\n   > Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("   + Dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"   X Failed to install dependencies: {e}")
            return False

    return True


def check_file_structure():
    """Check project file structure"""
    print("\nChecking file structure...")

    required_files = [
        "backend/main.py",
        "backend/config.py",
        "graph/workflow.py",
        "frontend/app.py",
        "requirements.txt",
    ]

    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   + {file_path}")
        else:
            print(f"   X {file_path} missing")
            missing.append(file_path)

    if missing:
        print("   ! Some files are missing - system may not work correctly")
        return False

    return True


def test_gemini_connection():
    """Test Gemini API connection"""
    print("\nTesting Gemini connection...")

    # Load API key from .env
    api_key = None
    try:
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(("GOOGLE_API_KEY=", "GEMINI_API_KEY=")):
                    api_key = line.split("=", 1)[1].strip()
                    if api_key and api_key not in [
                        "your-gemini-api-key-here",
                        "your-api-key-here",
                    ]:
                        break
    except:
        pass

    if not api_key or api_key in ["your-gemini-api-key-here", "your-api-key-here"]:
        print("   ! No valid API key found - skipping test")
        print("     Get your key from: https://aistudio.google.com/app/apikey")
        print("     Add it to .env file: GOOGLE_API_KEY=your-actual-key")
        return False

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        # Test with simple prompt
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("Hello, respond with 'API working'")

        if response and response.text:
            print("   + Gemini API working")
            print(f"     Response: {response.text[:50]}...")
            return True
        else:
            print("   X Gemini API returned empty response")
            return False

    except Exception as e:
        error_str = str(e)
        print(f"   X Gemini API error: {error_str}")

        if "API_KEY_INVALID" in error_str:
            print("     > API key is invalid")
            print("     > Get a new key from: https://aistudio.google.com/app/apikey")
            print("     > Make sure to enable the Generative AI API")
        elif "quota" in error_str.lower():
            print("     > API quota exceeded - check your usage")
        elif "billing" in error_str.lower():
            print("     > Billing issue - check your Google Cloud billing")

        return False


def check_ports():
    """Check if required ports are available"""
    print("\nChecking ports...")

    import socket

    ports_to_check = [8000, 8501]

    for port in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()

        if result == 0:
            print(f"   ! Port {port} is in use")
        else:
            print(f"   + Port {port} available")


def create_directories():
    """Create required directories"""
    print("\nCreating directories...")

    dirs = ["data", "logs", ".streamlit"]

    for dir_name in dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   + Created {dir_name}/")
        else:
            print(f"   + {dir_name}/ exists")


def fix_import_issues():
    """Fix common import issues"""
    print("\nFixing import issues...")

    # Add current directory to Python path
    current_dir = str(Path.cwd())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print("   + Added current directory to Python path")

    # Set PYTHONPATH environment variable
    pythonpath = os.environ.get("PYTHONPATH", "")
    if current_dir not in pythonpath:
        os.environ["PYTHONPATH"] = f"{current_dir};{pythonpath}"
        print("   + Set PYTHONPATH environment variable")


def test_backend_import():
    """Test backend imports"""
    print("\nTesting backend imports...")

    try:
        # Test main imports
        from backend.config import get_settings

        print("   + backend.config import OK")

        from graph.workflow import create_workflow

        print("   + graph.workflow import OK")

        # Test workflow creation
        llm_config = {
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "temperature": 0.1,
        }
        workflow = create_workflow(llm_config)
        print("   + Workflow creation OK")

        return True

    except Exception as e:
        print(f"   X Import error: {e}")
        return False


def generate_test_script():
    """Generate a simple test script with Windows-compatible encoding"""
    print("\nGenerating test script...")

    test_script = '''#!/usr/bin/env python3
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
    print(f"\\n{'System test PASSED!' if success else 'System test FAILED!'}")
'''

    try:
        with open("test_fixed_system.py", "w", encoding="utf-8") as f:
            f.write(test_script)
        print("   + Created test_fixed_system.py")
        return True
    except Exception as e:
        print(f"   X Failed to create test script: {e}")
        return False


def fix_api_key_instructions():
    """Provide detailed API key setup instructions"""
    print("\n" + "=" * 60)
    print("API KEY SETUP INSTRUCTIONS")
    print("=" * 60)
    print("Your Gemini API key is invalid or missing. Here's how to fix it:")
    print()
    print("1. Go to: https://aistudio.google.com/app/apikey")
    print("2. Click 'Create API Key'")
    print("3. Copy the key (starts with 'AIza')")
    print("4. Open your .env file and replace:")
    print("   GOOGLE_API_KEY=your-gemini-api-key-here")
    print("   with:")
    print("   GOOGLE_API_KEY=AIza... (your actual key)")
    print()
    print("5. Save the .env file and run this diagnostic again")
    print("=" * 60)


def main():
    """Run complete diagnostic"""
    print("Market Research System Diagnostic & Fix")
    print("=" * 50)

    checks = [
        ("Python Version", check_python_version),
        ("Environment File", check_environment_file),
        ("Dependencies", check_dependencies),
        ("File Structure", check_file_structure),
        ("Ports", check_ports),
        ("Directories", create_directories),
        ("Import Fixes", fix_import_issues),
        ("Backend Imports", test_backend_import),
        ("Gemini Connection", test_gemini_connection),
        ("Test Script", generate_test_script),
    ]

    passed = 0
    total = len(checks)
    api_key_issue = False

    for name, check_func in checks:
        try:
            if check_func():
                passed += 1
            elif name == "Gemini Connection":
                api_key_issue = True
        except Exception as e:
            print(f"   X {name} check failed: {e}")

    print(f"\nDiagnostic Summary: {passed}/{total} checks passed")

    # Provide next steps
    print("\nNext Steps:")
    if api_key_issue:
        fix_api_key_instructions()
    elif passed >= total - 1:  # Allow 1 failure
        print("1. + System looks good! Try running:")
        print("   python test_fixed_system.py")
        print("2. Start the backend:")
        print(
            "   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"
        )
        print("3. Start the frontend:")
        print("   cd frontend && streamlit run app.py")
    else:
        print("1. X Fix the issues above first")
        print("2. Make sure you have:")
        print("   - Python 3.9+")
        print("   - All required files from the artifacts")
        print("   - Valid Gemini API key in .env file")
        print("3. Run this diagnostic again:")
        print("   python fix.py")

    print("\nKey Fixes Applied:")
    print("- Fixed Gemini 'Unknown field for Part' error")
    print("- Updated to working model (gemini-1.5-flash)")
    print("- Improved error handling in workflow")
    print("- Fixed Windows encoding issues")

    print("\nDocumentation:")
    print("- Frontend: http://localhost:8501")
    print("- API docs: http://localhost:8000/docs")
    print("- Health check: http://localhost:8000/health")


if __name__ == "__main__":
    main()
