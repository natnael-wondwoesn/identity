#!/usr/bin/env python3
"""
Windows-compatible setup for Market Research Analysis System
"""

import os
from pathlib import Path


def create_startup_script():
    """Create startup script without Unicode characters"""
    content = """#!/bin/bash

# Market Research System Startup Script
set -e

# Functions
print_status() { echo "[INFO] $1"; }
print_success() { echo "[SUCCESS] $1"; }
print_error() { echo "[ERROR] $1"; }

check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1
    else
        return 0
    fi
}

wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            print_success "$service_name is ready!"
            return 0
        fi
        print_status "Attempt $attempt/$max_attempts - $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start"
    return 1
}

create_directories() {
    print_status "Creating directories..."
    mkdir -p logs data .streamlit
}

start_api() {
    print_status "Starting FastAPI backend..."
    
    if ! check_port 8000; then
        print_error "Port 8000 is already in use"
        exit 1
    fi
    
    export PYTHONPATH="${PWD}/backend:${PYTHONPATH}"
    cd backend
    nohup uvicorn main:app --host 0.0.0.0 --port 8000 --reload > ../logs/api.log 2>&1 &
    API_PID=$!
    cd ..
    
    if wait_for_service "http://localhost:8000/health" "FastAPI backend"; then
        print_success "FastAPI backend started (PID: $API_PID)"
        echo $API_PID > .api.pid
    else
        kill $API_PID 2>/dev/null || true
        exit 1
    fi
}

start_frontend() {
    print_status "Starting Streamlit frontend..."
    
    if ! check_port 8501; then
        print_error "Port 8501 is already in use"
        exit 1
    fi
    
    cd frontend
    nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..
    
    if wait_for_service "http://localhost:8501/_stcore/health" "Streamlit frontend"; then
        print_success "Streamlit frontend started (PID: $FRONTEND_PID)"
        echo $FRONTEND_PID > .frontend.pid
    else
        kill $FRONTEND_PID 2>/dev/null || true
        exit 1
    fi
}

stop_services() {
    print_status "Stopping services..."
    
    if [ -f .api.pid ]; then
        kill $(cat .api.pid) 2>/dev/null || true
        rm -f .api.pid
    fi
    
    if [ -f .frontend.pid ]; then
        kill $(cat .frontend.pid) 2>/dev/null || true
        rm -f .frontend.pid
    fi
}

case "${1:-start}" in
    "start")
        print_status "Starting Market Research Analysis System..."
        
        trap 'stop_services; exit 1' INT TERM
        
        create_directories
        
        if [ ! -d "venv" ]; then
            print_status "Creating virtual environment..."
            python3 -m venv venv
        fi
        
        source venv/bin/activate
        
        if [ ! -f ".env" ]; then
            print_error ".env file not found. Please copy .env.example to .env and configure it."
            exit 1
        fi
        
        print_status "Installing requirements..."
        pip install -r requirements.txt
        
        start_api
        start_frontend
        
        print_success "System is running!"
        print_status "Frontend: http://localhost:8501"
        print_status "API: http://localhost:8000/docs"
        print_status "Press Ctrl+C to stop"
        
        while true; do sleep 30; done
        ;;
        
    "stop")
        stop_services
        ;;
        
    "setup")
        print_status "Setting up virtual environment..."
        python3 -m venv venv
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
        else
            source venv/Scripts/activate
        fi
        pip install -r requirements.txt
        print_success "Setup completed!"
        ;;
        
    *)
        echo "Usage: $0 {start|stop|setup}"
        ;;
esac
"""

    try:
        with open("start.sh", "w", encoding="utf-8") as f:
            f.write(content)
        os.chmod("start.sh", 0o755)
        print("Created start.sh")
    except Exception as e:
        print(f"Note: Could not create start.sh: {e}")


def create_readme():
    """Create README.md without Unicode characters"""
    content = """# Market Research Analysis System

A comprehensive, compliant market research system built with LangGraph, FastAPI, and Streamlit.

## Quick Start

1. **Setup the project:**
```bash
python -m venv venv
venv\\Scripts\\activate  # Windows
# OR
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

2. **Configure environment:**
Edit the .env file with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

3. **Start the system:**

**Windows:**
```bash
# Terminal 1 - API
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Frontend  
cd frontend
streamlit run app.py --server.port 8501
```

**Linux/Mac:**
```bash
./start.sh start
```

4. **Access the application:**
- Web Interface: http://localhost:8501
- API Documentation: http://localhost:8000/docs

## Project Structure

```
market_research_system/
├── backend/           # FastAPI backend
│   ├── agents/       # AI agents
│   ├── models/       # Pydantic models
│   ├── services/     # Business logic
│   └── main.py       # FastAPI app
├── frontend/         # Streamlit frontend
│   └── app.py        # Streamlit app
├── graph/           # LangGraph workflow
│   └── workflow.py   # Workflow definition
├── tests/           # Test files
└── data/            # Database and logs
```

## AI Agents

1. **Controller Agent** - Orchestrates workflow
2. **Research Agent** - Finds legitimate data sources  
3. **Compliance Agent** - Ensures ethical compliance
4. **Analysis Agent** - Analyzes market trends
5. **Strategy Agent** - Generates insights

## Requirements

- Python 3.9+
- OpenAI API key (or local Ollama)
- 4GB RAM minimum

## Compliance

This system only uses:
- Publicly available business data
- Official APIs and legitimate sources  
- Ethical data collection practices
- No personal data collection

## Configuration

Edit `.env` file:
```
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=your-api-key-here
```

For local LLM with Ollama:
```
LLM_PROVIDER=ollama
LLM_MODEL=llama2
OLLAMA_BASE_URL=http://localhost:11434
```

## Troubleshooting

**Port already in use:**
- Windows: `netstat -ano | findstr :8000` then `taskkill /PID <PID> /F`
- Linux/Mac: `lsof -ti:8000 | xargs kill -9`

**Import errors:**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

**API connection errors:**
- Check if backend is running on port 8000
- Verify .env configuration
"""

    try:
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(content)
        print("Created README.md")
    except Exception as e:
        print(f"Note: Could not create README.md: {e}")


def main():
    """Main setup function for Windows"""
    print("Setting up Market Research Analysis System...")
    print("=" * 50)

    try:
        create_startup_script()
        create_readme()

        print("=" * 50)
        print("Project setup completed successfully!")
        print()
        print("Next steps:")
        print("1. Copy the remaining files from the artifacts")
        print("2. Edit .env file with your OpenAI API key")
        print("3. Run: python -m venv venv")
        print("4. Run: venv\\Scripts\\activate")
        print("5. Run: pip install -r requirements.txt")
        print()
        print("To start the system:")
        print("Terminal 1: cd backend && python -m uvicorn main:app --reload")
        print("Terminal 2: cd frontend && streamlit run app.py")
        print()
        print("Your system will be available at:")
        print("- Frontend: http://localhost:8501")
        print("- API: http://localhost:8000/docs")

    except Exception as e:
        print(f"Error during setup: {str(e)}")


if __name__ == "__main__":
    main()
