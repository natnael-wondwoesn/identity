#!/bin/bash

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
    
    # Set Python path to project root so imports work correctly
    export PYTHONPATH="${PWD}:${PYTHONPATH}"
    
    # Run from project root, not from backend directory
    nohup uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload > logs/api.log 2>&1 &
    API_PID=$!
    
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
