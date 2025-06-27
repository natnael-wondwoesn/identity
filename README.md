# Market Research Analysis System

A comprehensive, compliant market research system built with LangGraph, FastAPI, and Streamlit.

## Quick Start

1. **Setup the project:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
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
