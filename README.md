# Hybrid Web Scraping & Research System

A comprehensive hybrid system that combines intelligent web scraping with AI-powered research analysis, built with Crawl4AI, LangGraph, FastAPI, and Streamlit.

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
Edit the .env file with your LLM provider API key:
```
# For OpenAI
OPENAI_API_KEY=your-openai-key-here

# For Google Gemini (recommended)
GOOGLE_API_KEY=your-gemini-key-here

# Or use local Ollama (no API key needed)
LLM_PROVIDER=ollama
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

5. **Try the hybrid system:**
```bash
# Run example script to test the system
python example_usage.py
```

## Usage Examples

### Web Scraping + Analysis
1. Enter a target URL (e.g., company website, article, etc.)
2. Provide an analysis query (e.g., "Extract product pricing and features")
3. The system will scrape the site and analyze the content

### Hybrid Research
1. Enter both a target URL and select an industry
2. The system will scrape the URL and supplement with traditional research sources
3. Get comprehensive analysis combining scraped and researched data

### Traditional Research
1. Leave URL field empty
2. Enter your research query and select industry/timeframe
3. System uses traditional legitimate sources for research

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

## System Features

### Hybrid Research Approach
- **Web Scraping**: Uses Crawl4AI for intelligent content extraction from any website
- **Traditional Research**: Leverages legitimate public data sources
- **AI-Powered Analysis**: Combines scraped data with additional research for comprehensive insights

### AI Agents
1. **Controller Agent** - Orchestrates the entire workflow
2. **Hybrid Research Agent** - Scrapes target URLs and conducts traditional research  
3. **Compliance Agent** - Ensures ethical compliance for all data sources
4. **Analysis Agent** - Analyzes scraped content and supplementary data
5. **Strategy Agent** - Generates actionable insights and recommendations

### Key Capabilities
- Scrape and analyze any publicly accessible website
- Extract structured data using AI-guided extraction strategies
- Cross-reference scraped data with traditional research sources
- Generate comprehensive reports combining multiple data sources
- Ensure compliance with ethical data collection practices

## Requirements

- Python 3.9+
- LLM Provider API key (OpenAI, Google Gemini, or local Ollama)
- 4GB RAM minimum
- Chrome/Chromium browser (for web scraping)
- Internet connection (for target website access)

## How It Works

1. **Enter URL & Query**: Provide a target website URL and describe what you want to analyze
2. **Intelligent Scraping**: Crawl4AI extracts relevant content from the website using AI-guided strategies
3. **Hybrid Research**: The system combines scraped data with traditional research sources
4. **AI Analysis**: Multiple AI agents analyze the combined data for insights
5. **Comprehensive Report**: Generate actionable reports with findings and recommendations

## Compliance & Ethics

This system only uses:
- Publicly accessible websites (respects robots.txt)
- Legitimate public data sources and APIs  
- Ethical data collection practices with rate limiting
- No personal data collection or privacy violations
- Transparent data sourcing and attribution

## Configuration

### LLM Provider Configuration
Edit `.env` file:

**For Google Gemini (Recommended):**
```
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash
GOOGLE_API_KEY=your-gemini-key-here
```

**For OpenAI:**
```
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=your-openai-key-here
```

**For local LLM with Ollama:**
```
LLM_PROVIDER=ollama
LLM_MODEL=llama2
OLLAMA_BASE_URL=http://localhost:11434
```

### Web Scraping Configuration
```
ENABLE_WEB_SCRAPING=true
CRAWL_HEADLESS=true
CRAWL_TIMEOUT=30
CRAWL_USER_AGENT=Hybrid-Research-Bot/1.0
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
