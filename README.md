
# Intelligent Career Strategist (SDG 8)

Streamlit app that:

1. Reads a PDF resume
2. Finds 3 missing keywords vs a job description
3. Rewrites the resume summary and generates interview questions

## Setup

Install dependencies:

`uv sync`

Set environment variables for Groq:

- `GROQ_API_KEY`
- `GROQ_MODEL_ID` (optional, default: `llama-3.1-70b-versatile`)
- `GROQ_TEMPERATURE` (optional, default: `0.2`)
- `GROQ_MAX_TOKENS` (optional, default: `700`)

You can also put them in a `.env` file.

## Run

`uv run streamlit run main.py`

