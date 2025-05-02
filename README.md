# Groq LLM API

A FastAPI application that uses Groq LLM to answer user queries.

## Local Development

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

5. Run the server:

```bash
uvicorn server:app --reload
```

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn server:app`
4. Add your GROQ_API_KEY in the Environment Variables section
5. Deploy!

## API Endpoints

- Test endpoint: `GET /llm/test?stream=false`
- Query endpoint: `GET /llm/response?query=your_query&sender=user1&stream=false`

## Environment Variables

- `GROQ_API_KEY`: Your Groq API key (required)
