# Blog Search API

This API allows you to index blog posts and query them using natural language. It uses LangChain, LangGraph, and Google's Gemini model for intelligent search and response generation.

## Features

- Index multiple blog posts from URLs
- Query indexed blog posts using natural language
- Intelligent response generation using RAG (Retrieval Augmented Generation)
- Health check endpoint

## API Endpoints

### 1. Index Blog Posts

```http
POST /index-blogs
Content-Type: application/json

{
    "urls": [
        "https://example.com/blog1",
        "https://example.com/blog2"
    ]
}
```

### 2. Query Blog Posts

```http
POST /query
Content-Type: application/json

{
    "query": "What does the blog say about AI agents?"
}
```

### 3. Health Check

```http
GET /health
```

## Environment Variables

The following environment variables need to be set:

- `QDRANT_HOST`: Your Qdrant vector database host URL
- `QDRANT_API_KEY`: Your Qdrant API key
- `GEMINI_API_KEY`: Your Google Gemini API key

## Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in a `.env` file
4. Run the application:
   ```bash
   python main.py
   ```

## Deployment

The application is configured for deployment on Render. Simply connect your repository and set the required environment variables in the Render dashboard.

## Example Usage

### Indexing Blogs

```python
import requests

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://example.com/blog"
]

response = requests.post(
    "https://your-api-url/index-blogs",
    json={"urls": urls}
)
print(response.json())
```

### Querying Blogs

```python
import requests

query = "What are the different types of agent memory?"

response = requests.post(
    "https://your-api-url/query",
    json={"query": query}
)
print(response.json())
```

## Built With

- FastAPI
- LangChain
- LangGraph
- Google Gemini
- Qdrant Vector Database
