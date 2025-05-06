import requests
import json
from typing import Dict, Any

def test_api(base_url: str) -> None:
    """Test all endpoints of the Blog Search API."""
    
    # Test 1: Health Check
    print("\n1. Testing Health Check Endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        print("✅ Health check successful!")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return

    # Test 2: Index Blog Posts
    print("\n2. Testing Blog Indexing Endpoint...")
    test_urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
    ]
    
    try:
        response = requests.post(
            f"{base_url}/index-blogs",
            json={"urls": test_urls}
        )
        response.raise_for_status()
        print("✅ Blog indexing successful!")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Blog indexing failed: {str(e)}")
        return

    # Test 3: Query Blog Posts
    print("\n3. Testing Query Endpoint...")
    test_queries = [
        "What are the different types of agent memory?",
        "What is prompt engineering?"
    ]
    
    for query in test_queries:
        try:
            print(f"\nQuerying: {query}")
            response = requests.post(
                f"{base_url}/query",
                json={"query": query}
            )
            response.raise_for_status()
            print("✅ Query successful!")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"❌ Query failed: {str(e)}")

if __name__ == "__main__":
    # Replace this with your Render deployment URL
    BASE_URL = "https://your-render-url.onrender.com"
    
    print("🚀 Starting API Tests...")
    print(f"Testing API at: {BASE_URL}")
    
    test_api(BASE_URL) 