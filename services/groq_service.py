from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
import os

class GroqService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name="mixtral-8x7b-32768"
        )
        
        self.system_prompt = """You are a helpful AI assistant. Provide clear, concise, and accurate responses to user queries.
        If you don't know something, be honest about it. Always maintain a professional and friendly tone."""
    
    async def get_response(self, query: str) -> str:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        return response.content 