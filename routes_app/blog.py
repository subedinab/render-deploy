from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from uuid import uuid4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from functools import partial
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel as PydanticBaseModel, Field
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Blog Search API", description="API for searching and querying blog posts")

# Set USER_AGENT if not already set
if 'USER_AGENT' not in os.environ or not os.environ.get('USER_AGENT'):
    os.environ['USER_AGENT'] = 'BlogSearchAgent/1.0'

# Initialize global variables
qdrant_host = os.getenv("QDRANT_HOST")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
indexed_urls = []

class BlogURLs(BaseModel):
    urls: List[str]

class Query(BaseModel):
    query: str

def initialize_components():
    """Initialize components requiring API keys."""
    if not all([qdrant_host, qdrant_api_key, gemini_api_key]):
        raise HTTPException(status_code=500, detail="API keys not configured")
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
        client = QdrantClient(
            qdrant_host,
            api_key=qdrant_api_key
        )
        collection_name = "blog_posts"
        try:
            client.get_collection(collection_name)
        except:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        db = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model
        )
        return embedding_model, client, db
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization error: {str(e)}")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """Determine if retrieved documents are relevant."""
    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score: 'yes' or 'no'")
    model = ChatGoogleGenerativeAI(api_key=gemini_api_key, temperature=0, model="gemini-1.5-flash")
    llm_with_tool = model.with_structured_output(Grade)
    prompt = PromptTemplate(
        template="""Assess the relevance of the retrieved document to the user question.
        Document: {context}
        Question: {question}
        Return 'yes' if the document is relevant, 'no' otherwise.""",
        input_variables=["context", "question"]
    )
    chain = prompt | llm_with_tool
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content
    score = chain.invoke({"question": question, "context": docs}).binary_score
    return "generate" if score == "yes" else "rewrite"

def agent(state, tools):
    """Invoke agent model to generate response."""
    messages = state["messages"]
    model = ChatGoogleGenerativeAI(api_key=gemini_api_key, temperature=0, model="gemini-1.5-flash")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}

def rewrite(state):
    """Transform query to improve it."""
    messages = state["messages"]
    question = messages[0].content
    msg = [HumanMessage(
        content=f"""Refine this question to make it clearer and more specific:
        {question}"""
    )]
    model = ChatGoogleGenerativeAI(api_key=gemini_api_key, temperature=0, model="gemini-1.5-flash")
    response = model.invoke(msg)
    return {"messages": [response]}

def generate(state):
    """Generate answer from retrieved documents."""
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content
    rag_prompt = PromptTemplate.from_template(
        """Answer the question using only the provided context from blog posts.
        Context: {context}
        Question: {question}
        Instructions:
        1. Use only the provided context.
        2. If the context lacks the answer, state: "The blog posts do not provide enough information to answer this question."
        3. Do not use external knowledge.
        Answer:"""
    )
    chat_model = ChatGoogleGenerativeAI(api_key=gemini_api_key, model="gemini-1.5-flash", temperature=0)
    rag_chain = rag_prompt | chat_model | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [HumanMessage(content=response)]}

def get_graph(retriever_tool):
    """Create the LangGraph workflow."""
    tools = [retriever_tool]
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", partial(agent, tools=tools))
    workflow.add_node("retrieve", ToolNode(tools))
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")
    return workflow.compile()

@app.post("/index-blogs")
async def index_blogs(blog_urls: BlogURLs):
    """Endpoint to index blog posts."""
    try:
        embedding_model, client, db = initialize_components()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200
        )
        indexed_count = 0
        for url in blog_urls.urls:
            if url in indexed_urls:
                continue
            loader = WebBaseLoader(url)
            docs = loader.load()
            if not docs:
                continue
            doc_chunks = text_splitter.split_documents(docs)
            for chunk in doc_chunks:
                chunk.metadata = chunk.metadata or {}
                chunk.metadata["source"] = url
            uuids = [str(uuid4()) for _ in doc_chunks]
            db.add_documents(documents=doc_chunks, ids=uuids)
            indexed_urls.append(url)
            indexed_count += len(doc_chunks)
        return {"message": f"Successfully indexed {indexed_count} chunks from {len(blog_urls.urls)} URLs"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_blogs(query: Query):
    """Endpoint to query indexed blog posts."""
    try:
        if not indexed_urls:
            raise HTTPException(status_code=400, detail="No blog posts indexed yet")
        
        embedding_model, client, db = initialize_components()
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_blog_posts",
            "Search for information in indexed blog posts."
        )
        
        inputs = {"messages": [HumanMessage(content=query.query)]}
        graph = get_graph(retriever_tool)
        response = graph.invoke(inputs)
        final_message = response["messages"][-1].content
        
        return {"response": final_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}