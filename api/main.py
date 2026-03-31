from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add the project root to sys.path so we can import 'rag'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.pipeline import RAGPipeline

app = FastAPI(
    title="RAG-Cratoss API",
    description="Query existing IoT documents via a RAG pipeline",
    version="1.0.0"
)

# Global pipeline instance (initialized at startup)
pipeline = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.on_event("startup")
async def startup_event():
    global pipeline
    print("\n" + "="*40)
    print("🔧 INITIALIZING RAG PIPELINE (FASTAPI)")
    print("="*40)
    try:
        pipeline = RAGPipeline()
        print("\n✅ API READY!")
    except Exception as e:
        print(f"\n❌ FATAL: Pipeline initialization failed: {e}")
        raise e

@app.get("/")
def health_check():
    return {
        "status": "ready" if pipeline else "initializing",
        "message": "RAG-Cratoss API is running locally"
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline is still initializing...")
    
    try:
        print(f"\n📩 API Query received: {request.question}")
        response = pipeline.query(request.question)
        
        # The user requested ONLY the LLM output in the response
        return QueryResponse(answer=response.answer)
    except Exception as e:
        print(f"❌ Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # If run directly: python api/main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
