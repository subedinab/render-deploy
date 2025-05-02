from fastapi import FastAPI,Request

from routes_app import routes

import time
import uvicorn
from dotenv import load_dotenv
import os
app = FastAPI()
load_dotenv()
PORT= "10000"

@app.get("/")
async def root():
    return {"message": "Connection Successful! Server is running."}

#check for response handeling time
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# register routes 
app.include_router(
    router=routes.router,
    prefix='/llm',
    responses={404: {'description': 'Not found'}},
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
