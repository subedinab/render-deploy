from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from typing import Optional
from controller.process_query import GroqService

# from controller.process_query import process_query_controller
# from models.schema import QueryParams
router = APIRouter()
import asyncio

groq_service = GroqService()

async def test_streamming_response(message:str):
    array=list(message.split(" "))
    for chunk in array:
        yield chunk+' ' or ''
        await asyncio.sleep(0.2)

'''
http://localhost:8000/llm/test?stream=true        
testing routes
'''
@router.get('/test')
async def test_app(stream:bool):
    message="This application is running, you can proceed further with api."
    if stream:
         return StreamingResponse(test_streamming_response(message), media_type="text/event-stream")
    return {"message":message}

'''
http://localhost:8000/llm/response?query=hi&sender=restart&source=web&language=hindi&stream=false
response routes
'''
@router.get('/response')
async def process_query(query:str, sender:str, language:Optional[str]=None, source:Optional[str]=None, stream:bool=False):
    try:
        response = await groq_service.get_response(query)
        if stream:
            return StreamingResponse(test_streamming_response(response), media_type="text/event-stream")
        return {"result": response}
    except Exception as e:
        return {"error": str(e)}  



