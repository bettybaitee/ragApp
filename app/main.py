from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.rag_engine import SimpleRAG

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
rag = SimpleRAG("data/docs.txt") 

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, query: str = Form(...), input_text: str = Form(...)):
    retrieved = rag.retrieve(query)
    response = rag.generate(input_text, retrieved)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": response,
        "query": query,
        "input_text": input_text
    })