import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from chain import chain
from langfuse.callback import CallbackHandler

app = FastAPI()

class QuizRequest(BaseModel):
    query: str


class Quiz(BaseModel):
    question: str
    options: List[str]
    correct_answer_index: int


class QuizResponse(BaseModel):
    questions: List[Quiz]


@app.post("/", response_model=QuizResponse)
def read_root(req: QuizRequest):
    langfuse_handler = CallbackHandler(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host="https://cloud.langfuse.com",
    )

    config = {"configurable": {"session_id": "abc123"}, "callbacks": [langfuse_handler]}
    res = chain.invoke({"query": req.query}, config)
    data = [Quiz(**d) for d in res]
    return QuizResponse(**{"questions": data})
