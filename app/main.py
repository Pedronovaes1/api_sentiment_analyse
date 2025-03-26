from fastapi import FastAPI
from app.models import analise_sentimento
from app.schemas import SentimentRequest, SentimentResponse

app = FastAPI()

@app.post("/analisar", response_model=SentimentResponse)
async def analisar_sentimentos(request: SentimentRequest):
    sentiment = analise_sentimento(request.text)
    return SentimentResponse(sentimet=sentiment)