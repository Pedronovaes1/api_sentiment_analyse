# Arquivo para definir os modelos Pydantic (entrada e saída de dados)
from pydantic import BaseModel

# Define o formato de entrada da requisição
class SentimentRequest(BaseModel):
    text: str 

# Define o formato da resposta que a API vai retornar
class SentimentResponse(BaseModel):
    sentimet: str