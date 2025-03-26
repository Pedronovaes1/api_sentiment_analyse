from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

# Define o caminho absoluto para o diretório do modelo
modelo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./modelo_sentimentos/"))

# Verifique se o arquivo pytorch_model.bin ou model.safetensors está presente no diretório
if not os.path.exists(os.path.join(modelo_path, "pytorch_model.bin")) and not os.path.exists(os.path.join(modelo_path, "model.safetensors")):
    raise FileNotFoundError(f"Arquivo 'pytorch_model.bin' ou 'model.safetensors' não encontrado no diretório {modelo_path}")

modelo = BertForSequenceClassification.from_pretrained(modelo_path)
tokenizer = BertTokenizer.from_pretrained(modelo_path)

def analise_sentimento(text: str):
    # Tokenizer o texto
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Fazer predição
    with torch.no_grad():
        outputs = modelo(**inputs)
        logits = outputs.logits
    
    # Obter a classe com maior probabilidade e retornar a string para o usuário ao invés do label
    sentimento_classe = torch.argmax(logits, dim=1).item()
    sentiments_labels={
        0: "Satisfeito",  
        1: "Agradecido", 
        2: "Confiante",  
        3: "Aliviado", 
        4: "Compreendido", 
        5: "Encantado",  
        6: "Feliz",
        7: "Calmo",  
        8: "Frustrado", 
        9: "Irritado" , 
        10: "Confuso", 
        11: "Desesperado", 
        12: "Desapontado", 
        13: "Indignado", 
        14: "Desconfortável", 
        15: "Cético",
        16: "Compreensivo",
        17: "Paciente", 
        18: "Atencioso", 
        19: "Positivo",
        20: "Solucionador",
        21: "Motivado",
        22: "Organizado", 
        23: "Desmotivado", 
        24: "Cansado",
        25: "Indiferente",
        26: "Sobrecarregado"  
    }
    return sentiments_labels.get(sentimento_classe, "Desconhecido")