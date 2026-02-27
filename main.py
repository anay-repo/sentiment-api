from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SentimentRequest(BaseModel):
    sentences: List[str]

# Load model once at startup
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def analyze_sentiment(sentence: str) -> str:
    result = classifier(sentence, truncation=True, max_length=512)[0]
    label = result["label"]
    score = result["score"]

    if label == "POSITIVE" and score > 0.6:
        return "happy"
    elif label == "NEGATIVE" and score > 0.6:
        return "sad"
    else:
        return "neutral"

@app.post("/sentiment")
def batch_sentiment(request: SentimentRequest):
    results = []
    for sentence in request.sentences:
        sentiment = analyze_sentiment(sentence)
        results.append({
            "sentence": sentence,
            "sentiment": sentiment
        })
    return {"results": results}
