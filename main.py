from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from textblob import TextBlob

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

def analyze_sentiment(sentence: str) -> str:
    # TextBlob returns polarity between -1.0 (sad/negative) and 1.0 (happy/positive)
    analysis = TextBlob(sentence)
    
    # We use a small threshold (0.1) to define "neutral"
    if analysis.sentiment.polarity > 0.1:
        return "happy"
    elif analysis.sentiment.polarity < -0.1:
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
