from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class SentimentRequest(BaseModel):
    sentences: List[str]

@app.get("/")
def root():
    return {"message": "API is running"}

def analyze_sentiment(sentence: str) -> str:
    sentence = sentence.lower()

    positive_words = ["love", "great", "excellent", "good", "happy", "awesome"]
    negative_words = ["hate", "bad", "terrible", "sad", "angry", "worst"]

    if any(word in sentence for word in positive_words):
        return "happy"
    elif any(word in sentence for word in negative_words):
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
