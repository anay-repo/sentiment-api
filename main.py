from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
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
    sentence = sentence.lower()

    positive_words = [
        "love", "great", "excellent", "good", "happy", "awesome",
        "amazing", "fantastic", "nice", "wonderful", "best",
        "excited", "like", "enjoy", "brilliant", "positive"
    ]

    negative_words = [
        "hate", "bad", "terrible", "sad", "angry", "worst",
        "awful", "horrible", "disappointed", "poor", "boring",
        "upset", "negative", "annoying", "dislike", "pain"
    ]

    positive_score = sum(word in sentence for word in positive_words)
    negative_score = sum(word in sentence for word in negative_words)

    if positive_score > negative_score:
        return "happy"
    elif negative_score > positive_score:
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
