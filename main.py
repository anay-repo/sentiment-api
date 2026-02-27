from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(sentence: str) -> str:
    # VADER score
    vader_score = analyzer.polarity_scores(sentence)['compound']
    
    # TextBlob score
    blob_score = TextBlob(sentence).sentiment.polarity
    
    # Average both scores
    combined = (vader_score + blob_score) / 2

    if combined >= 0.05:
        return "happy"
    elif combined <= -0.05:
        return "sad"
    else:
        return "neutral"

@app.post("/sentiment")
def batch_sentiment(request: SentimentRequest):
    results = []
    for sentence in request.sentences:
        sentiment = analyze_sentiment(sentence)
        results.append({"sentence": sentence, "sentiment": sentiment})
    return {"results": results}
