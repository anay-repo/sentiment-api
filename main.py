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

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the analyzer once outside the function for better performance
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(sentence: str) -> str:
    # VADER returns a 'compound' score between -1 and 1
    vs = analyzer.polarity_scores(sentence)
    compound = vs['compound']
    
    # Using a very low threshold to capture subtle sentiments
    if compound >= 0.05:
        return "happy"
    elif compound <= -0.05:
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

