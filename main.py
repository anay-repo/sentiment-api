from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import anthropic
import json

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
client = anthropic.Anthropic()

def analyze_with_claude(sentences: List[str]) -> List[str]:
    """Use Claude to analyze a batch of ambiguous sentences."""
    prompt = f"""Analyze the sentiment of each sentence below. 
For each sentence, respond with ONLY "happy", "sad", or "neutral".
- "happy" = positive, optimistic, joyful, excited, satisfied
- "sad" = negative, unhappy, frustrated, angry, disappointed, terrible
- "neutral" = factual, no clear emotion

Sentences:
{json.dumps(sentences)}

Respond with a JSON array of sentiment strings in the same order. Example: ["happy", "sad", "neutral"]
Only output the JSON array, nothing else."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = message.content[0].text.strip()
    return json.loads(result)


def analyze_sentiment_vader(sentence: str) -> tuple[str, float]:
    vs = analyzer.polarity_scores(sentence)
    compound = vs['compound']
    
    if compound >= 0.2:
        return "happy", abs(compound)
    elif compound <= -0.2:
        return "sad", abs(compound)
    else:
        return "neutral", abs(compound)  # low confidence


@app.post("/sentiment")
def batch_sentiment(request: SentimentRequest):
    results = []
    ambiguous_indices = []
    ambiguous_sentences = []

    # First pass: VADER with tighter thresholds
    for i, sentence in enumerate(request.sentences):
        sentiment, confidence = analyze_sentiment_vader(sentence)
        if confidence < 0.2:  # Low confidence -> send to Claude
            ambiguous_indices.append(i)
            ambiguous_sentences.append(sentence)
            results.append({"sentence": sentence, "sentiment": None})
        else:
            results.append({"sentence": sentence, "sentiment": sentiment})

    # Second pass: Claude for ambiguous ones (batch call)
    if ambiguous_sentences:
        claude_results = analyze_with_claude(ambiguous_sentences)
        for idx, sentiment in zip(ambiguous_indices, claude_results):
            results[idx]["sentiment"] = sentiment

    return {"results": results}
