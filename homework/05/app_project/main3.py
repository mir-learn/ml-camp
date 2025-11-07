from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the pipeline at startup
with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

app = FastAPI()

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(lead: Lead):
    # Convert to dictionary
    lead_dict = lead.model_dump()

    # Get prediction probability
    probability = pipeline.predict_proba([lead_dict])[0, 1]

    return {
        "probability": float(probability),
        "converted": bool(probability >= 0.5)
    }

@app.get("/")
def root():
    return {"message": "Lead Scoring Model API"}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000