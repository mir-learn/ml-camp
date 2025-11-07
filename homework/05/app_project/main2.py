# ML Zoomcamp - Deployment Homework Solution
# Python 3.12/3.13 recommended

# ============================================================================
# QUESTION 1: Install uv
# ============================================================================
# Run in terminal:
# pip install uv
# uv --version
#
# Initialize project:
# mkdir homework
# cd homework
# uv init

# ============================================================================
# QUESTION 2: Install Scikit-Learn 1.6.1
# ============================================================================
# Run in terminal:
# uv add scikit-learn==1.6.1
#
# Check pyproject.lock file for the first hash starting with sha256:

# ============================================================================
# QUESTION 3: Load and Score with Pickle
# ============================================================================

import pickle
import requests


def load_pipeline(filename='pipeline_v1.bin'):
    """Load the pickled pipeline"""
    with open(filename, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

def score_lead_q3():
    """Question 3: Score a single lead"""
    # Load the pipeline
    pipeline = load_pipeline('pipeline_v1.bin')

    # Client data for Question 3
    client = {
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0
    }

    # Score the client
    probability = pipeline.predict_proba([client])[0, 1]

    print(f"Question 3 - Probability of conversion: {probability:.3f}")
    return probability

# ============================================================================
# QUESTION 4: FastAPI Web Service
# ============================================================================

# File: app.py (save this as a separate file)
"""
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
"""

def test_fastapi_service_q4(url="http://localhost:8000/predict"):
    """Question 4: Test the FastAPI service"""
    client = {
        "lead_source": "organic_search",
        "number_of_courses_viewed": 4,
        "annual_income": 80304.0
    }

    response = requests.post(url, json=client)
    result = response.json()

    print(f"Question 4 - Probability: {result['probability']:.3f}")
    return result

# ============================================================================
# QUESTION 5: Docker Image Size
# ============================================================================
# Run in terminal:
# docker pull agrigorev/zoomcamp-model:2025
# docker images
# Look for the SIZE column for agrigorev/zoomcamp-model:2025

# ============================================================================
# QUESTION 6: Docker Container with FastAPI
# ============================================================================

# File: Dockerfile
"""
FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

# Copy requirements
COPY pyproject.toml .

# Install uv and dependencies
RUN pip install uv && \
    uv pip install --system fastapi uvicorn

# Copy the FastAPI application
COPY app.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# File: app_docker.py (FastAPI app for Docker with pipeline_v2.bin)
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the pipeline_v2.bin that comes with the base image
with open('pipeline_v2.bin', 'rb') as f:
    pipeline = pickle.load(f)

app = FastAPI()

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(lead: Lead):
    lead_dict = lead.model_dump()
    probability = pipeline.predict_proba([lead_dict])[0, 1]

    return {
        "probability": float(probability),
        "converted": bool(probability >= 0.5)
    }

@app.get("/")
def root():
    return {"message": "Lead Scoring Model API - Docker Version"}
"""

def test_docker_service_q6(url="http://localhost:8000/predict"):
    """Question 6: Test the Docker service"""
    client = {
        "lead_source": "organic_search",
        "number_of_courses_viewed": 4,
        "annual_income": 80304.0
    }

    response = requests.post(url, json=client)
    result = response.json()

    print(f"Question 6 - Probability: {result['probability']:.3f}")
    return result

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def download_pipeline():
    """Download the pipeline file"""
    import urllib.request

    url = "https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/refs/heads/master/cohorts/2025/05-deployment/pipeline_v1.bin"
    filename = "pipeline_v1.bin"

    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete!")

def verify_checksum():
    """Verify the MD5 checksum of the pipeline"""
    import hashlib

    with open('pipeline_v1.bin', 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    expected = "7d17d2e4dfbaf1e408e1a62e6e880d49"

    if file_hash == expected:
        print(f"✓ Checksum verified: {file_hash}")
    else:
        print(f"✗ Checksum mismatch!")
        print(f"  Expected: {expected}")
        print(f"  Got:      {file_hash}")

    return file_hash == expected

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ML Zoomcamp - Deployment Homework Solutions")
    print("=" * 70)

    # Download pipeline if not exists
    import os
    if not os.path.exists('pipeline_v1.bin'):
        print("\nDownloading pipeline...")
        download_pipeline()

    # Verify checksum
    print("\nVerifying pipeline checksum...")
    verify_checksum()

    # Question 3
    print("\n" + "=" * 70)
    print("QUESTION 3")
    print("=" * 70)
    score_lead_q3()

    # Question 4 - Instructions
    print("\n" + "=" * 70)
    print("QUESTION 4")
    print("=" * 70)
    print("To run the FastAPI service:")
    print("1. Create app.py with the FastAPI code above")
    print("2. Install FastAPI: uv add fastapi uvicorn")
    print("3. Run: uvicorn app:app --reload")
    print("4. Test with test_fastapi_service_q4()")

    # Question 5 - Instructions
    print("\n" + "=" * 70)
    print("QUESTION 5")
    print("=" * 70)
    print("Run in terminal:")
    print("  docker pull agrigorev/zoomcamp-model:2025")
    print("  docker images")

    # Question 6 - Instructions
    print("\n" + "=" * 70)
    print("QUESTION 6")
    print("=" * 70)
    print("1. Create Dockerfile with the content above")
    print("2. Build: docker build -t lead-scoring-app .")
    print("3. Run: docker run -p 8000:8000 lead-scoring-app")
    print("4. Test with test_docker_service_q6()")
    print("=" * 70)