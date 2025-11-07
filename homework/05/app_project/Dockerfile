FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy the FastAPI application
COPY fastapi_test.py ./

# DON'T copy pipeline_v1.bin - use pipeline_v2.bin from base image

# Expose port
EXPOSE 8000

# Run the application
ENTRYPOINT ["uv", "run", "uvicorn", "fastapi_test:app", "--host", "0.0.0.0", "--port", "8000"]