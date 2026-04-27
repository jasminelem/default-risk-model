# Dockerfile for Pulse Credit Risk Backend (FastAPI + ML Models)
# Optimized for deployment on Render, Fly.io, Railway, etc.

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for pandas, pyarrow, shap, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project (models, data, app, src, etc.)
# Note: .dockerignore will exclude unnecessary files
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Environment variables (override these in your hosting platform)
ENV PYTHONUNBUFFERED=1 \
    PORT=8000

# Run the FastAPI app with Uvicorn
# Use --host 0.0.0.0 so it's accessible outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
