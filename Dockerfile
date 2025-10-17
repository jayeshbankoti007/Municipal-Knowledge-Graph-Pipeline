FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
COPY src/ ./src/
COPY .env .env

# Install Python dependencies and Create output directory
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_md
RUN mkdir -p output/extractions

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "src.pipeline"]