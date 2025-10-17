#!/bin/bash

set -e  # Exit on error

echo "========================================"
echo "Municipal Knowledge Graph Pipeline"
echo "========================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Please create .env from .env.example and add your OPENAI_API_KEY"
    exit 1
fi

# Check if data directory exists
if [ ! -d "data/transcripts" ]; then
    echo "❌ Error: data/transcripts directory not found"
    echo "Please extract transcripts.zip to data/transcripts/"
    exit 1
fi

# Check if transcripts exist
TRANSCRIPT_COUNT=$(find data/transcripts -name "*.json" | wc -l)
if [ $TRANSCRIPT_COUNT -eq 0 ]; then
    echo "❌ Error: No transcript files found in data/transcripts/"
    echo "Please extract transcripts.zip to data/transcripts/"
    exit 1
fi

echo "✅ Found $TRANSCRIPT_COUNT transcript files"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker and try again"
    exit 1
fi

echo "Building Docker image..."
docker build -t municipal-kg-pipeline .

echo ""
echo "Running pipeline..."
echo ""

# Run with docker-compose if available, otherwise docker run
if command -v docker-compose &> /dev/null; then
    docker-compose up --abort-on-container-exit
else
    docker run --rm \
        -v "$(pwd)/data:/app/data:ro" \
        -v "$(pwd)/output:/app/output" \
        --env-file .env \
        municipal-kg-pipeline
fi

echo ""
echo "========================================"
echo "✅ Pipeline Complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "  - output/extractions/          (Individual transcript extractions)"
echo "  - output/knowledge_graph.pkl    (NetworkX pickle)"
echo "  - output/knowledge_graph_interactive.html (NetworkX interactive visualization)"
echo ""