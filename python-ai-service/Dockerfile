# Use a lightweight Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 7860

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Create directory for logs and models (ensure write permissions for Hugging Face)
RUN mkdir -p logs evaluation static && chmod -R 777 logs evaluation static

# Pre-download models during build to speed up startup
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline; \
    pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest'); \
    pipeline('summarization', model='facebook/bart-large-cnn')"

# Expose the port Hugging Face expects
EXPOSE 7860

# Start the server
CMD ["python", "ai_server.py"]
