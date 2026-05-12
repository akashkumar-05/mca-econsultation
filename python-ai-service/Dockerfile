# Use a lightweight Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 7860
ENV HOME=/home/user

# Create a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
WORKDIR $HOME/app

# Set path
ENV PATH="/home/user/.local/bin:$PATH"

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY --chown=user . .

# Create directory for logs and models
RUN mkdir -p logs evaluation static && chmod -R 777 logs evaluation static

# Pre-download models during build
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline; \
    pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest'); \
    pipeline('summarization', model='facebook/bart-large-cnn')"

# Expose the port Hugging Face expects
EXPOSE 7860

# Start the server using the Flask port Hugging Face expects
CMD ["python", "ai_server.py"]
