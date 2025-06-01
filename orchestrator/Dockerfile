# orchestrator_service/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the main orchestrator API script into the container at /app
COPY main_orchestrator_api.py .

EXPOSE 8080

# Define environment variables that the application will use.
# These can be overridden at runtime (e.g., by docker-compose or Kubernetes).
ENV FEATURE_EXTRACTOR_API_URL="http://feature-extractor-service:8001/extract_features"
ENV SIMILARITY_API_URL="http://similarity-service:8002/find_similar"
ENV RAG_SERVICE_URL="http://rag-api-service:8000/generate_moodboard"
ENV ORCHESTRATOR_PORT="8080"
# Add other ENV vars from your main_orchestrator_api.py's os.getenv calls if needed

# Run main_orchestrator_api.py when the container launches using Uvicorn
CMD ["uvicorn", "main_orchestrator_api:app", "--host", "0.0.0.0", "--port", "8080"]