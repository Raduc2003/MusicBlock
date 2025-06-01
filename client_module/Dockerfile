# Dockerfile for similarity_query_api_service
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app

# Copy the stats file into the image.
# Ensure global_mean_std_94FEATURES.json is in the same directory as this Dockerfile
# or adjust the COPY source path. This Dockerfile should be in similarity_pipeline/
COPY global_mean_std_94FEATURES.json /app/global_mean_std_94FEATURES.json

COPY requirements_similarity_query.txt .
RUN pip install --no-cache-dir -r requirements_similarity_query.txt

COPY similarity_query_api.py .
# If you have other shared utility modules from similarity_pipeline, copy them too.

EXPOSE 8002

CMD ["uvicorn", "similarity_query_api:app", "--host", "0.0.0.0", "--port", "8002"]