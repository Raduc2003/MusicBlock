# Use an official Python runtime as a parent image
# Using slim-buster for a smaller image size
FROM python:3.9-slim-buster

# Set environment variables
# Helps prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED 1
# Sets the locale - often helpful for Python string handling
ENV LANG C.UTF-8

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if any are needed (unlikely for this client)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size
# --upgrade ensures pip is up-to-date
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY client_similarity.py .
COPY client.py .
COPY test.py .
# # Copy the normalization data file
COPY global_min_max.json .
# If client_similarity.py imported other local modules/scripts, copy them too
# COPY utils/ some_util_script.py ./utils/

# Define metadata, like maintainer (optional)
# LABEL maintainer="Your Name <youremail@example.com>"

# Since we are using docker-compose run, CMD is not strictly necessary
# but you could set a default if desired, e.g.,
CMD ["python", "client_similarity.py", "--help"]