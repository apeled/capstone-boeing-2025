# Dockerfile
FROM python:3.9-slim

# Install R and required packages
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY . .

# Make sure data and models directories exist
RUN mkdir -p data models

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "application:application"]
