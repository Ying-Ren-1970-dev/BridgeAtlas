# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF and other libraries
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    mupdf-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional cloud dependencies
RUN pip install --no-cache-dir \
    google-cloud-storage==2.14.0 \
    google-cloud-firestore==2.14.0

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p Projects data vector_db

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)"

# Run the application
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT} --workers 1
