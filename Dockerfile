FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system deps (minimal) and pip requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

ENV PYTHONUNBUFFERED=1

# Expose web port
EXPOSE 5000

# Use gunicorn to run the Flask app (app.py must define `app`)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
