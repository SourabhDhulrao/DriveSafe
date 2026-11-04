FROM python:3.10-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by MediaPipe and OpenCV (OpenGL ES, EGL, GLib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgles2 \
    libegl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for caching efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Set default port (Render dynamically sets $PORT, default to 5000)
ENV PORT=5000
EXPOSE 5000

# Run multi-threaded Gunicorn for real-time video streaming & telemetry
CMD exec gunicorn --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:${PORT} app:app
