FROM python:3.11-slim

# Install system dependencies for OpenCV/Graphics
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (dynamic on Railway)
EXPOSE 8501

# Command to run Streamlit on the dynamically assigned Railway PORT
CMD ["sh", "-c", "streamlit run dashboard.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
