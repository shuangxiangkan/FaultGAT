FROM python:3.10.12-slim

LABEL maintainer="Shuangxiang Kan <kansx@example.com>"
LABEL description="FaultGAT: Graph Attention-Based Approach for Intermittent Fault Diagnosis"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create directories for data and results
RUN mkdir -p /app/datasets /app/results_RQ1 /app/results_RQ2 /app/results_RQ3

# Verify FaultGAT can be imported
RUN python -c "from models import FaultGAT; print('✅ FaultGAT imported successfully')"

# Default command
CMD ["python", "-c", "print('🚀 FaultGAT Docker environment is ready!'); print('📖 Available commands:'); print('  python RQ1.py  # Theoretical diagnosability'); print('  python RQ2.py  # Fault ratio comparison'); print('  python RQ3.py  # Partial symptom analysis'); print('🔗 For interactive shell: docker run -it ksx/faultgat bash')"] 