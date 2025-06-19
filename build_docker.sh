#!/bin/bash

# FaultGAT Docker Build and Push Script

set -e

IMAGE_NAME="ksx/faultgat"
VERSION="latest"

echo "🐳 Building FaultGAT Docker image..."

# Build the Docker image
docker build -t ${IMAGE_NAME}:${VERSION} .

echo "✅ Docker image built successfully: ${IMAGE_NAME}:${VERSION}"

# Test the image
echo "🧪 Testing the Docker image..."
docker run --rm ${IMAGE_NAME}:${VERSION} python -c "from models import FaultGAT; print('✅ FaultGAT test passed')"

echo "🎉 Docker image is ready!"
echo ""
echo "📋 Available commands:"
echo "  # Run interactive shell:"
echo "  docker run -it ${IMAGE_NAME}:${VERSION} bash"
echo ""
echo "  # Run experiments:"
echo "  docker run --rm -v \$(pwd)/results_RQ1:/app/results_RQ1 ${IMAGE_NAME}:${VERSION} python RQ1.py"
echo "  docker run --rm -v \$(pwd)/results_RQ2:/app/results_RQ2 ${IMAGE_NAME}:${VERSION} python RQ2.py"
echo "  docker run --rm -v \$(pwd)/results_RQ3:/app/results_RQ3 ${IMAGE_NAME}:${VERSION} python RQ3.py"
echo ""
echo "  # Or use docker-compose:"
echo "  docker-compose up faultgat"
echo "  docker-compose run rq1"
echo "  docker-compose run rq2"  
echo "  docker-compose run rq3"

# Optional: Push to Docker Hub (uncomment if needed)
# echo "🚀 Pushing to Docker Hub..."
# docker push ${IMAGE_NAME}:${VERSION}
# echo "✅ Image pushed to Docker Hub: ${IMAGE_NAME}:${VERSION}" 