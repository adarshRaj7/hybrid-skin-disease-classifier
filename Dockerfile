FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set environment variables for HF Spaces
ENV MODEL_CHECKPOINT_PATH=models/best_model.pt
ENV HF_MODEL_REPO_ID=adarshRaj7/skin-disease-classifier
ENV CORS_ALLOW_ORIGINS=*
ENV PORT=7860

# Expose port
EXPOSE 7860

# Run FastAPI server (uvicorn)
# The frontend will be served as static files via FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
