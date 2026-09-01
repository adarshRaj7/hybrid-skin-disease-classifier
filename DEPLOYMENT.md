# Deployment Guide

## Local Testing (Development)

### 1. Run FastAPI Backend + Frontend

```bash
# Activate venv
source .venv/bin/activate  # On Linux/Mac
.\.venv\Scripts\Activate.ps1  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Then open: **http://localhost:8000**

The HTML frontend will load and call the API at `http://localhost:8000`.

---

## Deploy to Hugging Face Spaces (Docker)

### 1. Create a New Space on HF

- Go to **huggingface.co → Spaces → Create new Space**
- Name: `skin-disease-classifier`
- License: `apache-2.0`
- Space SDK: **Docker** (not Gradio)
- Visibility: **Public**
- Click "Create Space"

### 2. Clone the Space Locally

```bash
git clone https://huggingface.co/spaces/adarshRaj7/skin-disease-classifier
cd skin-disease-classifier
```

### 3. Copy Files from Your Project

Copy these files to the cloned Space repo:

```bash
# Copy your Dockerfile
cp ../hybrid-skin-disease-classifier/Dockerfile .

# Copy your code
cp -r ../hybrid-skin-disease-classifier/src ./
cp -r ../hybrid-skin-disease-classifier/api ./
cp -r ../hybrid-skin-disease-classifier/frontend ./
cp ../hybrid-skin-disease-classifier/requirements.txt .
cp ../hybrid-skin-disease-classifier/pyproject.toml .
cp ../hybrid-skin-disease-classifier/README.md .
```

### 4. Commit and Push to HF Space

```bash
git add .
git commit -m "Deploy skin disease classifier"
git push
```

HF Spaces will automatically build the Docker image and deploy! 🚀

**Your app will be live at:** `https://huggingface.co/spaces/adarshRaj7/skin-disease-classifier`

---

## Environment Variables (Optional)

Set these in HF Space settings if needed:

- `HF_MODEL_REPO_ID`: `adarshRaj7/skin-disease-classifier` (default)
- `MODEL_CHECKPOINT_PATH`: `models/best_model.pt` (not needed if using HF Hub)
- `CORS_ALLOW_ORIGINS`: `*` (default)
- `PORT`: `7860` (HF Spaces default)

---

## How It Works

1. **FastAPI** (`api/main.py`) runs on port 7860
2. **Frontend** (`frontend/index.html`) is served from `/` 
3. **Model** is loaded from Hugging Face Hub: `adarshRaj7/skin-disease-classifier`
4. **API endpoints**:
   - `GET /` → Serves HTML frontend
   - `POST /predict` → Prediction from uploaded image
   - `POST /predict/gradcam` → Prediction + Grad-CAM visualization
   - `GET /health` → Model status

---

## Troubleshooting

**"Model is not loaded yet"**
- Wait 30 seconds after deployment starts
- Check that `HF_MODEL_REPO_ID` is set correctly
- Verify internet connection (needs to download model from HF Hub)

**Frontend can't reach API**
- The frontend automatically detects the API URL
- On HF Spaces, it uses the same host as the frontend
- For local testing, make sure uvicorn is running on `http://localhost:8000`

**Docker build fails**
- Ensure `.dockerignore` excludes large files (data/, notebooks/, etc.)
- Check that `requirements.txt` has all dependencies
- Ensure `frontend/index.html` exists
