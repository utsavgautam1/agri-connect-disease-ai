# Agri-Connect Disease Detection Backend

Standalone Flask service for plant disease detection using an embedded TFLite model.

## Before deploying — 2 things you MUST fill in

1. **`model/disease_model.tflite`** — copy your exported TFLite model (~4.88MB) into this file.
2. **`model/class_names.json`** — replace the placeholder with your actual 43 class labels,
   **in the exact order they were indexed during training** (i.e. the order from
   `train_generator.class_indices` or `dataset.class_names` in your Colab notebook).
   Getting this order wrong will silently give you wrong predictions with high confidence.

## Local test

```bash
pip install -r requirements.txt
python app.py
```

Test it:
```bash
curl -X POST -F "image=@/path/to/leaf.jpg" http://localhost:5000/predict
```

## Deploy to Render (new service, separate from your existing backend)

1. Push this folder to a new GitHub repo (e.g. `agri-connect-disease-ai`).
2. Go to https://dashboard.render.com → **New +** → **Web Service**.
3. Connect the GitHub repo.
4. Render should auto-detect `render.yaml`. If not, set manually:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Environment Variable:** `PYTHON_VERSION` = `3.11.9`
5. Click **Create Web Service**. Wait for the build (TensorFlow install takes a few minutes).
6. Once live, Render gives you a URL like:
   ```
   https://agri-connect-disease-ai.onrender.com
   ```
   **This is your new endpoint path.** Full predict URL:
   ```
   https://agri-connect-disease-ai.onrender.com/predict
   ```

## Update your React Native `.env`

```dotenv
DISEASE_API_URL=https://agri-connect-disease-ai.onrender.com
```

And make sure the frontend call hits `${DISEASE_API_URL}/predict` — check the disease
detection service file in your RN app (likely under `src/services/` or `src/api/`) to
confirm it's building the URL this way, since that's what caused the 404 on the old
ngrok/Railway URLs.

## Notes

- Free-tier Render services spin down after inactivity — first request after idle can
  take 20-30s (cold start). Keep your UptimeRobot keep-alive pointed at `/health` on
  this service too, same as your main backend.
- If your model expects a different input size than 224x224, update `IMG_SIZE` in `app.py`.
