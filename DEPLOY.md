# Putting DermaLens online

Everything below runs in the cloud on free plans. Your computer only needs a browser.

| Part | Service | Free plan notes |
|---|---|---|
| Training | Google Colab | Free GPU sessions; needs a Google account |
| Trained model files | GitHub Releases | Free on public repos |
| API (`backend/`) | Render | Free web service; sleeps after ~15 min idle, first request then takes ~1 min |
| Website (`frontend/`) | Vercel | Free hobby plan |

## 1. Train in Colab

1. Open `ml/train_in_colab.ipynb` from GitHub in Colab: go to https://colab.research.google.com → **GitHub** tab → paste `SahilGBhuva/DermaLens`, pick the `redesign-3d-and-ml` branch and the notebook.
2. **Runtime → Change runtime type → T4 GPU**.
3. Read the dataset licence cell (HAM10000 is CC BY-NC 4.0: non-commercial, with attribution), then **Runtime → Run all**.
4. When it finishes, your browser downloads `dermalens_efficientnet_b0.pt`, `model_config.json`, `evaluation.json` and `training_history.json`. Note the `sha256` it prints for the `.pt` file.

## 2. Publish the trained files

On GitHub: **Releases → Draft a new release**, tag `model-v1`, attach `dermalens_efficientnet_b0.pt`, `model_config.json` and `evaluation.json`, publish.

`model_config.json` tells the API how the model was tested (preprocessing, flip-averaging, calibration) and holds the weights' SHA-256; the API refuses weights that don't match it.

Each attached file then has a direct link like:

```
https://github.com/SahilGBhuva/DermaLens/releases/download/model-v1/dermalens_efficientnet_b0.pt
```

## 3. Deploy the API on Render

1. https://render.com → sign in with GitHub → **New → Blueprint** → choose this repo. Render reads `render.yaml`.
2. Fill the environment variables it asks for:
   - `MODEL_URL` — the `.pt` release link from step 2
   - `CONFIG_URL` — the `model_config.json` release link
   - `EVALUATION_URL` — the `evaluation.json` release link
   - `CORS_ORIGINS` — your Vercel address from step 4 (you can come back and set it after)
   - optionally `MODEL_SHA256` — the hash Colab printed, so a corrupted download is refused
3. Deploy. When it is live, open `https://<your-api>.onrender.com/health` — it should show `"model_loaded": true`.

If the free instance runs out of memory loading PyTorch, switch the service to the Starter plan or host the API on a Hugging Face Docker Space instead.

## 4. Deploy the website on Vercel

1. https://vercel.com → sign in with GitHub → **Add New → Project** → import this repo.
2. **Root Directory:** `frontend`.
3. **Environment variable:** `NEXT_PUBLIC_API_URL` = your Render address, e.g. `https://dermalens-api.onrender.com`.
4. Deploy. Then put the Vercel address into Render's `CORS_ORIGINS` and redeploy the API.

## 5. Check it end to end

Open the Vercel site, scroll to the sandbox, and confirm the status pill says **Model online**. Upload an image you are allowed to use, run the stress test, and check that the Evidence section shows the held-out results.
