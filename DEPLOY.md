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

To use the new model locally first, install it from your Downloads folder. This checks that the weights, settings and evaluation belong together, installs them into `models/<version>/` (moving any older copy of that version to `models/.backups/`), and prints a comparison with every installed version:

```bash
python ml/install_model.py --from ~/Downloads --version v3
```

## 2. Publish the trained files

Each model version is one GitHub Release holding its four files, taken from that version's folder in `models/` (after `ml/install_model.py`, which adds the version label and evaluation fingerprint to `model_config.json`):

| Release tag | Files (from) |
|---|---|
| `model-v1` | `models/v1/`: `dermalens_efficientnet_b0.pt`, `model_config.json`, `evaluation.json`, `training_history.json` |
| `model-v2` | `models/v2/`: same four files |

On GitHub: **Releases → Draft a new release**, set the tag, attach the four files, and note in the description that the weights were trained on HAM10000 (CC BY-NC 4.0; Tschandl, Rosendahl & Kittler, 2018). Or from a terminal:

```bash
gh release create model-v2 models/v2/* --title "DermaLens model v2" --notes "Trained on HAM10000 (CC BY-NC 4.0)."
gh release create model-v1 models/v1/* --title "DermaLens model v1" --notes "Trained on HAM10000 (CC BY-NC 4.0)."
```

`model_config.json` holds each model's SHA-256; the API refuses weights that don't match it, and only publishes metrics measured on those exact weights.

## 3. Deploy the API on Render

1. https://render.com → sign in with GitHub → **New → Blueprint** → choose this repo. Render reads `render.yaml`.
2. Fill the environment variables it asks for:
   - `MODEL_RELEASES` — every version to serve, **default first**:
     `v2=https://github.com/SahilGBhuva/DermaLens/releases/download/model-v2,v1=https://github.com/SahilGBhuva/DermaLens/releases/download/model-v1`
   - `CORS_ORIGINS` — your Vercel address from step 4 (you can come back and set it after)
3. Deploy. When it is live, open `https://<your-api>.onrender.com/health` — it should list `"models": ["v2", "v1"]`.

Memory: with both versions, local CPU measurements peaked around 470 MB under 12 concurrent requests (free limit 512 MB; one analysis runs at a time). If the free instance is ever OOM-killed, serve one version (`MODEL_RELEASES=v2=...`) or move to the Starter plan.

Single-model deployments can still use `MODEL_URL`, `CONFIG_URL`, `EVALUATION_URL` and `HISTORY_URL` instead.

## 4. Deploy the website on Vercel

1. https://vercel.com → sign in with GitHub → **Add New → Project** → import this repo.
2. **Root Directory:** `frontend`.
3. **Environment variable:** `NEXT_PUBLIC_API_URL` = your Render address, e.g. `https://dermalens-api.onrender.com`.
4. Deploy. Then put the Vercel address into Render's `CORS_ORIGINS` and redeploy the API.

## 5. Check it end to end

Open the Vercel site, scroll to the sandbox, and confirm the status pill says **Model online**. Upload an image you are allowed to use, run the stress test, and check that the Evidence section shows the held-out results.
