FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch: the default Linux wheels bundle CUDA and are several GB.
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app

# Trained weights are not in git. Set MODEL_URL (and EVALUATION_URL) to hosted
# files and the API downloads them into /models on startup.
# Run as an unprivileged user that can only write to /models.
RUN useradd --create-home --uid 10001 dermalens \
    && mkdir -p /models \
    && chown dermalens /models
USER dermalens

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
