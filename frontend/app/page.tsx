"use client";

import { useMemo, useState } from "react";

type Prediction = {
  top_class: string;
  confidence: number;
  uncertainty: number;
  probabilities: Record<string, number>;
  demo_mode: boolean;
  disclaimer: string;
};

const labels: Record<string, string> = {
  akiec: "Actinic keratosis / intraepithelial carcinoma",
  bcc: "Basal cell carcinoma",
  bkl: "Benign keratosis-like lesion",
  df: "Dermatofibroma",
  mel: "Melanoma",
  nv: "Melanocytic nevus",
  vasc: "Vascular lesion",
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const preview = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  async function analyze() {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const base = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
      const response = await fetch(`${base}/predict`, {
        method: "POST",
        body: form,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? "Analysis failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  const sorted = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <main>
      <nav>
        <div className="brand">DermaLens</div>
        <div className="badge">Research prototype</div>
      </nav>

      <section className="hero">
        <p className="eyebrow">Explainable medical-image ML</p>
        <h1>See what the model sees.</h1>
        <p className="lede">
          Explore how a skin-lesion classifier responds to an image, including
          uncertainty and transparent class probabilities.
        </p>
      </section>

      <section className="grid">
        <div className="card uploadCard">
          <div>
            <p className="kicker">01 — Image</p>
            <h2>Upload a lesion image</h2>
            <p className="muted">JPEG, PNG, or WebP. Maximum 10 MB.</p>
          </div>

          <label className="dropzone">
            {preview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={preview} alt="Selected lesion preview" />
            ) : (
              <div>
                <div className="plus">+</div>
                <strong>Select image</strong>
                <span>or drag one here</span>
              </div>
            )}
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(e) => {
                setFile(e.target.files?.[0] ?? null);
                setResult(null);
                setError("");
              }}
            />
          </label>

          <button onClick={analyze} disabled={!file || loading}>
            {loading ? "Analyzing…" : "Analyze image"}
          </button>

          {error && <p className="error">{error}</p>}
        </div>

        <div className="card resultCard">
          <p className="kicker">02 — Model output</p>

          {!result ? (
            <div className="empty">
              <div className="orb" />
              <p>Analysis will appear here.</p>
            </div>
          ) : result.demo_mode ? (
            <div className="demo">
              <h2>Model not loaded yet</h2>
              <p>
                The app is running correctly, but trained weights have not been
                installed. No medical prediction was generated.
              </p>
            </div>
          ) : (
            <>
              <div className="headlineResult">
                <div>
                  <span>Highest model score</span>
                  <h2>{labels[result.top_class] ?? result.top_class}</h2>
                </div>
                <strong>{Math.round(result.confidence * 100)}%</strong>
              </div>

              <div className="uncertainty">
                <span>Uncertainty</span>
                <strong>{Math.round(result.uncertainty * 100)}%</strong>
              </div>

              <div className="bars">
                {sorted.map(([key, value]) => (
                  <div key={key} className="barRow">
                    <div className="barLabel">
                      <span>{labels[key] ?? key}</span>
                      <span>{(value * 100).toFixed(1)}%</span>
                    </div>
                    <div className="track">
                      <div className="fill" style={{ width: `${value * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </section>

      <section className="explain">
        <p className="kicker">03 — Why this matters</p>
        <div className="explainGrid">
          <h2>A model score is not a diagnosis.</h2>
          <p>
            DermaLens is designed to make model behavior visible rather than hide
            it behind a single answer. The final CAC version will add Grad-CAM
            attention maps and robustness tests so users can see both what the
            model focuses on and when its predictions become unstable.
          </p>
        </div>
      </section>

      <footer>
        Educational research prototype. Not a medical device. Do not use this
        tool to make healthcare decisions.
      </footer>
    </main>
  );
}
