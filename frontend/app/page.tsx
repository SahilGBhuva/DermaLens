"use client";

import { DragEvent, useEffect, useMemo, useState } from "react";

type Prediction = {
  top_class: string;
  confidence: number;
  uncertainty: number;
  entropy: number;
  probabilities: Record<string, number>;
  demo_mode: boolean;
  heatmap_data_url: string | null;
  disclaimer: string;
};

type StressRow = {
  variant: string;
  top_class: string;
  confidence: number;
  entropy: number;
};

type StressResponse = {
  demo_mode: boolean;
  stability: number | null;
  original_class?: string;
  results: StressRow[];
};

type ResearchStatus = {
  model_loaded: boolean;
  evaluation_available: boolean;
  evaluation: null | {
    accuracy: number | null;
    balanced_accuracy: number | null;
    macro_f1: number | null;
    weighted_f1: number | null;
    macro_ovr_roc_auc: number | null;
  };
  implemented: Record<string, boolean>;
  note: string;
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
  const [stress, setStress] = useState<StressResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [stressLoading, setStressLoading] = useState(false);
  const [error, setError] = useState("");
  const [research, setResearch] = useState<ResearchStatus | null>(null);

  const preview = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    fetch(`${base}/research-status`)
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (data) setResearch(data);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  function chooseFile(nextFile: File | null) {
    setFile(nextFile);
    setResult(null);
    setStress(null);
    setError("");
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    const dropped = event.dataTransfer.files?.[0] ?? null;
    if (dropped) chooseFile(dropped);
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    setStress(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const base = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
      const response = await fetch(`${base}/predict`, {
        method: "POST",
        body: form,
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Analysis failed.");
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  async function runStressTest() {
    if (!file || !result || result.demo_mode) return;
    setStressLoading(true);
    setError("");

    const form = new FormData();
    form.append("file", file);

    try {
      const base = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
      const response = await fetch(`${base}/stress-test`, {
        method: "POST",
        body: form,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? "Stress test failed.");
      setStress(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Stress test failed.");
    } finally {
      setStressLoading(false);
    }
  }

  const sorted = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <main>
      <nav>
        <div className="brand">DermaLens</div>
        <div className="badge">Educational research prototype</div>
      </nav>

      <section className="hero">
        <p className="eyebrow">Explainable medical-image ML</p>
        <h1>See what the model sees.</h1>
        <p className="lede">
          Explore how a skin-lesion model responds, where it focuses, and whether
          its answer survives simple changes in image conditions.
        </p>
      </section>

      <section className="grid">
        <div className="card uploadCard">
          <div>
            <p className="kicker">01 — Image</p>
            <h2>Upload a lesion image</h2>
            <p className="muted">JPEG, PNG, or WebP. Maximum 10 MB.</p>
          </div>

          <label
            className="dropzone"
            onDragOver={(event) => event.preventDefault()}
            onDrop={onDrop}
          >
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
              onChange={(event) => chooseFile(event.target.files?.[0] ?? null)}
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
                The app is working, but trained weights are not installed.
                No medical prediction was generated.
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

              <div className="metricStrip">
                <div>
                  <span>1 − top score</span>
                  <strong>{Math.round(result.uncertainty * 100)}%</strong>
                </div>
                <div>
                  <span>Normalized entropy</span>
                  <strong>{Math.round(result.entropy * 100)}%</strong>
                </div>
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

      {result?.heatmap_data_url && !result.demo_mode && (
        <section className="heatmapSection">
          <div>
            <p className="kicker">03 — Model attention</p>
            <h2>Where the network focused.</h2>
            <p className="muted">
              Grad-CAM highlights image regions that influenced the top-scoring
              class. Attention does not prove medical significance.
            </p>
          </div>
          <div className="heatmapFrame">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={result.heatmap_data_url} alt="Grad-CAM model attention heatmap" />
          </div>
        </section>
      )}

      {result && !result.demo_mode && (
        <section className="stressSection">
          <div className="stressIntro">
            <div>
              <p className="kicker">04 — Robustness lab</p>
              <h2>Does the answer survive a worse photo?</h2>
              <p className="muted">
                Re-run the same image after controlled brightness, contrast, and
                blur changes. A class flip is a visible warning that the model is
                sensitive to image conditions.
              </p>
            </div>
            <button
              className="secondaryButton"
              onClick={runStressTest}
              disabled={stressLoading}
            >
              {stressLoading ? "Stress testing…" : "Run stress test"}
            </button>
          </div>

          {stress && !stress.demo_mode && (
            <div className="stressResults">
              <div className="stabilityScore">
                <span>Top-class stability</span>
                <strong>{Math.round((stress.stability ?? 0) * 100)}%</strong>
                <small>
                  Share of tested image conditions that kept the original top class.
                </small>
              </div>

              <div className="stressTable">
                <div className="stressHeader">
                  <span>Condition</span>
                  <span>Top class</span>
                  <span>Score</span>
                  <span>Entropy</span>
                </div>
                {stress.results.map((row) => {
                  const changed = row.top_class !== stress.original_class;
                  return (
                    <div className="stressRow" key={row.variant}>
                      <span>{row.variant}</span>
                      <span className={changed ? "changedClass" : ""}>
                        {labels[row.top_class] ?? row.top_class}
                        {changed ? " · changed" : ""}
                      </span>
                      <span>{Math.round(row.confidence * 100)}%</span>
                      <span>{Math.round(row.entropy * 100)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </section>
      )}


      <section className="researchSection">
        <div className="researchHeading">
          <div>
            <p className="kicker">05 — Research status</p>
            <h2>What is built. What is actually proven.</h2>
          </div>
          <p className="muted">
            DermaLens keeps implementation claims separate from measured model
            performance. Test metrics only appear after a real held-out evaluation.
          </p>
        </div>

        <div className="researchGrid">
          <div className="researchChecklist">
            {[
              ["Lesion-level split", "lesion_level_split"],
              ["Class-weighted training", "class_weighted_training"],
              ["Held-out test pipeline", "held_out_test_pipeline"],
              ["Grad-CAM explainability", "grad_cam"],
              ["Robustness lab", "robustness_lab"],
            ].map(([label, key]) => (
              <div className="checkRow" key={key}>
                <span>{label}</span>
                <strong>{research?.implemented?.[key] ? "Implemented" : "—"}</strong>
              </div>
            ))}
          </div>

          <div className="metricPanel">
            <div className="metricPanelTop">
              <span>Held-out evaluation</span>
              <strong>
                {research?.evaluation_available ? "Available" : "Pending training"}
              </strong>
            </div>

            {research?.evaluation_available && research.evaluation ? (
              <div className="evaluationMetrics">
                {[
                  ["Accuracy", research.evaluation.accuracy],
                  ["Balanced accuracy", research.evaluation.balanced_accuracy],
                  ["Macro F1", research.evaluation.macro_f1],
                  ["ROC-AUC", research.evaluation.macro_ovr_roc_auc],
                ].map(([label, value]) => (
                  <div key={String(label)}>
                    <span>{label}</span>
                    <strong>
                      {typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—"}
                    </strong>
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted researchPending">
                No performance number is shown until trained weights are evaluated
                on the untouched test split.
              </p>
            )}
          </div>
        </div>
      </section>

      <section className="explain">
        <p className="kicker">06 — Limitations</p>
        <div className="explainGrid">
          <h2>A model score is not a diagnosis.</h2>
          <p>
            DermaLens is an educational research system. Image quality, dataset
            bias, skin tone representation, device differences, class imbalance,
            and distribution shift can all change model behavior. Concerning
            lesions should be evaluated by a qualified clinician.
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
