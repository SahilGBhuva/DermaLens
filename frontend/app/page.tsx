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

const researchItems = [
  ["Lesion-level split", "lesion_level_split"],
  ["Class-weighted training", "class_weighted_training"],
  ["Held-out test pipeline", "held_out_test_pipeline"],
  ["Grad-CAM explainability", "grad_cam"],
  ["Robustness lab", "robustness_lab"],
] as const;

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [stress, setStress] = useState<StressResponse | null>(null);
  const [research, setResearch] = useState<ResearchStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [stressLoading, setStressLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  const preview = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    fetch(`${base}/research-status`)
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => data && setResearch(data))
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
    setDragging(false);
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
      <div className="ambient ambientOne" />
      <div className="ambient ambientTwo" />

      <nav className="siteNav">
        <a className="brand" href="#top" aria-label="DermaLens home">
          <span className="brandMark" />
          DermaLens
        </a>
        <div className="navLinks">
          <a href="#analyze">Analyze</a>
          <a href="#research">Research</a>
          <a href="#limits">Limits</a>
        </div>
        <div className="navStatus">
          <span className={research?.model_loaded ? "statusDot live" : "statusDot"} />
          {research?.model_loaded ? "Model online" : "Research build"}
        </div>
      </nav>

      <section className="hero" id="top">
        <div className="heroCopy">
          <div className="heroEyebrow">
            <span>Explainable medical-image intelligence</span>
            <span>Research prototype / 2026</span>
          </div>

          <h1>
            Don&apos;t just trust
            <span className="ghostWord"> the answer.</span>
            <br />
            <em>Interrogate it.</em>
          </h1>

          <div className="heroBottom">
            <p>
              DermaLens turns a skin-lesion prediction into something you can inspect:
              probability, uncertainty, model attention, and whether the answer survives
              changes to the image.
            </p>

            <div className="heroActions">
              <button
                className="primaryCta"
                onClick={() =>
                  document.getElementById("analyze")?.scrollIntoView({ behavior: "smooth" })
                }
              >
                Enter the model
                <span>↘</span>
              </button>
              <a className="textCta" href="#research">
                View methodology <span>→</span>
              </a>
            </div>
          </div>
        </div>

        <div className="heroVisual" aria-hidden="true">
          <div className="visualGrid" />
          <div className="halo haloA" />
          <div className="halo haloB" />
          <div className="halo haloC" />
          <div className="coreSphere">
            <div className="sphereNoise" />
            <div className="scanBand" />
          </div>
          <div className="orbit orbitOne"><span /></div>
          <div className="orbit orbitTwo"><span /></div>
          <div className="visualLabel labelOne">LATENT / 07</div>
          <div className="visualLabel labelTwo">UNCERTAINTY FIELD</div>
          <div className="visualLabel labelThree">X 0.384 &nbsp; Y 0.617</div>
        </div>
      </section>

      <section className="marquee" aria-label="DermaLens capabilities">
        <div>
          <span>Explainability</span><i>◆</i>
          <span>Robustness</span><i>◆</i>
          <span>Calibration</span><i>◆</i>
          <span>Transparency</span><i>◆</i>
          <span>Failure analysis</span><i>◆</i>
          <span>Explainability</span><i>◆</i>
          <span>Robustness</span><i>◆</i>
          <span>Calibration</span><i>◆</i>
        </div>
      </section>

      <section className="statementSection">
        <p className="sectionIndex">00 / premise</p>
        <div className="statement">
          <p className="statementLead">A prediction is easy.</p>
          <p>
            Knowing <em>when it becomes fragile</em> is harder. DermaLens is built
            around that gap.
          </p>
        </div>
      </section>

      <section className="analysisSection" id="analyze">
        <div className="sectionTopline">
          <div>
            <p className="sectionIndex">01 / analysis chamber</p>
            <h2>Put an image under pressure.</h2>
          </div>
          <p>
            One image. Seven classes. Multiple ways to inspect what the network is
            actually doing.
          </p>
        </div>

        <div className="analysisShell">
          <div className="uploadPanel">
            <div className="panelHeader">
              <div>
                <span className="microLabel">INPUT</span>
                <strong>Skin-lesion image</strong>
              </div>
              <span className="panelCode">IMG / RGB</span>
            </div>

            <label
              className={dragging ? "dropzone dragging" : "dropzone"}
              onDragEnter={() => setDragging(true)}
              onDragLeave={() => setDragging(false)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={onDrop}
            >
              {preview ? (
                <div className="previewWrap">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={preview} alt="Selected lesion preview" />
                  <div className="scanOverlay" />
                  <div className="corner cornerTL" />
                  <div className="corner cornerTR" />
                  <div className="corner cornerBL" />
                  <div className="corner cornerBR" />
                  <span className="previewTag">{file?.name}</span>
                </div>
              ) : (
                <div className="dropInner">
                  <div className="uploadGlyph">
                    <span>+</span>
                  </div>
                  <strong>Drop image into field</strong>
                  <span>or click to browse · JPEG / PNG / WebP · 10 MB max</span>
                </div>
              )}
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={(event) => chooseFile(event.target.files?.[0] ?? null)}
              />
            </label>

            <button className="analyzeButton" onClick={analyze} disabled={!file || loading}>
              <span>{loading ? "Running inference" : "Run analysis"}</span>
              <span className={loading ? "buttonPulse active" : "buttonPulse"} />
            </button>

            {error && <p className="error">{error}</p>}
          </div>

          <div className="resultPanel">
            <div className="panelHeader">
              <div>
                <span className="microLabel">OUTPUT</span>
                <strong>Model response</strong>
              </div>
              <span className="panelCode">7-CLASS / SOFTMAX</span>
            </div>

            {!result ? (
              <div className="idleState">
                <div className="latentMap">
                  <span className="latentPoint p1" />
                  <span className="latentPoint p2" />
                  <span className="latentPoint p3" />
                  <span className="latentPoint p4" />
                  <span className="latentPoint p5" />
                  <span className="latentPoint p6" />
                  <div className="latentRing r1" />
                  <div className="latentRing r2" />
                  <div className="latentCross" />
                </div>
                <p>Awaiting image signal.</p>
                <span>The model output will resolve here.</span>
              </div>
            ) : result.demo_mode ? (
              <div className="demoState">
                <span className="demoPill">DEMO MODE</span>
                <h3>The interface is live. The trained model is not.</h3>
                <p>
                  DermaLens will never fabricate a medical prediction. Install the
                  trained weights and real model output will appear here.
                </p>
              </div>
            ) : (
              <div className="resultContent">
                <div className="resultLead">
                  <div>
                    <span className="microLabel">TOP MODEL SCORE</span>
                    <h3>{labels[result.top_class] ?? result.top_class}</h3>
                  </div>
                  <strong>{Math.round(result.confidence * 100)}<sup>%</sup></strong>
                </div>

                <div className="metricTiles">
                  <div>
                    <span>Uncertainty</span>
                    <strong>{Math.round(result.uncertainty * 100)}%</strong>
                  </div>
                  <div>
                    <span>Entropy</span>
                    <strong>{Math.round(result.entropy * 100)}%</strong>
                  </div>
                </div>

                <div className="probabilityList">
                  {sorted.map(([key, value], index) => (
                    <div className="probabilityRow" key={key}>
                      <div className="probabilityMeta">
                        <span className="rank">0{index + 1}</span>
                        <span>{labels[key] ?? key}</span>
                        <strong>{(value * 100).toFixed(1)}%</strong>
                      </div>
                      <div className="probabilityTrack">
                        <div style={{ width: `${value * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {result?.heatmap_data_url && !result.demo_mode && (
        <section className="attentionSection">
          <div className="attentionCopy">
            <p className="sectionIndex">02 / attention field</p>
            <h2>See where the network looked.</h2>
            <p>
              Grad-CAM reveals which image regions most influenced the top-scoring
              class. It is a window into model behavior, not proof of medical meaning.
            </p>
          </div>
          <div className="heatmapStage">
            <div className="heatmapTop">
              <span>GRAD-CAM / CLASS ACTIVATION</span>
              <span>LIVE OUTPUT</span>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={result.heatmap_data_url} alt="Grad-CAM model attention heatmap" />
            <div className="heatmapReticle" />
          </div>
        </section>
      )}

      <section className="robustnessSection">
        <div className="robustnessHeader">
          <div>
            <p className="sectionIndex">03 / robustness lab</p>
            <h2>What if the image changes, but the lesion doesn&apos;t?</h2>
          </div>
          <div className="robustnessAside">
            <p>
              We deliberately perturb brightness, contrast, and blur, then measure
              whether the model keeps the same top class.
            </p>
            <button
              className="outlineButton"
              onClick={runStressTest}
              disabled={!result || result.demo_mode || stressLoading}
            >
              {stressLoading ? "Applying perturbations…" : "Stress-test this image"}
            </button>
          </div>
        </div>

        {stress && !stress.demo_mode ? (
          <div className="stressGrid">
            <div className="stabilityCard">
              <span>TOP-CLASS STABILITY</span>
              <strong>{Math.round((stress.stability ?? 0) * 100)}%</strong>
              <div className="stabilityMeter">
                <div style={{ width: `${(stress.stability ?? 0) * 100}%` }} />
              </div>
              <p>Across five controlled image conditions.</p>
            </div>
            <div className="stressRows">
              {stress.results.map((row, index) => {
                const changed = row.top_class !== stress.original_class;
                return (
                  <div className="stressRow" key={row.variant}>
                    <span className="stressIndex">0{index + 1}</span>
                    <strong>{row.variant}</strong>
                    <span className={changed ? "classChanged" : ""}>
                      {labels[row.top_class] ?? row.top_class}
                    </span>
                    <span>{Math.round(row.confidence * 100)}%</span>
                    <span className="stressState">{changed ? "FLIPPED" : "STABLE"}</span>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="robustnessPlaceholder">
            <div className="placeholderSignal">
              <span /><span /><span /><span /><span />
            </div>
            <div>
              <strong>Robustness remains invisible until you test it.</strong>
              <p>Run a real analysis first, then stress-test the same image.</p>
            </div>
          </div>
        )}
      </section>

      <section className="researchSection" id="research">
        <div className="researchIntro">
          <p className="sectionIndex">04 / evidence layer</p>
          <h2>Built to show the receipts.</h2>
          <p>
            No invented benchmarks. No mystery pipeline. Implementation claims and
            measured performance stay separate until the held-out test evaluation exists.
          </p>
        </div>

        <div className="researchLayout">
          <div className="researchChecklist">
            <div className="researchCardHeader">
              <span>PIPELINE STATUS</span>
              <span>{research ? "SYNCED" : "CONNECTING"}</span>
            </div>
            {researchItems.map(([label, key], index) => (
              <div className="checkRow" key={key}>
                <span className="checkNumber">0{index + 1}</span>
                <span>{label}</span>
                <strong>{research?.implemented?.[key] ? "READY" : "—"}</strong>
              </div>
            ))}
          </div>

          <div className="metricPanel">
            <div className="researchCardHeader">
              <span>HELD-OUT EVALUATION</span>
              <span>{research?.evaluation_available ? "AVAILABLE" : "PENDING"}</span>
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
              <div className="pendingMetrics">
                <div className="pendingOrb" />
                <div>
                  <strong>No fake numbers.</strong>
                  <p>
                    Metrics unlock only after trained weights are evaluated on the
                    untouched test split.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="limitsSection" id="limits">
        <p className="sectionIndex">05 / limits</p>
        <div className="limitsHeadline">
          <h2>Confidence is not certainty.</h2>
          <p>
            DermaLens is an educational research system—not a diagnostic device.
            Image quality, dataset composition, skin-tone representation, hardware,
            class imbalance, and distribution shift can all change model behavior.
          </p>
        </div>

        <div className="limitsTicker">
          <span>DATASET SHIFT</span>
          <span>IMAGE QUALITY</span>
          <span>CLASS IMBALANCE</span>
          <span>REPRESENTATION</span>
          <span>DEVICE VARIATION</span>
          <span>CALIBRATION</span>
        </div>
      </section>

      <footer>
        <div>
          <span className="brand footerBrand"><span className="brandMark" />DermaLens</span>
          <p>Researching when image classifiers deserve trust.</p>
        </div>
        <div className="footerMeta">
          <span>Educational research prototype</span>
          <span>Not a medical device</span>
          <span>© 2026</span>
        </div>
      </footer>
    </main>
  );
}
