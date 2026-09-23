"use client";

import {
  DragEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

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

const LESION_IMAGE =
  "https://isic-archive.s3.amazonaws.com/images/ISIC_0016128.jpg";
const LAB_IMAGE =
  "https://cdn2.picryl.com/photo/2008/07/24/researcher-looks-through-microscope-2-bfc551-1024.jpg";

const story = [
  {
    eyebrow: "01 / Analyze",
    title: "Start with the image.",
    body:
      "Upload a dermatoscopic image and inspect the model's complete probability distribution instead of only a single label.",
  },
  {
    eyebrow: "02 / Attention",
    title: "See what influenced it.",
    body:
      "Grad-CAM reveals where the network concentrated its attention, helping separate a prediction from the visual evidence behind it.",
  },
  {
    eyebrow: "03 / Robustness",
    title: "Change the conditions.",
    body:
      "Brightness, contrast, and blur perturbations expose whether the same image keeps producing the same top class.",
  },
  {
    eyebrow: "04 / Evidence",
    title: "Keep claims measurable.",
    body:
      "DermaLens separates what is implemented from what has actually been measured on the held-out evaluation split.",
  },
];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [stress, setStress] = useState<StressResponse | null>(null);
  const [research, setResearch] = useState<ResearchStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [stressLoading, setStressLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [activeStory, setActiveStory] = useState(0);
  const [sandboxTab, setSandboxTab] = useState<"prediction" | "attention" | "robustness">("prediction");
  const [sampleLoading, setSampleLoading] = useState(false);
  const storyRef = useRef<HTMLElement | null>(null);
  const heroFileRef = useRef<HTMLInputElement | null>(null);

  const preview = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    Promise.allSettled([
      fetch(`${base}/research-status`).then((response) =>
        response.ok ? response.json() : null
      ),
      fetch(`${base}/health`).then((response) => response.ok),
    ]).then(([researchResult, healthResult]) => {
      if (researchResult.status === "fulfilled" && researchResult.value) {
        setResearch(researchResult.value);
      }
      setApiOnline(
        healthResult.status === "fulfilled" ? healthResult.value : false
      );
    });
  }, []);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  useEffect(() => {
    function updateStory() {
      const node = storyRef.current;
      if (!node) return;
      const rect = node.getBoundingClientRect();
      const scrollable = Math.max(node.offsetHeight - window.innerHeight, 1);
      const passed = Math.min(Math.max(-rect.top, 0), scrollable);
      const progress = passed / scrollable;
      setActiveStory(Math.min(3, Math.floor(progress * 4)));
    }

    updateStory();
    window.addEventListener("scroll", updateStory, { passive: true });
    window.addEventListener("resize", updateStory);
    return () => {
      window.removeEventListener("scroll", updateStory);
      window.removeEventListener("resize", updateStory);
    };
  }, []);

  function goToStory(index: number) {
    const node = storyRef.current;
    if (!node) return;
    const rect = node.getBoundingClientRect();
    const sectionTop = rect.top + window.scrollY;
    const scrollable = Math.max(node.offsetHeight - window.innerHeight, 1);
    const target = sectionTop + (index / Math.max(story.length - 1, 1)) * scrollable;
    window.scrollTo({ top: target, behavior: "smooth" });
  }

  function chooseFile(nextFile: File | null) {
    if (nextFile) {
      const allowed = ["image/jpeg", "image/png", "image/webp"];
      if (!allowed.includes(nextFile.type)) {
        setError("Choose a JPEG, PNG, or WebP image.");
        return;
      }
      if (nextFile.size > 10 * 1024 * 1024) {
        setError("Image must be under 10 MB.");
        return;
      }
    }

    setFile(nextFile);
    setResult(null);
    setStress(null);
    setSandboxTab("prediction");
    setError("");
  }

  function resetSandbox() {
    setFile(null);
    setResult(null);
    setStress(null);
    setSandboxTab("prediction");
    setError("");
  }

  function exportResearchResult() {
    if (!result || result.demo_mode) return;
    const report = {
      generated_at: new Date().toISOString(),
      file_name: file?.name ?? null,
      top_class: result.top_class,
      top_class_label: labels[result.top_class] ?? result.top_class,
      confidence: result.confidence,
      uncertainty: result.uncertainty,
      entropy: result.entropy,
      probabilities: result.probabilities,
      robustness: stress,
      disclaimer: result.disclaimer,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "dermalens-research-result.json";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  async function loadSampleImage() {
    setSampleLoading(true);
    setError("");
    try {
      const response = await fetch("/api/sample");
      if (!response.ok) throw new Error("Could not load the sample image.");
      const blob = await response.blob();
      const sample = new File([blob], "sample-lesion.jpg", {
        type: blob.type || "image/jpeg",
      });
      chooseFile(sample);
      document.getElementById("sandbox")?.scrollIntoView({ behavior: "smooth" });
    } catch {
      setError("Could not load the sample image. You can still upload your own image.");
    } finally {
      setSampleLoading(false);
    }
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
      setSandboxTab("prediction");
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
      setSandboxTab("robustness");
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
      <div className="announcement">
        <span className="announceDot">✦</span>
        DermaLens research preview is live.
        <a href="#sandbox">Try the sandbox ↗</a>
      </div>

      <nav className="floatingNav">
        <a className="brand" href="#top" aria-label="DermaLens home">
          <span className="brandIcon">D</span>
          <span>DermaLens</span>
        </a>

        <div className="navLinks">
          <a href="#product">Product⌄</a>
          <a href="#research">Research⌄</a>
          <a href="#sandbox">Sandbox</a>
          <a href="#evidence">Evidence⌄</a>
        </div>

        <button
          className="navCta"
          onClick={() =>
            document.getElementById("sandbox")?.scrollIntoView({ behavior: "smooth" })
          }
        >
          Try DermaLens
        </button>
      </nav>

      <section className="hero" id="top">
        <p className="heroKicker">Explainable image intelligence</p>
        <h1>See what the model sees.</h1>
        <p className="heroSub">
          Inspect probability, attention, and robustness for skin-lesion image
          classification in one continuous research workflow.
        </p>

        <div className="visualRail fullBleed" aria-label="DermaLens capabilities">
          <article className="railCard imageCard">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={LESION_IMAGE} alt="" />
          </article>

          <article className="railCard riskCard">
            <span>Top model score</span>
            <strong>87%</strong>
            <div className="scoreBar"><i /></div>
            <small>example interface</small>
          </article>

          <article className="railCard heatCard">
            <div className="fakeHeat">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={LESION_IMAGE} alt="" />
              <div className="heatBlob heatBlobOne" />
              <div className="heatBlob heatBlobTwo" />
            </div>
            <span>Attention map</span>
          </article>

          <article className="railCard conditionCard">
            <span>Robustness</span>
            <div className="miniConditions">
              {[0, 1, 2].map((item) => (
                <div key={item}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={LESION_IMAGE} alt="" />
                </div>
              ))}
            </div>
          </article>

          <article className="railCard labCard">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={LAB_IMAGE} alt="" />
          </article>

          <article className="railCard manifestoCard">
            <span>Explainable AI</span>
            <strong>From images<br />to evidence.</strong>
          </article>

          <article className="railCard chartCard">
            <span>Prediction confidence</span>
            <svg viewBox="0 0 220 95" role="img" aria-label="Illustrative confidence curves">
              <path d="M0 82 C36 78 40 20 72 20 C104 20 111 76 150 76 C175 76 185 51 220 48" />
              <path d="M0 85 C48 83 75 48 104 48 C140 48 150 76 220 79" />
            </svg>
          </article>

          <article className="railCard copyCard">
            <span>Research, not diagnosis.</span>
            <strong>Measure first.<br />Claim second.</strong>
          </article>

          <article className="railCard labCard secondaryLabCard">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={LAB_IMAGE} alt="" />
          </article>

          <article className="railCard searchCard">
            <span className="searchIcon">⌕</span>
            <div>
              <i />
              <i />
              <i />
            </div>
            <small>Inspect every class</small>
          </article>

          <article className="railCard robustnessRailCard">
            <span>Stress test</span>
            <div className="robustnessThumbRow">
              {[0, 1, 2, 3].map((item) => (
                <div key={item}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={LESION_IMAGE}
                    alt=""
                    style={{
                      filter:
                        item === 1
                          ? "brightness(1.2)"
                          : item === 2
                            ? "brightness(.72)"
                            : item === 3
                              ? "contrast(.68)"
                              : "none",
                    }}
                  />
                </div>
              ))}
            </div>
          </article>

          <article className="railCard editorialCard">
            <span>Better skin health</span>
            <strong>through<br />transparent AI.</strong>
          </article>

          <article className="railCard imageCard secondLesionCard">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={LESION_IMAGE} alt="" />
            <span>Attention ≠ diagnosis</span>
          </article>
        </div>

        <div className="heroComposer">
          <p>
            Analyze a dermatoscopic image, show what influenced the prediction,
            and compare robustness under lighting changes.
          </p>
          <div className="composerBottom">
            <div className="composerPills">
              <button
                className="roundPill composerUpload"
                onClick={() => heroFileRef.current?.click()}
                aria-label="Choose image"
              >
                ＋
              </button>
              <button
                className="softPill composerUpload"
                onClick={() => heroFileRef.current?.click()}
              >
                ▧ Image
              </button>
              <span className="softPill">◉ DermaLens Research⌄</span>
            </div>
            <input
              ref={heroFileRef}
              className="heroHiddenInput"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(event) => {
                chooseFile(event.target.files?.[0] ?? null);
                if (event.target.files?.[0]) {
                  document.getElementById("sandbox")?.scrollIntoView({ behavior: "smooth" });
                }
              }}
            />
            <button
              className="startButton"
              onClick={() =>
                document.getElementById("sandbox")?.scrollIntoView({ behavior: "smooth" })
              }
            >
              Start <span>→</span>
            </button>
          </div>
        </div>

        <div className="quickActions">
          <span>▧ Analyze image</span>
          <span>◌ View attention</span>
          <span>◫ Test robustness</span>
          <span>▥ Metrics</span>
          <span>▤ Research status</span>
        </div>
      </section>

      <section className="scrollStory fullBleed" id="product" ref={storyRef}>
        <div className="storySticky">
          <div className="storyCopy">
            <div className="storyCopySwap" key={activeStory}>
              <span className="storyEyebrow">{story[activeStory].eyebrow}</span>
              <h2>{story[activeStory].title}</h2>
              <p>{story[activeStory].body}</p>
            </div>
            <div className="storyProgressMeta">
              <span>{String(activeStory + 1).padStart(2, "0")} / 04</span>
              <span>Scroll to explore</span>
            </div>
            <div className="storyDots" aria-label="Scroll story progress">
              {story.map((_, index) => (
                <button
                  key={index}
                  className={index === activeStory ? "storyDot active" : "storyDot"}
                  onClick={() => goToStory(index)}
                  aria-label={`Show story step ${index + 1}`}
                />
              ))}
            </div>
          </div>

          <div className="storyFrame">
            <div className="mockChrome">
              <div className="mockBrand"><span>D</span> DermaLens</div>
              <div className="mockTabs">
                <span className={activeStory === 0 ? "active" : ""}>Analyze</span>
                <span className={activeStory === 1 ? "active" : ""}>Attention</span>
                <span className={activeStory === 2 ? "active" : ""}>Robustness</span>
                <span className={activeStory === 3 ? "active" : ""}>Evidence</span>
              </div>
            </div>

            <div className="storyScreen storyScreenStack">
              <div className={activeStory === 0 ? "storyPane active" : "storyPane"}>
                <div className="analyzeMock">
                  <div className="mockImage">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={LESION_IMAGE} alt="" />
                  </div>
                  <div className="mockPrediction">
                    <span>Prediction</span>
                    <h3>Melanoma</h3>
                    <strong>0.87</strong>
                    <div className="mockBar"><i /></div>
                    <ul>
                      <li><span>Nevus</span><b>0.08</b></li>
                      <li><span>Basal cell carcinoma</span><b>0.03</b></li>
                      <li><span>Actinic keratosis</span><b>0.02</b></li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className={activeStory === 1 ? "storyPane active" : "storyPane"}>
                <div className="attentionMock">
                  <div className="attentionPhoto">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={LESION_IMAGE} alt="" />
                    <div className="heatBlob heatBlobLarge" />
                  </div>
                  <div className="attentionText">
                    <span>Grad-CAM</span>
                    <h3>What influenced the top score?</h3>
                    <p>
                      The highlighted region marks the pixels that contributed most
                      strongly to the model&apos;s selected class.
                    </p>
                  </div>
                </div>
              </div>

              <div className={activeStory === 2 ? "storyPane active" : "storyPane"}>
                <div className="robustMock">
                  {[
                    ["Original", "none"],
                    ["Brighter", "brightness(1.25)"],
                    ["Darker", "brightness(.72)"],
                    ["Lower contrast", "contrast(.72)"],
                  ].map(([name, filter]) => (
                    <div className="robustTile" key={name}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={LESION_IMAGE} alt="" style={{ filter }} />
                      <span>{name}</span>
                    </div>
                  ))}
                  <div className="stabilityMock">
                    <span>Top-class stability</span>
                    <strong>80%</strong>
                  </div>
                </div>
              </div>

              <div className={activeStory === 3 ? "storyPane active" : "storyPane"}>
                <div className="evidenceMock">
                  <div className="evidenceIntro">
                    <span>Held-out evaluation</span>
                    <h3>Only show what has been measured.</h3>
                  </div>
                  <div className="evidenceRows">
                    <div><span>Lesion-level split</span><b>Implemented</b></div>
                    <div><span>Class weighting</span><b>Implemented</b></div>
                    <div><span>Calibration metrics</span><b>Implemented</b></div>
                    <div><span>Final test results</span><b>Pending training</b></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="sandboxSection" id="sandbox">
        <div className="sectionIntro">
          <span>Sandbox</span>
          <h2>Try the research workflow yourself.</h2>
          <p>
            Upload an image, inspect the model output, then stress-test the same image
            against controlled changes in image conditions.
          </p>
        </div>

        <div className="sandboxWindow">
          <div className="sandboxTopbar">
            <div>
              <span className="sandboxLogo">D</span>
              <strong>DermaLens Sandbox</strong>
            </div>
            <div className="sandboxTopActions">
              <div className="healthPill">
                <i className={research?.model_loaded ? "healthDot live" : apiOnline ? "healthDot api" : "healthDot"} />
                {research?.model_loaded
                  ? "Model online"
                  : apiOnline
                    ? "API online · weights pending"
                    : "Local mode"}
              </div>
              <button className="resetButton" onClick={resetSandbox} disabled={!file && !result}>
                Reset
              </button>
              <button className="exportButton" onClick={exportResearchResult} disabled={!result || result.demo_mode}>
                Export result
              </button>
            </div>
          </div>

          <div className="sandboxBody">
            <div className="uploadColumn">
              <div className="sandboxLabel">01 / Input</div>
              <label
                className={dragging ? "uploadDrop dragging" : "uploadDrop"}
                onDragEnter={() => setDragging(true)}
                onDragLeave={() => setDragging(false)}
                onDragOver={(event) => event.preventDefault()}
                onDrop={onDrop}
              >
                {preview ? (
                  <div className="sandboxPreview">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={preview} alt="Selected lesion preview" />
                    <span>{file?.name}</span>
                  </div>
                ) : (
                  <div className="uploadEmpty">
                    <div className="uploadCircle">＋</div>
                    <strong>Drop an image here</strong>
                    <span>JPEG, PNG, WebP · up to 10 MB</span>
                  </div>
                )}
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  aria-label="Choose a skin-lesion image for analysis"
                  onChange={(event) => chooseFile(event.target.files?.[0] ?? null)}
                />
              </label>

              <div className="uploadActions">
                <button
                  className="analyzeButton"
                  onClick={analyze}
                  disabled={!file || loading}
                >
                  {loading ? "Running analysis…" : "Analyze image"}
                  <span>→</span>
                </button>
                <button
                  className="sampleButton"
                  onClick={loadSampleImage}
                  disabled={sampleLoading}
                >
                  {sampleLoading ? "Loading sample…" : "Use sample image"}
                </button>
              </div>
              {error && <p className="error">{error}</p>}
            </div>

            <div className="resultColumn">
              <div className="resultColumnHeader">
                <div className="sandboxLabel">02 / Inspect</div>
                <div className="sandboxTabs" role="tablist" aria-label="Analysis views">
                  <button
                    className={sandboxTab === "prediction" ? "active" : ""}
                    onClick={() => setSandboxTab("prediction")}
                  >
                    Prediction
                  </button>
                  <button
                    className={sandboxTab === "attention" ? "active" : ""}
                    onClick={() => setSandboxTab("attention")}
                    disabled={!result || result.demo_mode || !result.heatmap_data_url}
                  >
                    Attention
                  </button>
                  <button
                    className={sandboxTab === "robustness" ? "active" : ""}
                    onClick={() => setSandboxTab("robustness")}
                    disabled={!stress}
                  >
                    Robustness
                  </button>
                </div>
              </div>

              {!result ? (
                <div className="resultEmpty">
                  <div className="emptyHalo" />
                  <strong>Model output will appear here.</strong>
                  <span>
                    The sandbox stays empty until you upload and analyze an image.
                  </span>
                </div>
              ) : result.demo_mode ? (
                <div className="demoResult">
                  <span>Demo mode</span>
                  <h3>The interface works. Trained weights are not loaded yet.</h3>
                  <p>
                    DermaLens intentionally does not fabricate a medical prediction.
                  </p>
                </div>
              ) : sandboxTab === "attention" && result.heatmap_data_url ? (
                <div className="attentionResult">
                  <div className="attentionCompare">
                    <div>
                      <span className="compareLabel">Original</span>
                      <div className="compareImage">
                        {preview && (
                          /* eslint-disable-next-line @next/next/no-img-element */
                          <img src={preview} alt="Original uploaded lesion" />
                        )}
                      </div>
                    </div>
                    <div>
                      <span className="compareLabel">Grad-CAM</span>
                      <div className="compareImage">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={result.heatmap_data_url} alt="Grad-CAM attention map" />
                      </div>
                    </div>
                  </div>
                  <div className="attentionNote">
                    <strong>Attention shows influence, not medical meaning.</strong>
                    <p>
                      The highlighted regions are the pixels that most affected the top-scoring class.
                      They are not proof that a highlighted region is malignant or clinically important.
                    </p>
                  </div>
                </div>
              ) : sandboxTab === "robustness" && stress ? (
                <div className="robustnessResult">
                  <div className="robustnessSummary">
                    <span>Top-class stability</span>
                    <strong>{Math.round((stress.stability ?? 0) * 100)}%</strong>
                    <p>Share of tested image conditions that kept the original top class.</p>
                  </div>
                  <div className="robustnessTable">
                    {stress.results.map((row) => (
                      <div className="robustnessTableRow" key={row.variant}>
                        <span>{row.variant}</span>
                        <span>{labels[row.top_class] ?? row.top_class}</span>
                        <b>{Math.round(row.confidence * 100)}%</b>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="liveResult">
                  <div className="resultHeadline">
                    <div>
                      <span>Highest model score</span>
                      <h3>{labels[result.top_class] ?? result.top_class}</h3>
                    </div>
                    <strong>{Math.round(result.confidence * 100)}%</strong>
                  </div>

                  <div className="resultStats">
                    <div><span>Uncertainty</span><b>{Math.round(result.uncertainty * 100)}%</b></div>
                    <div><span>Entropy</span><b>{Math.round(result.entropy * 100)}%</b></div>
                  </div>

                  <div className="probabilityRows">
                    {sorted.map(([key, value]) => (
                      <div className="probabilityRow" key={key}>
                        <div><span>{labels[key] ?? key}</span><b>{(value * 100).toFixed(1)}%</b></div>
                        <div className="probabilityTrack"><i style={{ width: `${value * 100}%` }} /></div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {result && !result.demo_mode && (
            <div className="sandboxBottom">
              <div>
                <span>03 / Robustness</span>
                <strong>Does the answer survive a worse photo?</strong>
              </div>
              <button onClick={runStressTest} disabled={stressLoading}>
                {stressLoading ? "Testing…" : "Run stress test"} →
              </button>
            </div>
          )}

          {stress && !stress.demo_mode && sandboxTab !== "robustness" && (
            <div className="stressResults compactStressResults">
              <div className="stabilityCard">
                <span>Top-class stability</span>
                <strong>{Math.round((stress.stability ?? 0) * 100)}%</strong>
              </div>
              <div className="stressRows">
                {stress.results.map((row) => (
                  <div className="stressRow" key={row.variant}>
                    <span>{row.variant}</span>
                    <span>{labels[row.top_class] ?? row.top_class}</span>
                    <b>{Math.round(row.confidence * 100)}%</b>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      <section className="researchSection" id="research">
        <div className="researchCopy">
          <span>Research</span>
          <h2>Designed to make uncertainty visible.</h2>
        </div>

        <div className="researchCards">
          <article>
            <span>01</span>
            <h3>Lesion-level split</h3>
            <p>
              Images from the same lesion stay within one split, reducing leakage
              between training and evaluation.
            </p>
          </article>
          <article>
            <span>02</span>
            <h3>Class-aware training</h3>
            <p>
              Weighted loss helps keep majority classes from dominating the learning
              objective.
            </p>
          </article>
          <article>
            <span>03</span>
            <h3>Robustness checks</h3>
            <p>
              Controlled image perturbations expose sensitivity to lighting,
              contrast, and blur.
            </p>
          </article>
        </div>
      </section>

      <section className="evidenceSection" id="evidence">
        <div className="evidenceCopy">
          <span>Evidence</span>
          <h2>No performance claims until the test split says so.</h2>
          <p>
            The interface separates implemented methodology from measured results.
            Held-out metrics appear only after real trained weights have been evaluated.
          </p>
        </div>

        <div className="statusPanel">
          <div className="statusPanelTop">
            <span>Evaluation status</span>
            <strong>{research?.evaluation_available ? "Available" : "Pending training"}</strong>
          </div>

          {research?.evaluation_available && research.evaluation ? (
            <div className="metricGrid">
              {[
                ["Accuracy", research.evaluation.accuracy],
                ["Balanced accuracy", research.evaluation.balanced_accuracy],
                ["Macro F1", research.evaluation.macro_f1],
                ["ROC-AUC", research.evaluation.macro_ovr_roc_auc],
              ].map(([label, value]) => (
                <div key={String(label)}>
                  <span>{label}</span>
                  <strong>{typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—"}</strong>
                </div>
              ))}
            </div>
          ) : (
            <div className="pendingEvidence">
              <div className="pendingOrb" />
              <div>
                <strong>Waiting for the real evaluation.</strong>
                <p>
                  No placeholder accuracy, no invented benchmark, no diagnostic claim.
                </p>
              </div>
            </div>
          )}
        </div>
      </section>

      <section className="responsibility">
        <div>
          <span>Responsible use</span>
          <h2>A model score is not a diagnosis.</h2>
        </div>
        <p>
          DermaLens is an educational research prototype. Image quality, dataset
          composition, skin-tone representation, hardware, and distribution shift can
          all affect model behavior. Concerning lesions should be evaluated by a
          qualified clinician.
        </p>
      </section>

      <footer>
        <div className="brand footerBrand">
          <span className="brandIcon">D</span>
          <span>DermaLens</span>
        </div>
        <div className="footerMeta">
          <span>Educational research prototype</span>
          <span>Not a medical device</span>
          <span>2026</span>
        </div>
      </footer>
    </main>
  );
}
