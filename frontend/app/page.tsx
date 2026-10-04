"use client";

import {
  ChangeEvent,
  DragEvent,
  PointerEvent as ReactPointerEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import dynamic from "next/dynamic";
import LesionScope, { ScopeMode } from "./components/LesionScope";
import { rasterizeSvg } from "./lib/rasterize";

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
  model?: {
    status: string;
    version: string | null;
    recipe: string | null;
    tta: boolean;
    calibrated: boolean;
    min_mel_sensitivity_target: number | null;
  };
  training?: {
    epoch: number;
    train_accuracy: number | null;
    val_accuracy: number | null;
    val_balanced_accuracy: number | null;
  }[] | null;
  evaluation_available: boolean;
  evaluation: null | {
    accuracy: number | null;
    balanced_accuracy: number | null;
    macro_f1: number | null;
    weighted_f1: number | null;
    macro_ovr_roc_auc: number | null;
    expected_calibration_error?: number | null;
    multiclass_brier_score?: number | null;
    test_images?: number | null;
    per_class?: {
      label: string;
      sensitivity: number | null;
      specificity: number | null;
      support: number;
    }[];
    confusion_matrix?: number[][] | null;
    classes?: string[];
  };
  implemented: Record<string, boolean>;
  note: string;
};

// three.js is large, so the 3D view is split into its own chunk.
const AttentionTerrain = dynamic(() => import("./components/AttentionTerrain"), {
  ssr: false,
  loading: () => <div className="terrainStage loading" />,
});

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const labels: Record<string, string> = {
  akiec: "Actinic keratosis / intraepithelial carcinoma",
  bcc: "Basal cell carcinoma",
  bkl: "Benign keratosis-like lesion",
  df: "Dermatofibroma",
  mel: "Melanoma",
  nv: "Melanocytic nevus",
  vasc: "Vascular lesion",
};

/* Illustrative distribution used only by the hero and story mockups. */
const EXAMPLE: [string, number][] = [
  ["mel", 0.62], ["nv", 0.21], ["bkl", 0.08], ["bcc", 0.05],
  ["akiec", 0.02], ["df", 0.01], ["vasc", 0.01],
];

/* Mirrors the variants in backend/app/model.py stress_test. */
const VARIANTS = [
  { name: "Original", brightness: 1, contrast: 1, blur: 0, param: "—" },
  { name: "Darker", brightness: 0.65, contrast: 1, blur: 0, param: "brightness 0.65" },
  { name: "Brighter", brightness: 1.35, contrast: 1, blur: 0, param: "brightness 1.35" },
  { name: "Lower contrast", brightness: 1, contrast: 0.65, blur: 0, param: "contrast 0.65" },
  { name: "Blur", brightness: 1, contrast: 1, blur: 1.5, param: "gaussian r 1.5" },
];

const variantFilter = (name: string) => {
  const v = VARIANTS.find((item) => item.name === name);
  return v
    ? `brightness(${v.brightness}) contrast(${v.contrast}) blur(${v.blur}px)`
    : "none";
};

const STORY = [
  {
    tab: "Analyze",
    title: "What does the model predict?",
    body: "Every class gets a probability. The runner-up matters as much as the winner.",
  },
  {
    tab: "Attention",
    title: "What influenced it?",
    body: "Grad-CAM highlights the regions that drove the selected class.",
  },
  {
    tab: "Robustness",
    title: "Does it survive a worse photo?",
    body: "The same image is re-scored darker, brighter, flatter and blurred.",
  },
  {
    tab: "Evidence",
    title: "What has actually been measured?",
    body: "Implemented methodology is kept apart from held-out results.",
  },
];

const STORY_VARIANTS = [
  { top: "MEL", conf: "62%" },
  { top: "NV", conf: "44%", flip: true },
  { top: "MEL", conf: "57%" },
  { top: "MEL", conf: "51%" },
  { top: "MEL", conf: "55%" },
];

const METHODS = [
  ["Lesion-level train / val / test split", true],
  ["Class-weighted training", true],
  ["Normalized predictive entropy", true],
  ["Grad-CAM attention maps", true],
  ["Perturbation stress test", true],
  ["Final held-out test metrics", false],
] as const;

const MODES: { id: ScopeMode; label: string; icon: string }[] = [
  {
    id: "original",
    label: "Original",
    icon: "M2 8c1.6-3 3.6-4.5 6-4.5S12.4 5 14 8c-1.6 3-3.6 4.5-6 4.5S3.6 11 2 8zM8 6a2 2 0 100 4 2 2 0 000-4z",
  },
  {
    id: "attention",
    label: "Attention",
    icon: "M8 2v2M8 12v2M2 8h2M12 8h2M8 5.5a2.5 2.5 0 100 5 2.5 2.5 0 000-5z",
  },
  {
    id: "contour",
    label: "Contour",
    icon: "M3 5V3h2M11 3h2v2M13 11v2h-2M5 13H3v-2M5 8h6M8 5v6",
  },
];

const CALLOUTS: Record<ScopeMode, { kicker: string; body: string }> = {
  original: {
    kicker: "Input",
    body: "Dermoscopy image, resized to 224 × 224 and normalized before inference.",
  },
  attention: {
    kicker: "Grad-CAM peak",
    body: "The strongest contribution sits in the darker, asymmetric core — not the surrounding skin.",
  },
  contour: {
    kicker: "Region of interest",
    body: "Border and symmetry axes drawn for inspection. Nothing here is measured.",
  },
};

const RING_R = 288;
const RING_C = 2 * Math.PI * RING_R;
const RING_COLORS = ["#5B8CFF", "#7C8FAE", "#63779A", "#526584", "#455670", "#3B4A61", "#344257"];

function ringSegments() {
  const gap = 10;
  const usable = RING_C - gap * EXAMPLE.length;
  let start = 0;
  return EXAMPLE.map(([code, p], i) => {
    const len = Math.max(p * usable, 2);
    const seg = {
      code: code.toUpperCase(),
      pct: `${Math.round(p * 100)}%`,
      color: RING_COLORS[i],
      dash: `${len.toFixed(1)} ${RING_C.toFixed(1)}`,
      offset: (-start).toFixed(1),
    };
    start += len + gap;
    return seg;
  });
}

function ringTicks() {
  let d = "";
  for (let i = 0; i < 144; i++) {
    const a = (i / 144) * Math.PI * 2;
    const r1 = i % 12 === 0 ? 302 : 309;
    const r2 = 316;
    d += `M${(320 + r1 * Math.cos(a)).toFixed(1)} ${(320 + r1 * Math.sin(a)).toFixed(1)}L${(320 + r2 * Math.cos(a)).toFixed(1)} ${(320 + r2 * Math.sin(a)).toFixed(1)}`;
  }
  return d;
}

const pct = (value: number, digits = 0) => `${(value * 100).toFixed(digits)}%`;

function Arrow({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
      <path
        d="M3 8h10M9 4l4 4-4 4"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function BrandMark({ size = 30 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 30 30" aria-hidden="true">
      <circle cx="15" cy="15" r="13" fill="none" stroke="currentColor" strokeWidth="2" />
      <circle cx="17.5" cy="12.5" r="6" fill="#2462E0" />
    </svg>
  );
}

function scrollToId(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
}

export default function Home() {
  const [research, setResearch] = useState<ResearchStatus | null>(null);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    Promise.allSettled([
      fetch(`${API}/research-status`).then((r) => (r.ok ? r.json() : null)),
      fetch(`${API}/health`).then((r) => r.ok),
    ]).then(([researchResult, healthResult]) => {
      if (researchResult.status === "fulfilled" && researchResult.value) {
        setResearch(researchResult.value);
      }
      setApiOnline(healthResult.status === "fulfilled" ? healthResult.value : false);
    });
  }, []);

  return (
    <main>
      <div className="announcement">
        <span className="announceDot" />
        <span>DermaLens research preview — probability, attention and robustness in one workflow.</span>
        <a href="#sandbox">Try the sandbox →</a>
      </div>

      <Nav />

      <Hero />
      <Story />
      <Terrain />
      <Sandbox research={research} apiOnline={apiOnline} />
      <RobustnessLab />
      <Evidence research={research} />

      <section className="responsibility">
        <div>
          <span className="kicker">Responsible use</span>
          <h2>
            A model score is <em>not a diagnosis.</em>
          </h2>
        </div>
        <div className="responsibilityCopy">
          <p>
            DermaLens is an educational research prototype. Image quality, dataset
            composition, skin-tone representation, hardware and distribution shift can
            all change model behavior.
          </p>
          <p className="strong">Any concerning lesion should be evaluated by a qualified clinician.</p>
        </div>
      </section>

      <footer>
        <span className="brand footerBrand">
          <BrandMark size={26} />
          <span>DermaLens</span>
        </span>
        <div className="footerMeta">
          <span>Educational research prototype</span>
          <span>Not a medical device</span>
          <span>2026</span>
        </div>
        <a href="#top">Back to top ↑</a>
        {/* Decorative wordmark, drawn as SVG so it isn't read or audited as text. */}
        <svg className="footerWordmark" viewBox="0 0 1000 200" aria-hidden="true" focusable="false">
          <text x="500" y="168" textAnchor="middle">DermaLens</text>
        </svg>
      </footer>
    </main>
  );
}

/* ───────────────────────── Nav ───────────────────────── */

const NAV_LINKS = [
  { id: "product", label: "How it works" },
  { id: "sandbox", label: "Sandbox" },
  { id: "robustness", label: "Robustness" },
  { id: "evidence", label: "Evidence" },
];

function Nav() {
  const [active, setActive] = useState<string | null>(null);
  const [pill, setPill] = useState<{ left: number; width: number } | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const linkRefs = useRef<Record<string, HTMLAnchorElement | null>>({});
  const navRef = useRef<HTMLElement | null>(null);

  // Small-screen menu: close on Escape or a tap outside.
  useEffect(() => {
    if (!menuOpen) return;
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && setMenuOpen(false);
    const onPointer = (event: PointerEvent) => {
      if (!navRef.current?.contains(event.target as Node)) setMenuOpen(false);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onPointer);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onPointer);
    };
  }, [menuOpen]);

  // Scrollspy: the last section whose top has passed 45% of the viewport wins.
  // Methodology (#research) belongs to Evidence; the hero (#top) clears it.
  useEffect(() => {
    const ids = ["top", ...NAV_LINKS.map((link) => link.id), "research"];
    let frame = 0;
    function update() {
      frame = 0;
      const line = window.innerHeight * 0.45;
      let current: string | null = null;
      for (const id of ids) {
        const node = document.getElementById(id);
        if (node && node.getBoundingClientRect().top <= line) current = id;
      }
      setActive(current === "research" ? "evidence" : current === "top" ? null : current);
    }
    function onScroll() {
      if (!frame) frame = requestAnimationFrame(update);
    }
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, []);

  useEffect(() => {
    function measure() {
      const link = active ? linkRefs.current[active] : null;
      setPill(link ? { left: link.offsetLeft, width: link.offsetWidth } : null);
    }
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [active]);

  return (
    <nav className="floatingNav" ref={navRef}>
      <a className="brand" href="#top" aria-label="DermaLens home">
        <BrandMark />
        <span>DermaLens</span>
      </a>
      <div className="navLinks">
        <span
          className="navPill"
          aria-hidden="true"
          style={pill ? { transform: `translateX(${pill.left}px)`, width: pill.width, opacity: 1 } : { opacity: 0 }}
        />
        {NAV_LINKS.map(({ id, label }) => (
          <a
            key={id}
            href={`#${id}`}
            ref={(node) => {
              linkRefs.current[id] = node;
            }}
            className={active === id ? "active" : ""}
            aria-current={active === id ? "location" : undefined}
          >
            {label}
          </a>
        ))}
      </div>
      <button className="navCta" onClick={() => scrollToId("sandbox")}>
        Open sandbox <Arrow />
      </button>
      <button
        className="navMenuButton"
        aria-expanded={menuOpen}
        aria-controls="nav-menu"
        aria-label={menuOpen ? "Close menu" : "Open menu"}
        onClick={() => setMenuOpen((open) => !open)}
      >
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
          {menuOpen ? (
            <path d="M4 4l10 10M14 4L4 14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          ) : (
            <path d="M3 6h12M3 12h12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          )}
        </svg>
      </button>
      {menuOpen && (
        <div className="navMenu" id="nav-menu">
          {[...NAV_LINKS.slice(0, 1), { id: "terrain", label: "3D attention" }, ...NAV_LINKS.slice(1)].map(
            ({ id, label }) => (
              <a
                key={id}
                href={`#${id}`}
                className={active === id ? "active" : ""}
                aria-current={active === id ? "location" : undefined}
                onClick={() => setMenuOpen(false)}
              >
                {label}
              </a>
            )
          )}
        </div>
      )}
    </nav>
  );
}

/* ───────────────────────── Hero ───────────────────────── */

function Hero() {
  const [mode, setMode] = useState<ScopeMode>("attention");
  const [hovered, setHovered] = useState<string | null>(null);
  const instrumentRef = useRef<HTMLDivElement | null>(null);
  const segments = useMemo(ringSegments, []);
  const ticks = useMemo(ringTicks, []);
  const callout = CALLOUTS[mode];

  // Layered parallax toward the cursor (2D only, so text stays crisp);
  // skipped for touch and reduced motion.
  function tilt(event: ReactPointerEvent<HTMLDivElement>) {
    const node = instrumentRef.current;
    if (!node || event.pointerType !== "mouse") return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const rect = node.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width - 0.5;
    const y = (event.clientY - rect.top) / rect.height - 0.5;
    node.style.setProperty("--px", x.toFixed(3));
    node.style.setProperty("--py", y.toFixed(3));
  }

  function resetTilt() {
    instrumentRef.current?.style.setProperty("--px", "0");
    instrumentRef.current?.style.setProperty("--py", "0");
  }

  return (
    <section className="hero" id="top">
      <div className="heroRings" aria-hidden="true">
        <i />
        <i />
        <i />
      </div>

      <div className="heroCopy">
        <span className="kicker">Explainable dermoscopy research</span>
        <h1>
          See what
          <br />
          the model
          <br />
          <em>sees.</em>
        </h1>
        <p>
          Inspect the full probability distribution, the pixels behind it, and whether
          the answer survives a worse photo — for every skin-lesion image you upload.
        </p>
        <div className="heroActions">
          <button className="btnDark" onClick={() => scrollToId("sandbox")}>
            Analyze an image <Arrow />
          </button>
          <button className="btnGhost" onClick={() => scrollToId("evidence")}>
            Model card &amp; evidence
          </button>
        </div>
        <ul className="trustRow">
          <li>Not a medical device</li>
          <li>No invented metrics</li>
          <li>Open methodology</li>
        </ul>
      </div>

      <div
        className="instrument"
        ref={instrumentRef}
        onPointerMove={tilt}
        onPointerLeave={resetTilt}
      >
        <div className="instrumentHead">
          <span>
            <i className="pulseDot" />
            Scope · EfficientNet-B0 · 224×224
          </span>
          <span className="instrumentTag">Illustrative output</span>
        </div>

        <div className="scopeStage">
          <svg className="scopeRing" viewBox="0 0 640 640" aria-hidden="true">
            <g className="spinSlowRev">
              <path d={ticks} fill="none" stroke="#3A4E6D" strokeWidth="1" />
            </g>
            <g className="spinSlow">
              <circle cx="320" cy="320" r="270" fill="none" stroke="#2A3B56" strokeDasharray="2 10" />
            </g>
            <g
              transform="rotate(-90 320 320)"
              className={hovered ? "ringFocus" : undefined}
            >
              <circle cx="320" cy="320" r={RING_R} fill="none" stroke="#16233A" strokeWidth="10" />
              {segments.map((s) => (
                <circle
                  key={s.code}
                  className={hovered === s.code ? "ringSegment on" : "ringSegment"}
                  onPointerEnter={() => setHovered(s.code)}
                  onPointerLeave={() => setHovered(null)}
                  cx="320"
                  cy="320"
                  r={RING_R}
                  fill="none"
                  stroke={s.color}
                  strokeWidth="10"
                  strokeLinecap="round"
                  strokeDasharray={s.dash}
                  strokeDashoffset={s.offset}
                />
              ))}
            </g>
          </svg>
          <div className="scopeLens">
            <LesionScope mode={mode} />
          </div>
        </div>

        <div className="floatCard floatScore">
          <span className="monoLabel">Highest score</span>
          <strong className="serifName">Melanoma</strong>
          <strong className="serifBig">62%</strong>
          <div className="meter"><i style={{ width: "62%" }} /></div>
        </div>

        <div className="floatCard floatLegend">
          {segments.slice(0, 4).map((s) => (
            <span
              key={s.code}
              className={hovered === s.code ? "on" : undefined}
              onPointerEnter={() => setHovered(s.code)}
              onPointerLeave={() => setHovered(null)}
            >
              <i style={{ background: s.color }} />
              <span>{s.code}</span>
              <b>{s.pct}</b>
            </span>
          ))}
        </div>

        <div className="floatCard floatStats">
          <span><span>Uncertainty</span><b>38%</b></span>
          <span><span>Entropy</span><b>0.41</b></span>
          <span><span>Stability</span><b>4 / 5</b></span>
        </div>

        <div className="floatCard floatCallout" key={mode}>
          <span className="monoLabel accent">{callout.kicker}</span>
          <p>{callout.body}</p>
          {mode === "attention" && (
            <div className="heatLegend">
              <div className="heatScale" />
              <div><span>Low</span><span>Contribution</span><span>High</span></div>
            </div>
          )}
        </div>

        <div className="modeSwitch" role="group" aria-label="Scope view">
          {MODES.map((m) => (
            <button
              key={m.id}
              aria-pressed={mode === m.id}
              className={mode === m.id ? "active" : ""}
              onClick={() => setMode(m.id)}
            >
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
                <path
                  d={m.icon}
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <div className="facts">
        <div><strong>7 classes</strong><span>HAM10000 diagnostic categories, from nevus to melanoma.</span></div>
        <div><strong>224 × 224</strong><span>EfficientNet-B0 transfer learning on ImageNet-normalized input.</span></div>
        <div><strong>Grad-CAM</strong><span>Attention from the final convolutional block, returned with each prediction.</span></div>
        <div><strong>5 conditions</strong><span>Original, darker, brighter, lower contrast and blur — re-scored on demand.</span></div>
      </div>
    </section>
  );
}

/* ───────────────────────── Story ───────────────────────── */

function Story() {
  const [step, setStep] = useState(0);
  const [paused, setPaused] = useState(false);
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLElement | null>(null);

  // Start paused for people who asked their system to reduce motion.
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) setPaused(true);
  }, []);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { threshold: 0.35 }
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (paused || !visible) return;
    const timer = window.setTimeout(() => setStep((s) => (s + 1) % STORY.length), 5000);
    return () => window.clearTimeout(timer);
  }, [step, paused, visible]);

  const autoplaying = !paused && visible;

  return (
    <section className="story" id="product" ref={ref}>
      <div className="storyInner">
        <div className="storyNav">
          <div className="storyHead">
            <span className="kicker">How it works</span>
            <h2>
              One image.
              <br />
              <em>Four questions.</em>
            </h2>
          </div>

          <div className="storySteps">
            {STORY.map((item, index) => {
              const active = index === step;
              return (
                <button
                  key={item.tab}
                  className={active ? "storyStep active" : "storyStep"}
                  aria-pressed={active}
                  onClick={() => {
                    setStep(index);
                    setPaused(true);
                  }}
                >
                  <span className="storyStepTitle">
                    <span className="mono">0{index + 1}</span>
                    <span>{item.title}</span>
                  </span>
                  {active && (
                    <>
                      <span className="storyStepBody">{item.body}</span>
                      <span className="storyProgress">
                        <i
                          key={`${step}-${autoplaying}`}
                          className={autoplaying ? "running" : "full"}
                        />
                      </span>
                    </>
                  )}
                </button>
              );
            })}
          </div>

          <div className="storyFooter">
            <button
              className="storyPause"
              onClick={() => setPaused((p) => !p)}
              aria-label={paused ? "Resume automatic steps" : "Pause automatic steps"}
            >
              <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
                {paused ? (
                  <path d="M3 2l9 5-9 5z" fill="currentColor" />
                ) : (
                  <path d="M3 2h3v10H3zM8 2h3v10H8z" fill="currentColor" />
                )}
              </svg>
              {paused ? "Play" : "Pause"}
            </button>
            <a className="storyLink" href="#sandbox">Try it on your own image →</a>
          </div>
        </div>

        <div className="storyScreen">
          <div className="storyChrome">
            <span className="chromeDots"><i /><i /><i /></span>
            <span className="chromeTabs">
              {STORY.map((item, index) => (
                <span key={item.tab} className={index === step ? "active" : ""}>
                  {item.tab}
                </span>
              ))}
            </span>
            <span className="monoLabel">Illustrative</span>
          </div>

          <div className="storyPane" key={step}>
            {step === 0 && (
              <div className="paneAnalyze">
                <div className="paneLens"><LesionScope /></div>
                <div className="paneProbs">
                  <span className="monoLabel">Full distribution · 7 classes</span>
                  {EXAMPLE.map(([key, value], i) => (
                    <div key={key} className={i === 0 ? "probRow top" : "probRow"}>
                      <div><span>{labels[key]}</span><b>{pct(value)}</b></div>
                      <div className="track"><i style={{ width: `${Math.max(value * 100, 1)}%` }} /></div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {step === 1 && (
              <div className="paneAttention">
                <div className="paneLens large glow"><LesionScope mode="attention" /></div>
                <div className="paneText">
                  <span className="monoLabel">Grad-CAM · final conv block</span>
                  <h3>Which pixels pushed the top score?</h3>
                  <p>
                    Warm regions contributed most to the selected class. It separates a
                    prediction from the visual evidence behind it.
                  </p>
                  <div className="heatLegend">
                    <div className="heatScale" />
                    <div><span>Low</span><span>High</span></div>
                  </div>
                  <span className="caution">Attention is not a diagnosis.</span>
                </div>
              </div>
            )}

            {step === 2 && (
              <div className="paneRobust">
                <div className="variantGrid">
                  {VARIANTS.map((v, i) => (
                    <div key={v.name} className={STORY_VARIANTS[i].flip ? "variant flip" : "variant"}>
                      <div className="variantLens">
                        <LesionScope brightness={v.brightness} contrast={v.contrast} blur={v.blur} />
                      </div>
                      <strong>{v.name}</strong>
                      <span className="mono">{v.param}</span>
                      <span className="mono result">
                        {STORY_VARIANTS[i].top} · {STORY_VARIANTS[i].conf}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="stabilityBand">
                  <div>
                    <span className="monoLabel">Top-class stability</span>
                    <strong>80%</strong>
                  </div>
                  <p>
                    Four of five conditions kept the same top class. The darker capture
                    flipped it — a signal to distrust this image, not to trust the score.
                  </p>
                </div>
              </div>
            )}

            {step === 3 && (
              <div className="paneEvidence">
                <span className="monoLabel">Held-out evaluation</span>
                <h3>Only show what has been measured.</h3>
                <div className="methodRows">
                  {METHODS.map(([name, done]) => (
                    <div key={name}>
                      <span>{name}</span>
                      <b className={done ? "ok" : "pending"}>
                        <i />
                        {done ? "Implemented" : "Pending training"}
                      </b>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ───────────────────────── 3D attention terrain ───────────────────────── */

function Terrain() {
  const ref = useRef<HTMLElement | null>(null);
  const [near, setNear] = useState(false);

  // Fetch three.js only when the section is about to scroll into view.
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setNear(true);
          observer.disconnect();
        }
      },
      { rootMargin: "600px 0px" }
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return (
    <section className="terrainSection" id="terrain" ref={ref}>
      <div className="terrainCopy">
        <span className="kicker">Attention landscape</span>
        <h2>
          Explore attention <em>in 3D.</em>
        </h2>
        <p>
          The same Grad-CAM idea, raised into terrain: the higher the ground, the more
          that region pushed the model toward its top score. Drag to orbit, hover to
          read a point.
        </p>
        <ul className="terrainKey">
          <li><i className="peak" />Peaks: strongest contribution</li>
          <li><i className="flat" />Flat ground: little influence</li>
        </ul>
        <p className="terrainNote">
          Illustrative: a synthetic lesion with example attention, not model output.
        </p>
      </div>
      {near ? <AttentionTerrain /> : <div className="terrain"><div className="terrainStage loading" /></div>}
    </section>
  );
}

/* ───────────────────────── Sandbox ───────────────────────── */

function Sandbox({
  research,
  apiOnline,
}: {
  research: ResearchStatus | null;
  apiOnline: boolean | null;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [stress, setStress] = useState<StressResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [stressLoading, setStressLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [view, setView] = useState<"original" | "attention">("attention");
  const sampleRef = useRef<SVGSVGElement | null>(null);

  const preview = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);
  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  function chooseFile(next: File | null) {
    setFile(next);
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

  async function loadSample() {
    if (!sampleRef.current) return;
    try {
      chooseFile(await rasterizeSvg(sampleRef.current));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load the sample.");
    }
  }

  async function post<T>(path: string, fallback: string): Promise<T> {
    if (!file) throw new Error(fallback);
    const form = new FormData();
    form.append("file", file);
    const response = await fetch(`${API}${path}`, { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail ?? fallback);
    return data as T;
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    setStress(null);
    try {
      setResult(await post<Prediction>("/predict", "Analysis failed."));
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Could not reach the DermaLens API. Is the backend running?"
          : err instanceof Error
            ? err.message
            : "Analysis failed."
      );
    } finally {
      setLoading(false);
    }
  }

  async function runStressTest() {
    if (!file || !result || result.demo_mode) return;
    setStressLoading(true);
    setError("");
    try {
      setStress(await post<StressResponse>("/stress-test", "Stress test failed."));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Stress test failed.");
    } finally {
      setStressLoading(false);
    }
  }

  const sorted = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];
  const live = result && !result.demo_mode;

  // Plain-language reading of the output; scores are model outputs, not certainty.
  const trust = (() => {
    if (!live || sorted.length < 2) return null;
    const [[topKey, top], [secondKey, second]] = sorted;
    const band =
      top >= 0.8
        ? { level: "High model score", note: "The model strongly prefers one class. That is still not the same as being right." }
        : top >= 0.5
          ? { level: "Moderate model score", note: "The model leans toward one class but other classes remain plausible." }
          : { level: "Low model score", note: "The model is unsure. Treat this output as inconclusive." };
    const closeCall =
      top - second < 0.15 ? `Close call between ${labels[topKey] ?? topKey} and ${labels[secondKey] ?? secondKey}.` : null;
    const mel = research?.evaluation?.per_class?.find((row) => row.label === "mel");
    const melRecord =
      mel && typeof mel.sensitivity === "number"
        ? `On ${mel.support} held-out melanoma images, this model caught ${pct(mel.sensitivity)} and missed ${pct(1 - mel.sensitivity)}.`
        : null;
    return { ...band, closeCall, melRecord };
  })();
  const showHeat = live && view === "attention" && !!result.heatmap_data_url;
  const level = !result ? (loading ? 1 : 0) : stress ? 2 : 1;

  const health = research?.model_loaded
    ? { cls: "live", text: "Model online" }
    : apiOnline
      ? { cls: "api", text: "API online · weights pending" }
      : { cls: "", text: apiOnline === null ? "Checking API…" : "API offline" };

  return (
    <section className="sandboxSection" id="sandbox">
      <div className="sectionIntro">
        <div>
          <span className="kicker">Sandbox</span>
          <h2>
            Try the research
            <br />
            workflow <em>yourself.</em>
          </h2>
        </div>
        <p>
          Upload an image, read the whole distribution, see what the network attended
          to, then stress-test the same image under worse conditions.
        </p>
      </div>

      {/* Off-screen source for the synthetic sample image. */}
      <div className="offscreen" aria-hidden="true">
        <LesionScope svgRef={sampleRef} />
      </div>

      <div className="sandboxWindow">
        <div className="sandboxTopbar">
          <strong className="sandboxBrand"><BrandMark size={24} /> DermaLens Sandbox</strong>
          <span className="crumbs">
            {["Input", "Output", "Robustness"].map((label, i) => (
              <span key={label} className={i === level ? "active" : i < level ? "done" : ""}>
                <b>0{i + 1}</b>
                {label}
              </span>
            ))}
          </span>
          <span className="healthPill">
            <i className={`healthDot ${health.cls}`} />
            {health.text}
          </span>
        </div>

        <div className="sandboxBody">
          <div className="uploadColumn">
            <span className="stepLabel">01 / Input</span>

            {!file ? (
              <>
                <label
                  className={dragging ? "uploadDrop dragging" : "uploadDrop"}
                  onDragEnter={() => setDragging(true)}
                  onDragLeave={() => setDragging(false)}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={onDrop}
                >
                  <span className="uploadCircle">
                    <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true">
                      <path
                        d="M12 16V5M7 10l5-5 5 5M5 19h14"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.8"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </span>
                  <strong>Drop a dermoscopy image</strong>
                  <span>JPEG, PNG or WebP · up to 10 MB</span>
                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    aria-label="Choose a skin-lesion image for analysis"
                    onChange={(event: ChangeEvent<HTMLInputElement>) =>
                      chooseFile(event.target.files?.[0] ?? null)
                    }
                  />
                </label>
                <button className="btnOutline" onClick={loadSample}>
                  Use a synthetic sample
                </button>
              </>
            ) : (
              <>
                <div className="previewStage">
                  <div className="previewLens">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={showHeat ? result.heatmap_data_url! : preview}
                      alt={showHeat ? "Grad-CAM overlay of the selected image" : "Selected lesion preview"}
                    />
                  </div>
                  {loading && <div className="scanLine" />}
                  <span className="monoLabel previewTag">
                    {loading ? "Analyzing…" : showHeat ? "Grad-CAM overlay" : "Input"}
                  </span>
                </div>
                <div className="fileRow">
                  <span className="mono">{file.name}</span>
                  <button className="btnOutline small" onClick={() => chooseFile(null)}>
                    Replace
                  </button>
                </div>
              </>
            )}

            <button className="analyzeButton" onClick={analyze} disabled={!file || loading}>
              <span>{loading ? "Running analysis…" : result ? "Analyze again" : "Analyze image"}</span>
              <Arrow size={18} />
            </button>
            {error && <p className="error" role="alert">{error}</p>}
          </div>

          <div className="resultColumn" aria-live="polite">
            <span className="stepLabel">02 / Output</span>

            {loading ? (
              <div className="analyzing">
                <div className="logLines mono">
                  <span style={{ animationDelay: ".1s" }}>› Decoding image and converting to RGB</span>
                  <span style={{ animationDelay: ".5s" }}>› Resizing to 224 × 224, ImageNet normalization</span>
                  <span style={{ animationDelay: ".9s" }}>› EfficientNet-B0 forward pass · 7-way softmax</span>
                  <span style={{ animationDelay: "1.3s" }}>› Grad-CAM on the final convolutional block</span>
                  <span className="accent" style={{ animationDelay: "1.7s" }}>› Computing normalized entropy</span>
                </div>
                <div className="skeleton">
                  <i style={{ height: 52, width: "60%" }} />
                  <i style={{ width: "100%" }} />
                  <i style={{ width: "82%" }} />
                  <i style={{ width: "64%" }} />
                </div>
              </div>
            ) : !result ? (
              <div className="resultEmpty">
                <svg width="160" height="160" viewBox="0 0 160 160" aria-hidden="true">
                  <circle cx="80" cy="80" r="76" fill="none" stroke="#E1E8F1" />
                  <circle cx="80" cy="80" r="56" fill="none" stroke="#D5DFEB" strokeDasharray="3 6" />
                  <circle cx="80" cy="80" r="34" fill="#F3F6FA" stroke="#D5DFEB" />
                  <circle cx="88" cy="74" r="12" fill="#C9D7EC" />
                </svg>
                <strong>Model output will appear here.</strong>
                <span>Nothing is shown until an image is analyzed — no placeholder predictions.</span>
              </div>
            ) : result.demo_mode ? (
              <div className="demoResult">
                <span className="monoLabel accent">Demo mode</span>
                <h3>The interface works. Trained weights are not loaded yet.</h3>
                <p>DermaLens intentionally does not fabricate a medical prediction.</p>
              </div>
            ) : (
              <div className="liveResult">
                <div className="resultHeadline">
                  <div>
                    <span>Highest model score</span>
                    <h3>{labels[result.top_class] ?? result.top_class}</h3>
                  </div>
                  <strong>{pct(result.confidence)}</strong>
                </div>

                <div className="resultStats">
                  <div><span>Uncertainty</span><b>{pct(result.uncertainty)}</b></div>
                  <div><span>Normalized entropy</span><b>{result.entropy.toFixed(2)}</b></div>
                  <div>
                    <span>Input view</span>
                    <span className="viewToggle">
                      {(["original", "attention"] as const).map((v) => (
                        <button
                          key={v}
                          aria-pressed={view === v}
                          className={view === v ? "active" : ""}
                          disabled={v === "attention" && !result.heatmap_data_url}
                          onClick={() => setView(v)}
                        >
                          {v === "original" ? "Original" : "Grad-CAM"}
                        </button>
                      ))}
                    </span>
                  </div>
                </div>

                {trust && (
                  <div className="trustPanel" role="note">
                    <div className="trustHead">
                      <span className="stepLabel">How to read this</span>
                      <b>{trust.level}</b>
                    </div>
                    <p>{trust.note}</p>
                    {trust.closeCall && <p className="trustFlag">{trust.closeCall}</p>}
                    {trust.melRecord && <p>{trust.melRecord}</p>}
                    <p className="trustStrong">
                      This is not a diagnosis. If a spot is new, changing, bleeding, itchy or
                      worrying you, have it checked by a dermatologist.
                    </p>
                  </div>
                )}

                <div className="probabilityRows">
                  {sorted.map(([key, value], i) => (
                    <div className={i === 0 ? "probRow top" : "probRow"} key={key}>
                      <div><span>{labels[key] ?? key}</span><b>{pct(value, 1)}</b></div>
                      <div className="track">
                        <i className="grow" style={{ width: `${Math.max(value * 100, 0.5)}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {live && (
          <div className="sandboxBottom">
            <span className="stepLabel">03 / Robustness</span>
            <strong>Does the answer survive a worse photo?</strong>
            <button className="btnDark" onClick={runStressTest} disabled={stressLoading}>
              {stressLoading ? "Testing 5 conditions…" : stress ? "Run again" : "Run stress test"}
              <Arrow />
            </button>
          </div>
        )}

        {stress && !stress.demo_mode && (
          <div className="stressResults">
            <div className="stabilityCard">
              <span className="monoLabel">Top-class stability</span>
              <strong>{pct(stress.stability ?? 0)}</strong>
              <span>
                {stress.results.filter((r) => r.top_class === stress.original_class).length} of{" "}
                {stress.results.length} conditions agree
              </span>
            </div>
            <div className="stressGrid">
              {stress.results.map((row) => {
                const flipped = stress.original_class && row.top_class !== stress.original_class;
                return (
                  <div className={flipped ? "stressTile flip" : "stressTile"} key={row.variant}>
                    <div className="stressThumb">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={preview} alt="" style={{ filter: variantFilter(row.variant) }} />
                    </div>
                    <strong>{row.variant}</strong>
                    <span className="mono">
                      {row.top_class.toUpperCase()} · {pct(row.confidence)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      <p className="sandboxNote">
        Every number in the sandbox comes from the DermaLens API&apos;s <code>/predict</code> and{" "}
        <code>/stress-test</code> endpoints. The synthetic sample is a drawing, not a real lesion.
      </p>
    </section>
  );
}

/* ───────────────────────── Robustness lab ───────────────────────── */

function RobustnessLab() {
  const [b, setB] = useState(0.65);
  const [c, setC] = useState(1);
  const [bl, setBl] = useState(0);

  // A toy response curve for the visual demo; the sandbox runs the real model.
  const penalty = (b < 1 ? 0.55 * (1 - b) : 0.14 * (b - 1)) + 0.31 * Math.abs(c - 1) + 0.047 * bl;
  const mel = Math.max(0.62 - penalty, 0.05);
  const nv = 0.21 + (0.62 - mel) * 1.2;
  const flipped = nv > mel;
  const top = flipped ? nv : mel;
  const delta = Math.round((mel - 0.62) * 100);

  const sliders = [
    { label: "Brightness", value: b, set: setB, min: 0.4, max: 1.6, step: 0.05, fmt: b.toFixed(2) },
    { label: "Contrast", value: c, set: setC, min: 0.4, max: 1.6, step: 0.05, fmt: c.toFixed(2) },
    { label: "Gaussian blur", value: bl, set: setBl, min: 0, max: 4, step: 0.1, fmt: `r ${bl.toFixed(1)}` },
  ];

  return (
    <section className="lab" id="robustness">
      <div className="labInner">
        <div className="labHead">
          <div>
            <span className="kicker">Robustness lab</span>
            <h2>
              Break the photo.
              <br />
              <em>Watch the answer.</em>
            </h2>
          </div>
          <div className="presets" role="group" aria-label="Presets from the API stress test">
            {VARIANTS.map((v) => {
              const on =
                Math.abs(b - v.brightness) < 0.001 &&
                Math.abs(c - v.contrast) < 0.001 &&
                Math.abs(bl - v.blur) < 0.001;
              return (
                <button
                  key={v.name}
                  aria-pressed={on}
                  className={on ? "active" : ""}
                  onClick={() => {
                    setB(v.brightness);
                    setC(v.contrast);
                    setBl(v.blur);
                  }}
                >
                  {v.name}
                </button>
              );
            })}
          </div>
        </div>

        <div className="labBody">
          <div className="labControls">
            <span className="monoLabel">Conditions</span>
            {sliders.map((s) => (
              <label key={s.label} className="slider">
                <span><span>{s.label}</span><b className="mono">{s.fmt}</b></span>
                <input
                  type="range"
                  min={s.min}
                  max={s.max}
                  step={s.step}
                  value={s.value}
                  onChange={(event) => s.set(parseFloat(event.target.value))}
                />
              </label>
            ))}
            <p>
              Simulated response for this demo. The sandbox&apos;s stress test re-scores
              darker (0.65), brighter (1.35), lower contrast (0.65) and blur (r 1.5) with
              the real model.
            </p>
          </div>

          <div className="labStage">
            <div className="labCompare">
              <figure>
                <div className="labLens"><LesionScope /></div>
                <figcaption className="monoLabel">Original</figcaption>
              </figure>
              <div className="labDelta">
                <svg width="64" height="24" viewBox="0 0 64 24" aria-hidden="true">
                  <path d="M2 12h58M50 4l10 8-10 8" fill="none" stroke="#3A4E6D" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <span className={delta < -8 ? "mono warn" : "mono"}>
                  {delta >= 0 ? "+" : ""}{delta} pts MEL
                </span>
              </div>
              <figure>
                <div className={flipped ? "labLens flipped" : "labLens"}>
                  <LesionScope brightness={b} contrast={c} blur={bl} />
                </div>
                <figcaption className="monoLabel">Perturbed</figcaption>
              </figure>
            </div>

            <div className="labResults">
              <div><span>Original top class</span><strong>Melanoma · 62%</strong></div>
              <div>
                <span>Perturbed top class</span>
                <strong className={flipped ? "warn" : ""}>
                  {flipped ? "Nevus" : "Melanoma"} · {Math.round(top * 100)}%
                </strong>
              </div>
              <div className={flipped ? "verdict flipped" : "verdict"}>
                <span>Verdict</span>
                <strong>{flipped ? "Top class flipped" : "Top class held"}</strong>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ───────────────────────── Evidence ───────────────────────── */

type CurveRow = NonNullable<ResearchStatus["training"]>[number];

function TrainingCurve({ rows }: { rows: CurveRow[] }) {
  const W = 600, H = 220, L = 44, R = 16, T = 14, B = 34;
  const series = [
    { key: "train_accuracy", label: "Training images", color: "#8d9db4", dash: "5 5" },
    { key: "val_accuracy", label: "Validation images", color: "#2462e0", dash: "" },
    { key: "val_balanced_accuracy", label: "Validation, balanced", color: "#c66a0a", dash: "2 4" },
  ] as const;
  const present = series.filter((s) => rows.some((r) => typeof r[s.key] === "number"));
  const values = rows.flatMap((r) => present.map((s) => r[s.key])).filter((v): v is number => typeof v === "number");
  const lo = Math.max(0, Math.floor(Math.min(...values) * 10) / 10);
  const hi = Math.min(1, Math.ceil(Math.max(...values) * 10) / 10);
  const x = (epoch: number) => L + ((epoch - 1) / Math.max(rows.length - 1, 1)) * (W - L - R);
  const y = (v: number) => T + (1 - (v - lo) / (hi - lo || 1)) * (H - T - B);
  const ticks = Array.from({ length: Math.round((hi - lo) * 10) + 1 }, (_, i) => lo + i / 10);
  const last = rows[rows.length - 1];
  const gap =
    typeof last.train_accuracy === "number" && typeof last.val_accuracy === "number"
      ? last.train_accuracy - last.val_accuracy
      : null;

  return (
    <div className="curve">
      <div className="confusionHead">
        <span className="stepLabel">How training went</span>
        <p>
          Accuracy after each epoch on the images it learned from versus validation images
          it never trained on.
          {gap !== null && gap > 0.05 &&
            ` By the last epoch the gap is ${Math.round(gap * 100)} points — the model was starting to memorise its training images (overfitting). DermaLens keeps the epoch that did best on validation, not the last one.`}
        </p>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Training curve over ${rows.length} epochs. Final training accuracy ${pct(last.train_accuracy ?? 0)}, validation accuracy ${pct(last.val_accuracy ?? 0)}.`}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="#e1e8f1" />
            <text x={L - 8} y={y(t) + 4} textAnchor="end" className="curveAxis">{Math.round(t * 100)}%</text>
          </g>
        ))}
        {rows.map((r) => (
          <text key={r.epoch} x={x(r.epoch)} y={H - 12} textAnchor="middle" className="curveAxis">{r.epoch}</text>
        ))}
        {present.map((s) => {
          const pts = rows.filter((r) => typeof r[s.key] === "number").map((r) => `${x(r.epoch)},${y(r[s.key] as number)}`);
          return (
            <polyline key={s.key} points={pts.join(" ")} fill="none" stroke={s.color} strokeWidth="2.5" strokeDasharray={s.dash} strokeLinejoin="round" />
          );
        })}
      </svg>
      <ul className="curveLegend">
        {present.map((s) => (
          <li key={s.key}>
            <svg width="26" height="8" aria-hidden="true">
              <line x1="0" x2="26" y1="4" y2="4" stroke={s.color} strokeWidth="2.5" strokeDasharray={s.dash} />
            </svg>
            {s.label}
          </li>
        ))}
        <li className="curveAxisNote">Epoch →</li>
      </ul>
    </div>
  );
}

function ConfusionMatrix({ matrix, classes }: { matrix: number[][]; classes: string[] }) {
  return (
    <div className="confusion">
      <div className="confusionHead">
        <span className="stepLabel">Where the mistakes go</span>
        <p>
          Each row is the true class; each column is what the model said. The diagonal is
          correct. Orange cells are mistakes, darker means a larger share of that row.
        </p>
      </div>
      <div className="confusionScroll">
        <table>
          <caption className="srOnly">Confusion matrix on the held-out test split</caption>
          <thead>
            <tr>
              <th scope="col">True ↓ · Said →</th>
              {classes.map((c) => (
                <th scope="col" key={c} title={labels[c] ?? c}>{c.toUpperCase()}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => {
              const total = row.reduce((sum, n) => sum + n, 0) || 1;
              return (
                <tr key={classes[i]} className={classes[i] === "mel" ? "mel" : undefined}>
                  <th scope="row" title={labels[classes[i]] ?? classes[i]}>
                    {classes[i].toUpperCase()}
                  </th>
                  {row.map((count, j) => {
                    const share = count / total;
                    const style =
                      i === j
                        ? { background: `rgba(36, 98, 224, ${0.08 + share * 0.47})` } // ≥ 4.5:1 with ink text
                        : count
                          ? { background: `rgba(232, 131, 26, ${Math.min(0.1 + share * 1.6, 0.9)})` }
                          : undefined;
                    return (
                      <td
                        key={j}
                        style={style}
                        title={`${labels[classes[i]] ?? classes[i]} → ${labels[classes[j]] ?? classes[j]}: ${count} of ${total}`}
                      >
                        {count || ""}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Evidence({ research }: { research: ResearchStatus | null }) {
  const evaluation = research?.evaluation_available ? research.evaluation : null;
  const metrics: [string, number | null | undefined][] = [
    ["Accuracy", evaluation?.accuracy],
    ["Balanced accuracy", evaluation?.balanced_accuracy],
    ["Macro F1", evaluation?.macro_f1],
    ["Macro ROC-AUC", evaluation?.macro_ovr_roc_auc],
  ];

  return (
    <>
      <section className="evidenceSection" id="evidence">
        <div className="evidenceCopy">
          <span className="kicker">Evidence</span>
          <h2>
            No performance claims until the <em>test split</em> says so.
          </h2>
          <p>
            Implemented methodology and measured results live in separate places.
            Held-out metrics appear only after real trained weights have been evaluated.
          </p>
        </div>

        <div className="statusPanel">
          <div className="statusPanelTop">
            <span className="stepLabel">Evaluation status</span>
            <span className={evaluation ? "statusBadge ok" : "statusBadge"}>
              <i />
              {evaluation ? "Available" : "Pending training"}
            </span>
          </div>
          {research?.model_loaded && research.model && (
            <p className="modelLine">
              Model {research.model.version ?? "(unlabelled)"} · EfficientNet-B0 ·{" "}
              {research.model.tta ? "4-view averaged" : "single view"} ·{" "}
              {research.model.calibrated
                ? `tuned on validation${
                    research.model.min_mel_sensitivity_target
                      ? ` (melanoma floor ${pct(research.model.min_mel_sensitivity_target)})`
                      : ""
                  }`
                : "not tuned"}
            </p>
          )}
          <div className="metricGrid">
            {metrics.map(([label, value]) => (
              <div key={label}>
                <span>{label}</span>
                <strong className={typeof value === "number" ? "" : "empty"}>
                  {typeof value === "number" ? pct(value, 1) : "—"}
                </strong>
              </div>
            ))}
          </div>
          {evaluation && (
            <>
              <div className="evalMeta">
                {typeof evaluation.test_images === "number" && (
                  <span>
                    <b>{evaluation.test_images.toLocaleString()}</b> held-out test images
                  </span>
                )}
                {typeof evaluation.expected_calibration_error === "number" && (
                  <span>
                    Calibration error <b>{evaluation.expected_calibration_error.toFixed(3)}</b>
                  </span>
                )}
                {typeof evaluation.multiclass_brier_score === "number" && (
                  <span>
                    Brier score <b>{evaluation.multiclass_brier_score.toFixed(3)}</b>
                  </span>
                )}
              </div>

              <p className="evalCaveat">
                Measured once on held-out HAM10000 images. These numbers say nothing about
                other cameras, clinics or skin tones underrepresented in the dataset. Clinical
                use would need external validation, dermatologist review, prospective testing
                and regulatory approval.
              </p>

              {!!evaluation.per_class?.length && (
                <div className="perClass">
                  <div className="perClassHead">
                    <span>Class</span>
                    <span>Caught (sensitivity)</span>
                    <span>Specificity</span>
                  </div>
                  {evaluation.per_class.map((row) => (
                    <div key={row.label} className={row.label === "mel" ? "perClassRow mel" : "perClassRow"}>
                      <span>
                        {labels[row.label] ?? row.label}
                        <small>{row.support} images</small>
                      </span>
                      <span className="perClassBar">
                        <span className="track">
                          <i style={{ width: `${(row.sensitivity ?? 0) * 100}%` }} />
                        </span>
                        <b>{typeof row.sensitivity === "number" ? pct(row.sensitivity) : "—"}</b>
                      </span>
                      <b>{typeof row.specificity === "number" ? pct(row.specificity) : "—"}</b>
                    </div>
                  ))}
                </div>
              )}

              {evaluation.confusion_matrix && evaluation.classes && (
                <ConfusionMatrix matrix={evaluation.confusion_matrix} classes={evaluation.classes} />
              )}

              {research?.training && research.training.length > 1 && (
                <TrainingCurve rows={research.training} />
              )}
            </>
          )}

          {!evaluation && (
            <div className="pendingEvidence">
              <div className="pendingOrb"><i /><i /></div>
              <div>
                <strong>Waiting for the real evaluation.</strong>
                <p>No placeholder accuracy, no invented benchmark, no diagnostic claim.</p>
              </div>
            </div>
          )}
        </div>
      </section>

      <section className="researchSection" id="research">
        <div className="researchHead">
          <h2>
            Designed to make <em>uncertainty</em> visible.
          </h2>
          <span className="stepLabel">Methodology</span>
        </div>
        <div className="researchCards">
          <article>
            <svg width="44" height="44" viewBox="0 0 44 44" aria-hidden="true">
              <rect x="4" y="10" width="16" height="24" rx="4" fill="none" stroke="#2462E0" strokeWidth="1.6" />
              <rect x="24" y="10" width="16" height="24" rx="4" fill="none" stroke="#13223A" strokeWidth="1.6" />
              <circle cx="12" cy="22" r="3" fill="#2462E0" />
            </svg>
            <span className="mono">01</span>
            <h3>Lesion-level split</h3>
            <p>Images of the same lesion stay within one split, reducing leakage between training and evaluation.</p>
          </article>
          <article>
            <svg width="44" height="44" viewBox="0 0 44 44" aria-hidden="true">
              <path d="M6 36V26M14 36V14M22 36V22M30 36V8M38 36V30" fill="none" stroke="#13223A" strokeWidth="3" strokeLinecap="round" />
              <path d="M6 36V30M22 36V28M38 36V32" fill="none" stroke="#2462E0" strokeWidth="3" strokeLinecap="round" />
            </svg>
            <span className="mono">02</span>
            <h3>Class-aware training</h3>
            <p>Weighted loss keeps the majority nevus class from dominating the learning objective.</p>
          </article>
          <article>
            <svg width="44" height="44" viewBox="0 0 44 44" aria-hidden="true">
              <circle cx="22" cy="22" r="16" fill="none" stroke="#13223A" strokeWidth="1.6" />
              <path d="M22 6a16 16 0 010 32z" fill="#2462E0" opacity=".85" />
            </svg>
            <span className="mono">03</span>
            <h3>Robustness checks</h3>
            <p>Controlled perturbations expose sensitivity to lighting, contrast and blur before anyone trusts a score.</p>
          </article>
        </div>
        <ul className="checkList">
          {["Balanced accuracy & macro F1", "Confusion matrix", "Multiclass ROC-AUC", "Normalized predictive entropy"].map(
            (item) => (
              <li key={item}>
                <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M3 8.5l3 3 7-7" fill="none" stroke="#1E7A4C" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                {item}
              </li>
            )
          )}
        </ul>
      </section>
    </>
  );
}
