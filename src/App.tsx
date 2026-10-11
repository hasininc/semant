import { useState, useEffect, type ReactNode } from "react";

type Page = "playground" | "analytics" | "comparison" | "reports";

type IconName =
  | "shield"
  | "sparkles"
  | "chart"
  | "compare"
  | "file"
  | "arrow"
  | "download"
  | "check"
  | "alert"
  | "play"
  | "chevron"
  | "refresh";

const navItems: { id: Page; label: string; icon: IconName }[] = [
  { id: "playground", label: "Playground", icon: "play" },
  { id: "analytics", label: "Analytics", icon: "chart" },
  { id: "comparison", label: "Model Comparison", icon: "compare" },
  { id: "reports", label: "Reports", icon: "file" },
];

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, ReactNode> = {
    shield: (
      <>
        <path d="M12 3 5 6v5c0 4.6 2.9 8 7 10 4.1-2 7-5.4 7-10V6l-7-3Z" />
        <path d="m9.2 12 1.8 1.8 3.8-4" />
      </>
    ),
    sparkles: (
      <>
        <path d="m12 3 .8 2.2L15 6l-2.2.8L12 9l-.8-2.2L9 6l2.2-.8L12 3Z" />
        <path d="m18 13 .8 2.2L21 16l-2.2.8L18 19l-.8-2.2L15 16l2.2-.8L18 13Z" />
        <path d="m6 11 1.1 2.9L10 15l-2.9 1.1L6 19l-1.1-2.9L2 15l2.9-1.1L6 11Z" />
      </>
    ),
    chart: (
      <>
        <path d="M4 19V9" />
        <path d="M10 19V5" />
        <path d="M16 19v-7" />
        <path d="M22 19H2" />
      </>
    ),
    compare: (
      <>
        <path d="M8 7h12l-3-3" />
        <path d="m20 7-3 3" />
        <path d="M16 17H4l3 3" />
        <path d="m4 17 3-3" />
      </>
    ),
    file: (
      <>
        <path d="M6 3h8l4 4v14H6V3Z" />
        <path d="M14 3v5h4" />
        <path d="M9 13h6M9 17h6" />
      </>
    ),
    arrow: <path d="m9 18 6-6-6-6" />,
    download: (
      <>
        <path d="M12 3v12" />
        <path d="m7 10 5 5 5-5" />
        <path d="M5 21h14" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    alert: (
      <>
        <path d="M12 4 3 20h18L12 4Z" />
        <path d="M12 9v5M12 17.5v.1" />
      </>
    ),
    play: <path d="m8 5 11 7-11 7V5Z" />,
    chevron: <path d="m8 10 4 4 4-4" />,
    refresh: (
      <>
        <path d="M3 12a9 9 0 0 1 15.55-6.36L21 8" />
        <path d="M21 3v5h-5" />
        <path d="M21 12a9 9 0 0 1-15.55 6.36L3 16" />
        <path d="M3 21v-5h5" />
      </>
    ),
  };

  return (
    <svg
      aria-hidden="true"
      className="icon"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name]}
    </svg>
  );
}

function Button({
  children,
  icon,
  onClick,
  variant = "primary",
  type = "button",
  disabled = false,
}: {
  children: ReactNode;
  icon?: IconName;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "quiet";
  type?: "button" | "submit";
  disabled?: boolean;
}) {
  return (
    <button
      className={`button button-${variant}`}
      disabled={disabled}
      onClick={onClick}
      style={disabled ? { opacity: 0.6, cursor: "not-allowed" } : undefined}
      type={type}
    >
      {icon && <Icon name={icon} size={17} />}
      <span>{children}</span>
    </button>
  );
}

function PageHeader({
  eyebrow,
  title,
  subtitle,
}: {
  eyebrow: string;
  title: string;
  subtitle: string;
}) {
  return (
    <header className="page-header">
      <div className="eyebrow">{eyebrow}</div>
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </header>
  );
}

const API_BASE_URL = "http://127.0.0.1:8000";

async function fetchFromApi<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, options);
  } catch {
    response = await fetch(`/api${path}`, options);
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;

    try {
      const err = await response.json();
      if (err?.detail) detail = err.detail;
    } catch {
      // The response may not contain JSON.
    }

    throw new Error(detail);
  }

  return response.json();
}

type ApiPrediction = {
  text: string;
  sentiment: "Positive" | "Negative";
  confidence: number;
  probabilities: {
    negative: number;
    positive: number;
  };
  model: string;
};

type BackendMetrics = {
  model: string;
  dataset: string;
  training_samples: number;
  testing_samples: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix?: number[][];
};

type PerturbationMetrics = {
  accuracy: number;
  accuracy_drop?: number;
  flip_rate?: number;
  prediction_flip_rate?: number;
};

type RobustnessModelResult = {
  model: string;
  original_accuracy: number;
  perturbations: Record<string, PerturbationMetrics>;
};

type RobustnessResponse = {
  results: RobustnessModelResult[];
};

type ApiStatus = "checking" | "online" | "offline";

function AppNav({
  page,
  setPage,
  apiStatus,
  onRefreshStatus,
}: {
  page: Page;
  setPage: (page: Page) => void;
  apiStatus: ApiStatus;
  onRefreshStatus: () => void;
}) {
  return (
    <nav className="glass nav-shell" aria-label="Primary navigation">
      <button
        className="brand"
        onClick={() => setPage("playground")}
        type="button"
      >
        <span className="brand-icon">
          <Icon name="shield" size={22} />
        </span>
        <span className="brand-copy">
          <strong>SemanticShield</strong>
          <small>Test. Compare. Trust.</small>
        </span>
      </button>

      <div className="nav-links">
        {navItems.map((item) => (
          <button
            className={page === item.id ? "nav-item active" : "nav-item"}
            key={item.id}
            onClick={() => setPage(item.id)}
            type="button"
          >
            <Icon name={item.icon} size={16} />
            <span>{item.label}</span>
          </button>
        ))}
      </div>

      <div
        className={`status ${apiStatus}`}
        onClick={onRefreshStatus}
        role="button"
        tabIndex={0}
        title={
          apiStatus === "online"
            ? "API connected at http://127.0.0.1:8000 (click to test connection)"
            : "Cannot reach http://127.0.0.1:8000 (click to retry)"
        }
      >
        <span className={`status-dot ${apiStatus}`} />
        {apiStatus === "online" && "API Online · Port 8000"}
        {apiStatus === "offline" && "Backend Offline (Port 8000)"}
        {apiStatus === "checking" && "Connecting..."}
      </div>
    </nav>
  );
}

const samples = {
  "Positive Review":
    "The product exceeded my expectations and the support team was wonderfully helpful.",
  "Negative Review":
    "The app feels slow, confusing, and I regret paying for the subscription.",
  "Ambiguous Text":
    "It works differently than I expected, though that is not necessarily a bad thing.",
};

type PerturbationType =
  | "Spelling Changes"
  | "Paraphrase"
  | "Negation / Contrast"
  | "Custom Edit";

function generatePerturbation(
  input: string,
  type: PerturbationType,
): string {
  if (!input.trim()) return input;

  if (type === "Spelling Changes") {
    const typoDict: Record<string, string> = {
      the: "teh",
      this: "ths",
      that: "taht",
      with: "wiht",
      great: "graet",
      good: "gud",
      awesome: "awsome",
      terrible: "terrible",
      horrible: "horible",
      recommend: "recommed",
      wonderfully: "wonderfuly",
      helpful: "helpfull",
      product: "prduct",
      service: "servise",
      because: "becuase",
      really: "realy",
      definitely: "definitly",
      excellent: "excelent",
      worst: "wrst",
      best: "bst",
      love: "luv",
      loved: "lovd",
      hate: "h8",
      slow: "sloow",
      confusing: "confusin",
      subscription: "subcription",
      different: "diffrent",
    };

    const words = input.split(/(\s+|[.,!?;:])/);
    let altered = false;

    const newWords = words.map((word) => {
      const lower = word.toLowerCase();

      if (typoDict[lower]) {
        altered = true;
        const replacement = typoDict[lower];

        return word[0] === word[0].toUpperCase() && word.length > 1
          ? replacement[0].toUpperCase() + replacement.slice(1)
          : replacement;
      }

      return word;
    });

    if (altered) {
      return newWords.join("");
    }

    return input
      .split(" ")
      .map((word, index) => {
        if (index % 2 === 0 && word.length > 4) {
          return (
            word.slice(0, 2) +
            word[3] +
            word[2] +
            word.slice(4)
          );
        }

        return word;
      })
      .join(" ");
  }

  if (type === "Paraphrase") {
    const map: Record<string, string> = {
      "exceeded my expectations": "was better than I anticipated",
      "wonderfully helpful": "extremely supportive",
      "feels slow": "lacks prompt responsiveness",
      confusing: "unintuitive",
      "regret paying": "wish I had not purchased",
      subscription: "membership plan",
      "works differently": "functions in an unexpected manner",
      "not necessarily a bad thing": "not automatically a negative",
      great: "solid",
      terrible: "decidedly underwhelming",
      amazing: "quite impressive",
      loved: "strongly appreciated",
      hated: "was disappointed by",
      excellent: "remarkably good",
      "support team": "customer assistance staff",
    };

    let result = input;
    let found = false;

    for (const [key, replacement] of Object.entries(map)) {
      const regex = new RegExp(`\\b${key}\\b`, "gi");

      if (regex.test(result)) {
        result = result.replace(regex, replacement);
        found = true;
      }
    }

    if (!found) {
      if (result.toLowerCase().startsWith("the ")) {
        result = `In practice, ${result.charAt(0).toLowerCase() + result.slice(1)}`;
      } else {
        result = `From my perspective, ${result.charAt(0).toLowerCase() + result.slice(1)}`;
      }
    }

    return result;
  }

  if (type === "Negation / Contrast") {
    if (input.toLowerCase().includes("not ")) {
      return input.replace(/\bnot\s+([a-zA-Z]+)/gi, "hardly $1");
    }

    return `${input.replace(/[.]+$/, "")}, or so it initially seemed.`;
  }

  return input;
}

function Playground({
  goToAnalytics,
  onModelTested,
}: {
  goToAnalytics: () => void;
  onModelTested?: () => void;
}) {
  const [text, setText] = useState("");
  const [model, setModel] = useState("Logistic Regression (TF-IDF)");
  const [state, setState] = useState<
    "idle" | "loading" | "result" | "error"
  >("idle");
  const [showRobustness, setShowRobustness] = useState(false);
  const [perturbation, setPerturbation] =
    useState<PerturbationType>("Spelling Changes");
  const [modifiedText, setModifiedText] = useState("");
  const [isEditingVariant, setIsEditingVariant] = useState(false);

  const [result, setResult] = useState<ApiPrediction | null>(null);
  const [variantResult, setVariantResult] = useState<ApiPrediction | null>(null);
  const [testingVariant, setTestingVariant] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    if (!isEditingVariant && text.trim()) {
      setModifiedText(generatePerturbation(text, perturbation));
      setVariantResult(null);
    }
  }, [text, perturbation, isEditingVariant]);

  async function analyze() {
    const trimmed = text.trim();

    if (!trimmed) {
      setErrorMessage(
        "Please enter some text before running the sentiment analysis.",
      );
      setState("error");
      return;
    }

    setState("loading");
    setErrorMessage("");
    setShowRobustness(false);
    setVariantResult(null);

    try {
      const prediction = await fetchFromApi<ApiPrediction>("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed }),
      });

      setResult(prediction);
      setModifiedText(generatePerturbation(trimmed, perturbation));
      setState("result");
      onModelTested?.();
    } catch (error) {
      console.error("Prediction error:", error);

      setErrorMessage(
        error instanceof Error
          ? `${error.message}. Ensure the FastAPI server is running at http://127.0.0.1:8000.`
          : "Could not connect to the prediction API at http://127.0.0.1:8000.",
      );

      setState("error");
    }
  }

  async function runRobustnessTest() {
    const trimmedVariant = modifiedText.trim();

    if (!trimmedVariant) return;

    setTestingVariant(true);

    try {
      const prediction = await fetchFromApi<ApiPrediction>("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmedVariant }),
      });

      setVariantResult(prediction);
    } catch (error) {
      console.error("Variant prediction error:", error);

      setErrorMessage(
        error instanceof Error
          ? error.message
          : "Could not evaluate the modified variant.",
      );
    } finally {
      setTestingVariant(false);
    }
  }

  const isConsistent =
    result && variantResult && result.sentiment === variantResult.sentiment;

  const confidenceDiff =
    result && variantResult
      ? result.confidence - variantResult.confidence
      : 0;

  const isDegraded = isConsistent && confidenceDiff > 10;

  const isFlipped =
    result && variantResult && result.sentiment !== variantResult.sentiment;

  return (
    <main className="page playground-page">
      <PageHeader
        eyebrow="TEXT ROBUSTNESS LAB"
        subtitle="Evaluate text classifiers against real-world spelling changes, typos, and paraphrases to discover when predictions fail."
        title="Understand how robust your AI really is."
      />

      <section className="glass input-card">
        <div className="card-topline">
          <div>
            <span className="step-number">01</span>
            <h2>Enter your text</h2>
          </div>
          <span className="sample-badge">Try a sample</span>
        </div>

        <div className={`textarea-wrap ${state === "error" ? "has-error" : ""}`}>
          <textarea
            aria-label="Text to analyze"
            maxLength={1000}
            onChange={(event) => {
              setText(event.target.value);

              if (state === "error") {
                setState("idle");
                setErrorMessage("");
              }
            }}
            placeholder="Type a review, customer comment, or sentence to analyze..."
            value={text}
          />

          <span className="char-count">{text.length} / 1000</span>
        </div>

        {state === "error" && (
          <div className="error-message" role="alert">
            <Icon name="alert" size={15} />
            <span>{errorMessage}</span>
          </div>
        )}

        <div className="samples">
          {Object.entries(samples).map(([label, sample]) => (
            <button
              className="sample-chip"
              key={label}
              onClick={() => {
                setText(sample);
                setState("idle");
                setErrorMessage("");
                setResult(null);
                setVariantResult(null);
                setShowRobustness(false);
                setIsEditingVariant(false);
              }}
              type="button"
            >
              <span className={`chip-dot ${label.split(" ")[0].toLowerCase()}`} />
              {label}
            </button>
          ))}
        </div>

        <div className="input-actions">
          <label className="select-field">
            <span>Classification model</span>

            <span className="select-wrap">
              <select
                value={model}
                onChange={(event) => setModel(event.target.value)}
              >
                <option value="Logistic Regression (TF-IDF)">
                  TF-IDF + Logistic Regression (Live Model)
                </option>
              </select>
              <Icon name="chevron" size={16} />
            </span>
          </label>

          <Button
            disabled={state === "loading"}
            icon={state === "loading" ? undefined : "play"}
            onClick={analyze}
          >
            {state === "loading" ? "Analyzing with Model…" : "Analyze Text"}
          </Button>
        </div>
      </section>

      {state === "idle" && (
        <div className="empty-state">
          <span className="empty-icon">
            <Icon name="chart" size={20} />
          </span>
          <p>
            Your model prediction, confidence scores, and probability
            distribution will appear here.
          </p>
        </div>
      )}

      {state === "result" && result && (
        <section className="glass result-card">
          <div className="result-heading">
            <div>
              <span className="section-kicker">
                PREDICTION RESULT ·{" "}
                {result.model || "TF-IDF + Logistic Regression"}
              </span>

              <h2>
                <span
                  className={`sentiment-dot ${result.sentiment.toLowerCase()}`}
                />
                {result.sentiment} sentiment
              </h2>
            </div>

            <div className="confidence-orb">
              <strong>{result.confidence}%</strong>
              <span>confidence</span>
            </div>
          </div>

          <div className="confidence-track">
            <span style={{ width: `${result.confidence}%` }} />
          </div>

          {result.probabilities && (
            <div className="probability-breakdown">
              <div className="prob-item">
                <span className="prob-label">Positive</span>

                <div className="prob-bar-track">
                  <div
                    className="prob-bar-fill positive"
                    style={{ width: `${result.probabilities.positive}%` }}
                  />
                </div>

                <span className="prob-value">
                  {result.probabilities.positive}%
                </span>
              </div>

              <div className="prob-item">
                <span className="prob-label">Negative</span>

                <div className="prob-bar-track">
                  <div
                    className="prob-bar-fill negative"
                    style={{ width: `${result.probabilities.negative}%` }}
                  />
                </div>

                <span className="prob-value">
                  {result.probabilities.negative}%
                </span>
              </div>
            </div>
          )}

          <div className="explanation">
            <Icon name="shield" size={17} />
            <p>
              {result.sentiment === "Positive"
                ? `The trained classifier identified predominantly positive lexical markers and optimistic phrasing with ${result.confidence}% confidence.`
                : `The trained classifier identified critical or negative phrasing and sentiment markers with ${result.confidence}% confidence.`}
            </p>
          </div>

          <div className="result-actions">
            <Button
              icon="shield"
              onClick={() => {
                setShowRobustness(true);
                setVariantResult(null);
              }}
            >
              Test Robustness
            </Button>

            <Button
              icon="chart"
              onClick={goToAnalytics}
              variant="secondary"
            >
              View Model Analytics
            </Button>
          </div>
        </section>
      )}

      {state === "result" && result && showRobustness && (
        <section className="robustness-section">
          <div className="section-header">
            <div>
              <span className="section-kicker">
                ADVERSARIAL & ROBUSTNESS TEST
              </span>
              <h2>See how modifications affect your model</h2>
            </div>

            <div className="segment-control">
              {(
                [
                  "Spelling Changes",
                  "Paraphrase",
                  "Negation / Contrast",
                  "Custom Edit",
                ] as const
              ).map((option) => (
                <button
                  className={perturbation === option ? "selected" : ""}
                  key={option}
                  onClick={() => {
                    setPerturbation(option);
                    setVariantResult(null);

                    if (option === "Custom Edit") {
                      setIsEditingVariant(true);
                    } else {
                      setIsEditingVariant(false);
                      setModifiedText(generatePerturbation(text, option));
                    }
                  }}
                  type="button"
                >
                  {option}
                </button>
              ))}
            </div>
          </div>

          <div className="text-comparison">
            <article className="glass text-panel">
              <div className="panel-label">
                <span>Original sentence</span>
                <span className="panel-tag">CONTROL</span>
              </div>

              <p>{text}</p>

              <PredictionBadge
                confidence={result.confidence}
                label={result.sentiment}
              />
            </article>

            <div className="comparison-arrow">
              <Icon name="arrow" size={20} />
            </div>

            <article className="glass text-panel modified">
              <div className="panel-label">
                <span>Modified sentence</span>
                <span className="panel-tag">
                  {perturbation === "Custom Edit"
                    ? "EDITABLE VARIANT"
                    : "PERTURBATION"}
                </span>
              </div>

              {perturbation === "Custom Edit" || isEditingVariant ? (
                <textarea
                  className="editable-variant-input"
                  onChange={(event) => {
                    setModifiedText(event.target.value);
                    setVariantResult(null);
                  }}
                  value={modifiedText}
                />
              ) : (
                <p>{modifiedText}</p>
              )}

              {variantResult && (
                <PredictionBadge
                  confidence={variantResult.confidence}
                  label={variantResult.sentiment}
                />
              )}
            </article>
          </div>

          <div className="test-footer">
            {variantResult ? (
              <>
                {isFlipped && (
                  <div className="inconsistent-result">
                    <span className="result-alert danger">
                      <Icon name="alert" size={17} />
                    </span>

                    <div>
                      <strong>
                        Adversarial Vulnerability: Prediction Flipped!
                      </strong>
                      <small>
                        Classification changed from {result.sentiment} (
                        {result.confidence}%) to {variantResult.sentiment} (
                        {variantResult.confidence}%).
                      </small>
                    </div>
                  </div>
                )}

                {isDegraded && !isFlipped && (
                  <div className="degraded-result">
                    <span className="result-alert warning">
                      <Icon name="alert" size={17} />
                    </span>

                    <div>
                      <strong>
                        Confidence Degraded (-{confidenceDiff.toFixed(1)}%)
                      </strong>
                      <small>
                        Maintained {variantResult.sentiment} classification,
                        but confidence dropped from {result.confidence}% to{" "}
                        {variantResult.confidence}%.
                      </small>
                    </div>
                  </div>
                )}

                {isConsistent && !isDegraded && (
                  <div className="consistent-result">
                    <span className="result-check">
                      <Icon name="check" size={17} />
                    </span>

                    <div>
                      <strong>Prediction Robust & Consistent</strong>
                      <small>
                        The model remained stable with{" "}
                        {variantResult.confidence}% confidence after
                        modification.
                      </small>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <p>
                Click "Run Test" to evaluate the modified sentence with the
                live model.
              </p>
            )}

            <Button
              disabled={testingVariant || !modifiedText.trim()}
              icon={testingVariant ? undefined : "play"}
              onClick={runRobustnessTest}
            >
              {testingVariant ? "Testing Variant…" : "Run Test"}
            </Button>
          </div>
        </section>
      )}
    </main>
  );
}

function PredictionBadge({
  label,
  confidence,
}: {
  label: string;
  confidence: number;
}) {
  return (
    <div className="prediction-badge">
      <span className={`sentiment-dot ${label.toLowerCase()}`} />
      {label} · {confidence}%
    </div>
  );
}

function MetricCard({
  label,
  value,
  change,
  tone,
}: {
  label: string;
  value: string;
  change: string;
  tone: string;
}) {
  return (
    <article className={`metric-card ${tone}`}>
      <div className="metric-top">
        <span>{label}</span>
      </div>
      <strong>{value}</strong>
      <small>
        <b>{change}</b>
      </small>
    </article>
  );
}

function BarChart({
  series,
}: {
  series: { label: string; first: number; second?: number }[];
}) {
  return (
    <div className="bar-chart">
      {series.map((item) => (
        <div className="bar-group" key={item.label}>
          <div className="bars">
            <span
              className="bar bar-primary"
              style={{ height: `${item.first}%` }}
            >
              <em>{item.first.toFixed(1)}</em>
            </span>

            {item.second !== undefined && (
              <span
                className="bar bar-secondary"
                style={{ height: `${item.second}%` }}
              >
                <em>{item.second.toFixed(1)}</em>
              </span>
            )}
          </div>

          <small>{item.label}</small>
        </div>
      ))}
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  children,
  legend,
  badge = "EVALUATION DATA",
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
  legend?: string[];
  badge?: string;
}) {
  return (
    <article className="glass chart-card">
      <div className="chart-head">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>

        <span className="sample-data">{badge}</span>
      </div>

      {children}

      {legend && (
        <div className="legend">
          {legend.map((item, index) => (
            <span key={item}>
              <i className={index ? "secondary" : ""} />
              {item}
            </span>
          ))}
        </div>
      )}
    </article>
  );
}

function RobustnessTable({
  data,
}: {
  data: RobustnessModelResult[] | null;
}) {
  if (!data || data.length === 0) {
    return (
      <div className="empty-state">
        <p>
          Robustness results are not available yet. Run{" "}
          <code>python evaluate_robustness.py</code> in the backend folder,
          then refresh the API status.
        </p>
      </div>
    );
  }

  const wordModel =
    data.find((item) => item.model.toLowerCase().includes("word")) ?? data[0];

  const charModel =
    data.find((item) => item.model.toLowerCase().includes("character")) ??
    data[1];

  const perturbations = [
    { key: "character_deletion", label: "Character deletion" },
    { key: "character_swap", label: "Character swap" },
    { key: "character_replacement", label: "Character replacement" },
    { key: "character_repetition", label: "Character repetition" },
  ];

  const getFlipRate = (
    model: RobustnessModelResult | undefined,
    key: string,
  ) => {
    if (!model?.perturbations?.[key]) return null;

    const item = model.perturbations[key];

    if (typeof item.prediction_flip_rate === "number") {
      const value = item.prediction_flip_rate;
      return `${(value <= 1 ? value * 100 : value).toFixed(2)}%`;
    }

    return typeof item.flip_rate === "number"
      ? `${item.flip_rate.toFixed(2)}%`
      : "—";
  };

  return (
    <div className="comparison-table robustness-table-wrap">
      <div className="table-row table-head">
        <span>Perturbation</span>
        <span>Word-level model</span>
        <span>Character-level model</span>
      </div>

      {perturbations.map((item) => (
        <div className="table-row" key={item.key}>
          <span>{item.label}</span>
          <span>{getFlipRate(wordModel, item.key)}</span>
          <span>{getFlipRate(charModel, item.key)}</span>
        </div>
      ))}

      <p className="sample-note">
        Prediction flip rate: percentage of predictions that changed after a
        character edit. Lower means more stable; it does not by itself measure
        correctness.
      </p>
    </div>
  );
}

function Analytics({
  back,
  metricsData,
  robustnessData,
}: {
  back: () => void;
  metricsData: BackendMetrics | null;
  robustnessData: RobustnessModelResult[] | null;
}) {
  const accuracyPercent = metricsData
    ? `${(metricsData.accuracy * 100).toFixed(2)}%`
    : "—";

  const f1Percent = metricsData
    ? `${(metricsData.f1_score * 100).toFixed(2)}%`
    : "—";

  const precisionPercent = metricsData
    ? `${(metricsData.precision * 100).toFixed(2)}%`
    : "—";

  const recallPercent = metricsData
    ? `${(metricsData.recall * 100).toFixed(2)}%`
    : "—";

  const cm = metricsData?.confusion_matrix;

  return (
    <main className="page">
      <PageHeader
        eyebrow="PERFORMANCE OVERVIEW"
        subtitle="Metrics from your trained IMDb sentiment model and measured character-perturbation tests."
        title="Model Analytics"
      />

      <section className="evaluation-snapshot">
        <div className="snapshot-intro">
          <span className="section-kicker">LATEST EVALUATION</span>
          <h2>IMDb Test Set Summary</h2>
          <p>
            {metricsData?.model || "TF-IDF + Logistic Regression"} ·{" "}
            {metricsData?.testing_samples?.toLocaleString() ?? "—"} test reviews
          </p>
          <span className="sample-note">
            {metricsData
              ? "Live data from backend /metrics"
              : "Waiting for backend metrics"}
          </span>
        </div>

        <div className="metrics-grid">
          <MetricCard
            label="Clean Accuracy"
            value={accuracyPercent}
            change="Test set"
            tone="lilac"
          />
          <MetricCard
            label="Precision"
            value={precisionPercent}
            change="Test set"
            tone="blue"
          />
          <MetricCard
            label="F1 Score"
            value={f1Percent}
            change="Test set"
            tone="pink"
          />
        </div>
      </section>

      <div className="analytics-grid">
        <ChartCard
          badge={metricsData ? "LIVE MODEL METRICS" : "NO BACKEND DATA"}
          subtitle="Accuracy, precision, recall and F1 from the held-out test set."
          title="Baseline model performance"
        >
          <BarChart
            series={
              metricsData
                ? [
                    {
                      label: "Accuracy",
                      first: metricsData.accuracy * 100,
                    },
                    {
                      label: "Precision",
                      first: metricsData.precision * 100,
                    },
                    {
                      label: "Recall",
                      first: metricsData.recall * 100,
                    },
                    {
                      label: "F1 Score",
                      first: metricsData.f1_score * 100,
                    },
                  ]
                : []
            }
          />
        </ChartCard>

        <ChartCard
          badge="TEST SET CONFUSION MATRIX"
          subtitle="Actual labels compared with model predictions."
          title="Confusion Matrix"
        >
          {cm ? (
            <table className="confusion-matrix-table">
              <thead>
                <tr>
                  <th>Actual / Predicted</th>
                  <th>Negative</th>
                  <th>Positive</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th>Actual Negative</th>
                  <td className="cm-cell correct">
                    {cm[0]?.[0] ?? "—"}
                    <br />
                    <small>True Negative</small>
                  </td>
                  <td className="cm-cell incorrect">
                    {cm[0]?.[1] ?? "—"}
                    <br />
                    <small>False Positive</small>
                  </td>
                </tr>
                <tr>
                  <th>Actual Positive</th>
                  <td className="cm-cell incorrect">
                    {cm[1]?.[0] ?? "—"}
                    <br />
                    <small>False Negative</small>
                  </td>
                  <td className="cm-cell correct">
                    {cm[1]?.[1] ?? "—"}
                    <br />
                    <small>True Positive</small>
                  </td>
                </tr>
              </tbody>
            </table>
          ) : (
            <p className="sample-note">
              Confusion matrix will appear when the backend metrics load.
            </p>
          )}
        </ChartCard>

        <ChartCard
          badge="LIVE ROBUSTNESS RESULTS"
          subtitle="Prediction flip rate after each character perturbation. Lower is more stable."
          title="Robustness by change type"
        >
          <RobustnessTable data={robustnessData} />
        </ChartCard>

        <article className="glass failures-card">
          <div className="chart-head">
            <div>
              <h3>Experiment scope</h3>
              <p>What the current robustness test measures</p>
            </div>
            <span className="sample-data">METHODOLOGY</span>
          </div>

          <div className="failure-list">
            <FailureItem
              label="Character deletion"
              tone="lilac"
              value="Tested"
            />
            <FailureItem
              label="Character swap"
              tone="blue"
              value="Tested"
            />
            <FailureItem
              label="Replacement and repetition"
              tone="pink"
              value="Tested"
            />
          </div>

          <p className="sample-note">
            These tests alter one character in one randomly selected word per
            review. They do not yet measure general paraphrase robustness.
          </p>
        </article>
      </div>

      <div className="page-footer-action">
        <Button icon="arrow" onClick={back} variant="secondary">
          Back to Playground
        </Button>
      </div>
    </main>
  );
}

function HorizontalBar({
  label,
  values,
}: {
  label: string;
  values: number[];
}) {
  return (
    <div className="horizontal-row">
      <span>{label}</span>
      <div>
        <i className="hbar first" style={{ width: `${values[0]}%` }} />
        <i className="hbar second" style={{ width: `${values[1]}%` }} />
      </div>
    </div>
  );
}

function FailureItem({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: string;
}) {
  return (
    <div className="failure-item">
      <span className={`failure-icon ${tone}`}>
        <Icon name="alert" size={16} />
      </span>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function averageFlipRate(
  model: RobustnessModelResult | undefined,
): number | null {
  if (!model?.perturbations) return null;

  const values = Object.values(model.perturbations)
    .map((item) => {
      if (typeof item.prediction_flip_rate === "number") {
        const value = item.prediction_flip_rate;
        return value <= 1 ? value * 100 : value;
      }

      return typeof item.flip_rate === "number" ? item.flip_rate : null;
    })
    .filter((value): value is number => value !== null);

  return values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : null;
}

function Comparison({
  metricsData,
  robustnessData,
}: {
  metricsData: BackendMetrics | null;
  robustnessData: RobustnessModelResult[] | null;
}) {
  const wordModel = robustnessData?.find((item) =>
    item.model.toLowerCase().includes("word"),
  );

  const charModel = robustnessData?.find((item) =>
    item.model.toLowerCase().includes("character"),
  );

  const wordFlip = averageFlipRate(wordModel);
  const charFlip = averageFlipRate(charModel);

  const percent = (value: number | undefined | null) =>
    value == null ? "—" : `${(value * 100).toFixed(2)}%`;

  const flipPercent = (value: number | null) =>
    value == null ? "—" : `${value.toFixed(2)}%`;

  const models = [
    {
      name: "Word-level TF-IDF",
      type: "WORD N-GRAM BASELINE",
      description:
        "Word unigrams and bigrams with TF-IDF weighting and Logistic Regression.",
      accuracy: metricsData ? percent(metricsData.accuracy) : "—",
      f1: metricsData ? percent(metricsData.f1_score) : "—",
      flip: flipPercent(wordFlip),
      icon: "W",
    },
    {
      name: "Character-level TF-IDF",
      type: "CHARACTER N-GRAM MODEL",
      description:
        "Character n-grams designed to capture subword patterns and some spelling variations.",
      accuracy: charModel ? percent(charModel.original_accuracy) : "—",
      f1: "Not reported",
      flip: flipPercent(charFlip),
      icon: "C",
    },
  ];

  return (
    <main className="page">
      <PageHeader
        eyebrow="HEAD-TO-HEAD EVALUATION"
        subtitle="Compare the two models actually trained and evaluated in this project. No transformer benchmark is shown because one is not connected."
        title="Compare Your Models"
      />

      <div className="model-grid">
        {models.map((model) => (
          <article className="glass model-card" key={model.name}>
            <div className="model-title">
              <span className="model-icon">{model.icon}</span>
              <div>
                <small>{model.type}</small>
                <h2>{model.name}</h2>
              </div>
            </div>

            <p>{model.description}</p>

            <div className="model-stats">
              <div>
                <span>Clean accuracy</span>
                <strong>{model.accuracy}</strong>
              </div>
              <div>
                <span>F1-score</span>
                <strong>{model.f1}</strong>
              </div>
              <div>
                <span>Avg. flip rate</span>
                <strong>{model.flip}</strong>
              </div>
            </div>

            <span className="sample-note">
              {robustnessData
                ? "Measured project results"
                : "Waiting for robustness results"}
            </span>
          </article>
        ))}
      </div>

      <div className="comparison-lower">
        <article className="glass comparison-table-card">
          <div className="chart-head">
            <div>
              <h3>Detailed comparison</h3>
              <p>Values are populated from the backend evaluation files.</p>
            </div>
            <span className="sample-data">PROJECT RESULTS</span>
          </div>

          <div className="comparison-table">
            <div className="table-row table-head">
              <span>Metric</span>
              <span>Word-level</span>
              <span>Character-level</span>
            </div>

            <div className="table-row">
              <span>Clean accuracy</span>
              <span>
                {metricsData ? percent(metricsData.accuracy) : "—"}
              </span>
              <span>
                {charModel ? percent(charModel.original_accuracy) : "—"}
              </span>
            </div>

            <div className="table-row">
              <span>Average prediction flip rate</span>
              <span>{flipPercent(wordFlip)}</span>
              <span>{flipPercent(charFlip)}</span>
            </div>

            <div className="table-row">
              <span>Precision</span>
              <span>
                {metricsData ? percent(metricsData.precision) : "—"}
              </span>
              <span>Not reported</span>
            </div>

            <div className="table-row">
              <span>Recall</span>
              <span>{metricsData ? percent(metricsData.recall) : "—"}</span>
              <span>Not reported</span>
            </div>

            <div className="table-row">
              <span>F1-score</span>
              <span>{metricsData ? percent(metricsData.f1_score) : "—"}</span>
              <span>Not reported</span>
            </div>
          </div>
        </article>

        <ChartCard
          badge="MEASURED RESULTS"
          subtitle="Prediction flip rate by character edit. Lower values indicate fewer changed predictions."
          title="Robustness detail"
        >
          <RobustnessTable data={robustnessData} />
        </ChartCard>
      </div>
    </main>
  );
}

function Reports({
  metricsData,
  robustnessData,
}: {
  metricsData: BackendMetrics | null;
  robustnessData: RobustnessModelResult[] | null;
}) {
  const cleanAccuracy = metricsData
    ? `${(metricsData.accuracy * 100).toFixed(2)}%`
    : "—";

  const wordModel = robustnessData?.find((item) =>
    item.model.toLowerCase().includes("word"),
  );

  const charModel = robustnessData?.find((item) =>
    item.model.toLowerCase().includes("character"),
  );

  const wordFlip = averageFlipRate(wordModel);
  const charFlip = averageFlipRate(charModel);

  const flipText = (value: number | null) =>
    value == null ? "—" : `${value.toFixed(2)}%`;

  const download = (type: string) => {
    if (type === "PDF") {
      window.print();
      return;
    }

    const rows: (string | number)[][] = [
      [
        "model",
        "clean_accuracy_percent",
        "precision_percent",
        "recall_percent",
        "f1_percent",
        "average_prediction_flip_rate_percent",
        "test_samples",
      ],
      [
        metricsData?.model || "Word-level TF-IDF + Logistic Regression",
        metricsData ? (metricsData.accuracy * 100).toFixed(2) : "",
        metricsData ? (metricsData.precision * 100).toFixed(2) : "",
        metricsData ? (metricsData.recall * 100).toFixed(2) : "",
        metricsData ? (metricsData.f1_score * 100).toFixed(2) : "",
        wordFlip == null ? "" : wordFlip.toFixed(2),
        metricsData?.testing_samples ?? "",
      ],
      [
        charModel?.model || "Character-level TF-IDF + Logistic Regression",
        charModel ? (charModel.original_accuracy * 100).toFixed(2) : "",
        "",
        "",
        "",
        charFlip == null ? "" : charFlip.toFixed(2),
        "",
      ],
    ];

    const csvContent = rows
      .map((row) =>
        row
          .map((cell) => `"${String(cell).replace(/"/g, '""')}"`)
          .join(","),
      )
      .join("\n");

    const blob = new Blob([csvContent], {
      type: "text/csv;charset=utf-8;",
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "semanticshield-report.csv";
    link.click();

    URL.revokeObjectURL(url);
  };

  return (
    <main className="page">
      <PageHeader
        eyebrow="REPORT CENTER"
        subtitle="Export the metrics and robustness results generated by your own trained models."
        title="Evaluation Reports"
      />

      <article className="glass report-card">
        <div className="report-top">
          <div className="document-title">
            <span className="document-icon">
              <Icon name="file" size={25} />
            </span>

            <div>
              <span className="section-kicker">
                SEMANTICSHIELD · EVALUATION REPORT
              </span>
              <h2>Text Classification Robustness Report</h2>
              <p>Generated from the current backend evaluation files</p>
            </div>
          </div>

          <span className="complete-badge">
            <Icon name="check" size={14} /> Project report
          </span>
        </div>

        <div className="report-divider" />

        <section className="report-section summary-section">
          <div>
            <span className="report-index">01</span>
            <h3>Overall evaluation summary</h3>
            <p>
              The word-level TF-IDF + Logistic Regression model achieved{" "}
              {cleanAccuracy} accuracy on the held-out IMDb test set.
              Robustness results compare prediction changes after character
              deletion, swapping, replacement and repetition. Flip rate
              measures stability, not whether a prediction is correct.
            </p>
          </div>

          <div className="report-score">
            <strong>{cleanAccuracy}</strong>
            <span>CLEAN ACCURACY</span>
            <small>
              {metricsData
                ? "Measured test-set result"
                : "Waiting for backend"}
            </small>
          </div>
        </section>

        <section className="report-section">
          <div className="report-section-heading">
            <span className="report-index">02</span>
            <h3>Evaluation details</h3>
          </div>

          <div className="details-grid">
            <Detail
              label="Model evaluated"
              value={
                metricsData?.model ||
                "Word-level TF-IDF + Logistic Regression"
              }
            />

            <Detail
              label="Dataset"
              value={
                metricsData
                  ? `${metricsData.dataset} (${metricsData.testing_samples.toLocaleString()} test samples)`
                  : "Not loaded"
              }
            />

            <Detail label="Word model accuracy" value={cleanAccuracy} />

            <Detail
              label="Word model average flip rate"
              value={flipText(wordFlip)}
            />

            <Detail
              label="Character model accuracy"
              value={
                charModel
                  ? `${(charModel.original_accuracy * 100).toFixed(2)}%`
                  : "—"
              }
            />

            <Detail
              label="Character model average flip rate"
              value={flipText(charFlip)}
            />
          </div>
        </section>

        <div className="report-columns">
          <section className="report-section findings">
            <div className="report-section-heading">
              <span className="report-index">03</span>
              <h3>Measured findings</h3>
            </div>

            <ul>
              <li>
                <Icon name="check" size={15} />
                Clean accuracy and classification metrics are read from the
                trained model's metrics file.
              </li>
              <li>
                <Icon name="check" size={15} />
                Character-level and word-level models are compared using
                measured perturbation results.
              </li>
              <li>
                <Icon name="alert" size={15} />
                Current robustness evaluation applies one random character
                edit to one randomly selected word per review; it is not a
                comprehensive paraphrase test.
              </li>
            </ul>
          </section>

          <section className="report-section preview-chart">
            <div className="report-section-heading">
              <span className="report-index">04</span>
              <h3>Prediction flip rate</h3>
            </div>

            <RobustnessTable data={robustnessData} />
          </section>
        </div>

        <section className="report-section failure-example">
          <div className="report-section-heading">
            <span className="report-index">05</span>
            <h3>Interpretation note</h3>
          </div>

          <blockquote>
            Lower prediction flip rate means the model's label changed less
            often after the tested character edits.
            <span>
              It does not guarantee higher accuracy on modified text. Evaluate
              accuracy against ground-truth labels separately for each
              perturbation.
            </span>
          </blockquote>
        </section>

        <div className="report-actions">
          <p>
            Export the current metrics to CSV or use your browser's print
            dialog to save a PDF.
          </p>

          <div>
            <Button
              icon="download"
              onClick={() => download("CSV")}
              variant="secondary"
            >
              Download CSV
            </Button>

            <Button icon="file" onClick={() => download("PDF")}>
              Export PDF
            </Button>
          </div>
        </div>
      </article>
    </main>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("playground");
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [metricsData, setMetricsData] = useState<BackendMetrics | null>(null);
  const [robustnessData, setRobustnessData] =
    useState<RobustnessModelResult[] | null>(null);

  async function checkBackend() {
    setApiStatus("checking");

    try {
      const health = await fetchFromApi<{
        status: string;
        model?: string;
      }>("/health");

      if (health?.status !== "healthy") {
        setApiStatus("offline");
        return;
      }

      setApiStatus("online");

      try {
        const metrics =
          await fetchFromApi<BackendMetrics>("/metrics");
        setMetricsData(metrics);
      } catch (error) {
        console.error("Could not load model metrics:", error);
        setMetricsData(null);
      }

      try {
        const robustness =
          await fetchFromApi<RobustnessResponse>("/robustness");
        setRobustnessData(robustness.results);
      } catch (error) {
        console.error("Could not load robustness results:", error);
        setRobustnessData(null);
      }
    } catch (error) {
      console.error("Backend health check failed:", error);
      setApiStatus("offline");
      setMetricsData(null);
      setRobustnessData(null);
    }
  }

  useEffect(() => {
    void checkBackend();
  }, []);

  return (
    <div className="app">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <div className="ambient ambient-three" />

      <AppNav
        apiStatus={apiStatus}
        onRefreshStatus={checkBackend}
        page={page}
        setPage={setPage}
      />

      {page === "playground" && (
        <Playground
          goToAnalytics={() => setPage("analytics")}
          onModelTested={() => {
            if (apiStatus !== "online") {
              void checkBackend();
            }
          }}
        />
      )}

      {page === "analytics" && (
        <Analytics
          back={() => setPage("playground")}
          metricsData={metricsData}
          robustnessData={robustnessData}
        />
      )}

      {page === "comparison" && (
        <Comparison
          metricsData={metricsData}
          robustnessData={robustnessData}
        />
      )}

      {page === "reports" && (
        <Reports
          metricsData={metricsData}
          robustnessData={robustnessData}
        />
      )}

      <footer className="site-footer">
        <span>
          <Icon name="shield" size={15} /> SemanticShield
        </span>
        <small>
          Robustness insights for trustworthy text classification.
        </small>
      </footer>
    </div>
  );
}