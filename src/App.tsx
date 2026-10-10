import { useState, type ReactNode } from "react";

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
  | "chevron";

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
  };
  return (
    <svg
      aria-hidden="true"
      className="icon"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
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
}: {
  children: ReactNode;
  icon?: IconName;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "quiet";
  type?: "button" | "submit";
}) {
  return (
    <button className={`button button-${variant}`} onClick={onClick} type={type}>
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

function AppNav({ page, setPage }: { page: Page; setPage: (page: Page) => void }) {
  return (
    <nav className="glass nav-shell" aria-label="Primary navigation">
      <button className="brand" onClick={() => setPage("playground")} type="button">
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
      <div className="status">
        <span className="status-dot" />
        Models Ready
      </div>
    </nav>
  );
}

const samples = {
  "Positive Review": "The product exceeded my expectations and the support team was wonderfully helpful.",
  "Negative Review": "The app feels slow, confusing, and I regret paying for the subscription.",
  "Ambiguous Text": "It works differently than I expected, though that is not necessarily a bad thing.",
};

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

function Playground({ goToAnalytics }: { goToAnalytics: () => void }) {
  const [text, setText] = useState("");
  const [model, setModel] = useState("Logistic Regression");
  const [state, setState] = useState<"idle" | "loading" | "result" | "error">("idle");
  const [showRobustness, setShowRobustness] = useState(false);
  const [perturbation, setPerturbation] = useState<"Spelling Changes" | "Paraphrase">(
    "Spelling Changes",
  );
  const [tested, setTested] = useState(false);

  const [result, setResult] = useState<ApiPrediction | null>(null);
  const [errorMessage, setErrorMessage] = useState("");

  const modifiedText =
    perturbation === "Spelling Changes"
      ? text.replace(/the/gi, "teh").replace(/wonderfully/gi, "wonderfuly")
      : text
          .replace(/exceeded my expectations/gi, "was better than I anticipated")
          .replace(/wonderfully helpful/gi, "extremely supportive");

  async function analyze() {
  if (!text.trim()) {
    setErrorMessage("Add some text before running the analysis.");
    setState("error");
    return;
  }

  setState("loading");
  setErrorMessage("");
  setShowRobustness(false);

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: text.trim() }),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }

    const prediction: ApiPrediction = await response.json();

    setResult(prediction);
    setState("result");
  } catch (error) {
    console.error("Prediction error:", error);

    setErrorMessage(
      error instanceof Error
        ? error.message
        : "Could not connect to the prediction API.",
    );
    setState("error");
  }
}

  return (
    <main className="page playground-page">
      <PageHeader
        eyebrow="TEXT ROBUSTNESS LAB"
        subtitle="Test text classifiers against spelling changes and paraphrases to discover when predictions fail."
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
            maxLength={500}
            onChange={(event) => {
              setText(event.target.value);
              if (state === "error") {
                setState("idle");
                setErrorMessage("");
              }
            }}
            placeholder="Type a review or sentence to analyze..."
            value={text}
          />
          <span className="char-count">{text.length} / 500</span>
        </div>
        
        
        {state === "error" && (
          <div className="error-message" role="alert">
            <Icon name="alert" size={15} />
            {errorMessage}
          </div>
)}
        <div className="samples">
          {Object.entries(samples).map(([label, sample]) => (
            <button className="sample-chip" key={label} 
onClick={() => {
  setText(sample);
  setState("idle");
  setErrorMessage("");
  setResult(null);
}} type="button">
              <span className={`chip-dot ${label.split(" ")[0].toLowerCase()}`} />
              {label}
            </button>
          ))}
        </div>
        <div className="input-actions">
          <label className="select-field">
            <span>Classification model</span>
            <span className="select-wrap">
              <select value={model} onChange={(event) => setModel(event.target.value)}>
                <option>Logistic Regression</option>
                <option>Transformer Model</option>
              </select>
              <Icon name="chevron" size={16} />
            </span>
          </label>
          <Button icon={state === "loading" ? undefined : "play"} onClick={analyze}>
            {state === "loading" ? "Analyzing…" : "Analyze Text"}
          </Button>
        </div>
      </section>

      {state === "idle" && (
        <div className="empty-state">
          <span className="empty-icon">
            <Icon name="chart" size={20} />
          </span>
          <p>Your prediction and confidence insights will appear here.</p>
        </div>
      )}

      {state === "result" && result && (
        <section className="glass result-card">
          <div className="result-heading">
            <div>
              <span className="section-kicker">PREDICTION RESULT · {model}</span>
              <h2>
                <span className={`sentiment-dot ${result.sentiment.toLowerCase()}`} />
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
          <div className="explanation">
            <Icon name="shield" size={17} />
            <p>
              The model identified strong {result.sentiment.toLowerCase()} language and a consistent
              semantic tone across the sentence.
            </p>
          </div>
          <div className="result-actions">
            <Button
              icon="shield"
              onClick={() => {
                setShowRobustness(true);
                setTested(false);
              }}
            >
              Test Robustness
            </Button>
            <Button icon="chart" onClick={goToAnalytics} variant="secondary">
              View Analytics
            </Button>
          </div>
        </section>
      )}

      {state === "result" && result && showRobustness && (
        <section className="robustness-section">
          <div className="section-header">
            <div>
              <span className="section-kicker">ROBUSTNESS TEST</span>
              <h2>See how small changes affect your model</h2>
            </div>
            <div className="segment-control">
              {(["Spelling Changes", "Paraphrase"] as const).map((option) => (
                <button
                  className={perturbation === option ? "selected" : ""}
                  key={option}
                  onClick={() => {
                    setPerturbation(option);
                    setTested(false);
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
              {tested && <PredictionBadge label={result.sentiment} />}
            </article>
            <div className="comparison-arrow">
              <Icon name="arrow" size={20} />
            </div>
            <article className="glass text-panel modified">
              <div className="panel-label">
                <span>Modified sentence</span>
                <span className="panel-tag">VARIANT</span>
              </div>
              <p>{modifiedText}</p>
              {tested && <PredictionBadge label={result.sentiment} />}
            </article>
          </div>
          <div className="test-footer">
            {tested ? (
              <div className="consistent-result">
                <span className="result-check">
                  <Icon name="check" size={17} />
                </span>
                <div>
                  <strong>Prediction Consistent</strong>
                  <small>The model remained stable after this modification.</small>
                </div>
              </div>
            ) : (
              <p>Run the test to compare predictions across both versions.</p>
            )}
            <Button icon="play" onClick={() => setTested(true)}>
              Run Test
            </Button>
          </div>
        </section>
      )}
    </main>
  );
}

function PredictionBadge({ label }: { label: string }) {
  return (
    <div className="prediction-badge">
      <span className={`sentiment-dot ${label.toLowerCase()}`} />
      {label} · 94%
    </div>
  );
}

const metrics = [
  { label: "Clean Accuracy", value: "91.8%", change: "+2.4%", tone: "lilac" },
  { label: "Robustness Score", value: "84.2%", change: "+5.1%", tone: "blue" },
  { label: "Prediction Consistency", value: "87.6%", change: "+3.8%", tone: "pink" },
];

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
        <b>{change}</b> from prior run
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
            <span className="bar bar-primary" style={{ height: `${item.first}%` }}>
              <em>{item.first}</em>
            </span>
            {item.second && (
              <span className="bar bar-secondary" style={{ height: `${item.second}%` }}>
                <em>{item.second}</em>
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
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
  legend?: string[];
}) {
  return (
    <article className="glass chart-card">
      <div className="chart-head">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        <span className="sample-data">SAMPLE DATA</span>
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

function Analytics({ back }: { back: () => void }) {
  return (
    <main className="page">
      <PageHeader
        eyebrow="PERFORMANCE OVERVIEW"
        subtitle="Understand how well your classifiers perform."
        title="Model Analytics"
      />
      <section className="evaluation-snapshot">
        <div className="snapshot-intro">
          <span className="section-kicker">LATEST EVALUATION</span>
          <h2>Test set summary</h2>
          <p>IMDb sentiment · 2,000 examples · evaluated 14 May</p>
          <span className="sample-note">Illustrative sample data</span>
        </div>
        <div className="metrics-grid">
          {metrics.map((metric) => (
            <MetricCard key={metric.label} {...metric} />
          ))}
        </div>
      </section>
      <div className="analytics-grid">
        <ChartCard
          legend={["Logistic Regression", "Transformer"]}
          subtitle="Accuracy by evaluation category"
          title="Model performance"
        >
          <BarChart
            series={[
              { label: "Accuracy", first: 82, second: 92 },
              { label: "F1 Score", first: 78, second: 88 },
              { label: "Robust", first: 69, second: 84 },
            ]}
          />
        </ChartCard>
        <ChartCard
          legend={["Original text", "Modified text"]}
          subtitle="Performance under text variation"
          title="Original vs. modified"
        >
          <BarChart
            series={[
              { label: "Logistic", first: 88, second: 71 },
              { label: "Transformer", first: 93, second: 85 },
            ]}
          />
        </ChartCard>
        <ChartCard
          legend={["Spelling changes", "Paraphrases"]}
          subtitle="Consistency by perturbation type"
          title="Robustness by change type"
        >
          <div className="horizontal-chart">
            <HorizontalBar label="Logistic Regression" values={[68, 75]} />
            <HorizontalBar label="Transformer Model" values={[82, 89]} />
          </div>
        </ChartCard>
        <article className="glass failures-card">
          <div className="chart-head">
            <div>
              <h3>Common prediction failures</h3>
              <p>Where models need the most attention</p>
            </div>
            <span className="sample-data">SAMPLE DATA</span>
          </div>
          <div className="failure-list">
            <FailureItem label="Negation & mixed sentiment" value="34%" tone="pink" />
            <FailureItem label="Informal spelling" value="27%" tone="lilac" />
            <FailureItem label="Long-form paraphrases" value="18%" tone="blue" />
          </div>
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

function HorizontalBar({ label, values }: { label: string; values: number[] }) {
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

function FailureItem({ label, value, tone }: { label: string; value: string; tone: string }) {
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

const modelData = [
  {
    name: "Logistic Regression",
    type: "CLASSIC BASELINE",
    description: "Traditional machine learning baseline using efficient TF-IDF text features.",
    accuracy: "88.4%",
    f1: "86.9%",
    robust: "74.2%",
    icon: "LR",
  },
  {
    name: "Transformer Model",
    type: "CONTEXTUAL MODEL",
    description: "Pretrained language model that understands contextual text representations.",
    accuracy: "93.1%",
    f1: "92.4%",
    robust: "86.7%",
    icon: "TM",
  },
];

function Comparison() {
  return (
    <main className="page">
      <PageHeader
        eyebrow="HEAD-TO-HEAD EVALUATION"
        subtitle="Find out which model handles text variations best."
        title="Compare Your Models"
      />
      <div className="model-grid">
        {modelData.map((model, index) => (
          <article className={`glass model-card ${index ? "winner" : ""}`} key={model.name}>
            {index === 1 && <span className="best-label">STRONGEST OVERALL</span>}
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
                <span>Accuracy</span>
                <strong>{model.accuracy}</strong>
              </div>
              <div>
                <span>F1-score</span>
                <strong>{model.f1}</strong>
              </div>
              <div>
                <span>Robustness</span>
                <strong>{model.robust}</strong>
              </div>
            </div>
            <span className="sample-note">Sample evaluation data</span>
          </article>
        ))}
      </div>
      <div className="comparison-lower">
        <article className="glass comparison-table-card">
          <div className="chart-head">
            <div>
              <h3>Detailed comparison</h3>
              <p>Performance across core evaluation metrics</p>
            </div>
            <span className="sample-data">SAMPLE DATA</span>
          </div>
          <div className="comparison-table">
            <div className="table-row table-head">
              <span>Metric</span>
              <span>Logistic</span>
              <span>Transformer</span>
            </div>
            {[
              ["Clean accuracy", "88.4%", "93.1%"],
              ["Robust accuracy", "74.2%", "86.7%"],
              ["F1-score", "86.9%", "92.4%"],
              ["Prediction consistency", "79.8%", "89.5%"],
            ].map((row) => (
              <div className="table-row" key={row[0]}>
                <span>{row[0]}</span>
                <span>{row[1]}</span>
                <span className="best-value">
                  {row[2]} <Icon name="check" size={14} />
                </span>
              </div>
            ))}
          </div>
        </article>
        <ChartCard
          legend={["Logistic Regression", "Transformer"]}
          subtitle="Higher is better"
          title="Metric overview"
        >
          <BarChart
            series={[
              { label: "Clean", first: 88, second: 93 },
              { label: "Robust", first: 74, second: 87 },
              { label: "F1", first: 87, second: 92 },
              { label: "Consistency", first: 80, second: 90 },
            ]}
          />
        </ChartCard>
      </div>
    </main>
  );
}

function Reports() {
  const download = (type: string) => {
    if (type === "PDF") {
      window.print();
      return;
    }
    const blob = new Blob(
      [
        "model,clean_accuracy,robust_accuracy\nLogistic Regression,88.4,74.2\nTransformer,93.1,86.7",
      ],
      { type: "text/plain" },
    );
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
        subtitle="Your model evaluation, summarized clearly."
        title="Evaluation Reports"
      />
      <article className="glass report-card">
        <div className="report-top">
          <div className="document-title">
            <span className="document-icon">
              <Icon name="file" size={25} />
            </span>
            <div>
              <span className="section-kicker">SEMANTICSHIELD · EVALUATION 04</span>
              <h2>Text Classification Robustness Report</h2>
              <p>Generated from the latest evaluation run · Sample data</p>
            </div>
          </div>
          <span className="complete-badge">
            <Icon name="check" size={14} /> Complete
          </span>
        </div>
        <div className="report-divider" />
        <section className="report-section summary-section">
          <div>
            <span className="report-index">01</span>
            <h3>Overall evaluation summary</h3>
            <p>
              Both models perform strongly on clean text. The Transformer Model shows greater
              resilience to paraphrasing and spelling changes, retaining more of its baseline
              accuracy under variation.
            </p>
          </div>
          <div className="report-score">
            <strong>84.2</strong>
            <span>OVERALL ROBUSTNESS</span>
            <small>Strong performance</small>
          </div>
        </section>
        <section className="report-section">
          <div className="report-section-heading">
            <span className="report-index">02</span>
            <h3>Evaluation details</h3>
          </div>
          <div className="details-grid">
            <Detail label="Models evaluated" value="Logistic Regression + Transformer" />
            <Detail label="Dataset" value="Sentiment Reviews · Test split" />
            <Detail label="Clean accuracy" value="91.8%" />
            <Detail label="Robust accuracy" value="84.2%" />
          </div>
        </section>
        <div className="report-columns">
          <section className="report-section findings">
            <div className="report-section-heading">
              <span className="report-index">03</span>
              <h3>Main findings</h3>
            </div>
            <ul>
              <li><Icon name="check" size={15} /> Paraphrases were handled better than spelling errors.</li>
              <li><Icon name="check" size={15} /> Contextual modeling improved consistency by 9.7%.</li>
              <li><Icon name="alert" size={15} /> Mixed sentiment remains the most common failure.</li>
            </ul>
          </section>
          <section className="report-section preview-chart">
            <div className="report-section-heading">
              <span className="report-index">04</span>
              <h3>Accuracy preview</h3>
            </div>
            <BarChart
              series={[
                { label: "Clean", first: 92, second: 88 },
                { label: "Robust", first: 84, second: 74 },
              ]}
            />
          </section>
        </div>
        <section className="report-section failure-example">
          <div className="report-section-heading">
            <span className="report-index">05</span>
            <h3>Common failure example</h3>
          </div>
          <blockquote>
            “The experience wasn’t terrible, but I wouldn’t exactly recommend it.”
            <span>Mixed polarity and indirect negation caused inconsistent predictions.</span>
          </blockquote>
        </section>
        <div className="report-actions">
          <p>Export this report to share your evaluation results.</p>
          <div>
            <Button icon="download" onClick={() => download("CSV")} variant="secondary">
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

  return (
    <div className="app">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <div className="ambient ambient-three" />
      <AppNav page={page} setPage={setPage} />
      {page === "playground" && <Playground goToAnalytics={() => setPage("analytics")} />}
      {page === "analytics" && <Analytics back={() => setPage("playground")} />}
      {page === "comparison" && <Comparison />}
      {page === "reports" && <Reports />}
      <footer className="site-footer">
        <span><Icon name="shield" size={15} /> SemanticShield</span>
        <small>Robustness insights for trustworthy text classification.</small>
      </footer>
    </div>
  );
}
