# 🛡️ TrustLens

**Social Media Enforcement Intelligence — unit-correct analytics, anomaly detection, and forecasting for platform Trust & Safety transparency data.**

![Python](https://img.shields.io/badge/python-3.12-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.63-ff4b4b)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

TrustLens turns the enforcement-transparency reports that platforms like Meta, X,
WhatsApp, and others publish (accounts banned, content removed, proactive
detection rates, etc.) into a single interactive dashboard — with anomaly
detection, per-platform forecasting, and an offline Q&A assistant, no API
keys or internet connection required.

---

## Features

### 📊 Dashboard (`app.py`)
- KPIs: latest volume, average, trend direction, growth % (safe against
  divide-by-zero on a zero baseline)
- Unit-normalized trend chart with optional 3-month smoothing
- Organization ranking (bar chart + table)
- ML anomaly detection (Isolation Forest) with adjustable sensitivity
- Quick regression-based forecast
- 🤖 **Offline AI Analytics Assistant** — a chat-style Q&A panel with quick-question
  pills. It's a deterministic, keyword-based intent matcher (not an LLM) that
  answers from the currently filtered data — trend, rankings, volatility,
  topics, proactive rate, and anomalies. Nothing is sent anywhere.
- Export a PDF summary report or the filtered dataset as CSV
- **Bring your own data**: upload any CSV with the four required columns and
  the whole app — including every sub-page — re-analyzes your data instead of
  the bundled sample

### 🔮 Forecast
Per-organization projections with 95% confidence bands and a reliability
score (guarded against divide-by-zero for flat/near-zero forecasts).

### 🧭 Category Insights
Breaks enforcement volume down by violation category: top categories overall,
a category × platform heatmap, and a trend chart for any single category.

### 🛡️ Proactive Detection
Surfaces the platforms' proactive-detection **rate** (% of violations caught
before being reported) — a genuinely different metric from volume, kept
separate throughout the app so it's never accidentally summed with counts.

### 📋 Data Quality
Full transparency on the cleaning pipeline: rows read, rows dropped, unit
typos fixed, rate-rows excluded from volume totals, missing-value summary,
and a downloadable cleaned dataset.

---

## Project structure

```
trustlens/
├── app.py                          # Main dashboard
├── config.py                       # Central configuration (env vars, ML params, unit rules)
├── data_utils.py                   # The data-cleaning pipeline (unit normalization lives here)
├── theme.py                        # Shared CSS/header used by every page
├── pages/
│   ├── 1_🔮_Forecast.py
│   ├── 2_🧭_Category_Insights.py
│   ├── 3_🛡️_Proactive_Detection.py
│   └── 4_📋_Data_Quality.py
├── preprocessed_enforcement_data.csv   # Bundled sample dataset
├── requirements.txt
├── .env.example
├── .streamlit/config.toml
├── Procfile / render.yaml / setup.sh   # Deployment
└── README.md
```

---

## Getting started

### Requirements
- Python 3.12 (see `.python-version`)

### Install & run locally

```bash
git clone https://github.com/<your-username>/trustlens.git
cd trustlens
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. No API keys or external services
are needed — everything, including the AI assistant, runs locally.

---

## Using your own data

Upload any CSV from the sidebar with at least these columns:

| Column | Required | Notes |
|---|---|---|
| `date` | ✅ | Any format `pandas.to_datetime` can parse |
| `organization` | ✅ | Platform/entity name |
| `action_as_per_source` | ✅ | e.g. `Content Removed`, `Total Accounts Banned`, `Proactive Rate` |
| `standard_value` | ✅ | The numeric value |
| `units` | optional | `value in absolute number` / `thousands` / `millions` / `percentage` — enables unit normalization |
| `topic` | optional | Violation category — unlocks Category Insights |
| `proactive_flag` | optional | Unlocks additional proactive-detection detail |

---

## Methodology notes

- **Unit normalization**: `standard_value` is multiplied by 1, 1,000, or
  1,000,000 depending on the cleaned `units` string. Percentage/rate rows are
  set to `NaN` in `normalized_value` on purpose so they can never enter a
  volume `SUM` — see `data_utils.compute_normalized_value`.
- **Anomaly detection**: `sklearn.ensemble.IsolationForest` on the
  unit-normalized overall time series.
- **Forecasting**: ordinary least-squares linear regression per organization,
  clipped at zero (a count can't be negative), with a 95% confidence band
  from the residual standard deviation.
- **Reliability score**: `100 − coefficient of variation` of the forecast,
  floored at 0. It measures how noisy the *historical* data was — treat
  forecasts as directional, not guaranteed.

---

## Roadmap

- [ ] Seasonal decomposition for platforms with 2+ years of history
- [ ] Multi-file upload to compare two reporting periods side by side
- [ ] CSV schema auto-mapping wizard for datasets with differently-named columns
- [ ] Optional scheduled email/Slack digest of new anomalies

---

## Contributing

Issues and pull requests are welcome. Please run the app locally and confirm
`streamlit run app.py` starts cleanly before submitting a PR.

## License

MIT — see [LICENSE](LICENSE).

## Data source

The bundled sample dataset is content-moderation transparency data
self-reported by platforms in their public transparency reports. It contains
only aggregate statistics — no individual user data.
