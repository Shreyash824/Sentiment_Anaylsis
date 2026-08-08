# Sentiment Analysis

A domain-agnostic sentiment analyzer that segments user reviews (positive / negative / neutral)
from **CSV or Excel** files. Works for movie reviews, product reviews, service feedback,
tweets and any other text data.

## How it works

1. **Load** – reads `.csv` or `.xlsx` (auto-detects encoding for CSV).
2. **Column detection** – automatically finds the review/text column
   (looks for names like `review`, `text`, `comment`, `feedback`, ...); fallback to `--column`.
3. **Score** – uses VADER (`compound` score) with a built-in domain-agnostic lexicon fallback.
4. **Segment** – `>= 0.05` → positive, `<= -0.05` → negative, otherwise neutral.
5. **Save** – writes the full data with new `sentiment_score` and `sentiment` columns to:
   - `<name>_sentiment.csv`
   - `<name>_sentiment.xlsx`
   - `<name>_summary.csv` (segment counts + percentages)
   - `<name>_segmentation.png` (bar chart)

## Setup

```bash
pip install -r requirements.txt
```

## Interactive UI (recommended)

```bash
streamlit run app.py
```

Opens in the browser. Features:

- Drag & drop a CSV / Excel file (upload only, no bundled samples)
- Auto-detects the review column (override via dropdown)
- KPI cards + interactive Plotly bar / pie charts
- Searchable, segment-filterable review table
- Per-segment browsing with expandable lists
- One-click downloads of results (CSV + Excel) and the summary

## CLI usage

```bash
python sentiment_analysis.py "data/sample_movie_reviews.csv"
python sentiment_analysis.py "data/sample_product_reviews.xlsx"
python sentiment_analysis.py "data/your_file.xlsx" --column review_text --output-dir output
```

Results are written next to the input file in the `output` folder by default.

## Sample data

- `data/sample_movie_reviews.csv` – movie reviews
- `data/sample_product_reviews.xlsx` – product / service reviews

## Files

- `sentiment_engine.py` – shared core (loading, scoring, classification)
- `app.py` – Streamlit interactive UI
- `sentiment_analysis.py` – command-line version

## Example output

```
=== Sentiment Summary ===
positive      11  55.00%
negative       6  30.00%
neutral        3  15.00%
```
