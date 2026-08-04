import io
import os
import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

FALLBACK_LEXICON = {
    "excellent": 3, "amazing": 3, "awesome": 3, "brilliant": 3, "fantastic": 3,
    "outstanding": 3, "superb": 3, "wonderful": 3, "perfect": 3, "love": 3,
    "loved": 3, "best": 3, "masterpiece": 3, "stunning": 2, "fabulous": 3,
    "incredible": 3, "great": 2, "good": 1, "nice": 1, "happy": 2, "enjoyed": 2,
    "enjoy": 1, "recommend": 1, "like": 1, "liked": 1, "must-watch": 2,
    "satisfied": 2, "impressive": 2, "impressed": 2, "beautiful": 2, "delicious": 2,
    "comfortable": 1, "fast": 1, "easy": 1, "helpful": 1, "quality": 1, "fresh": 1,
    "terrible": -3, "awful": -3, "horrible": -3, "worst": -3, "disgusting": -3,
    "waste": -3, "disappointed": -2, "disappointing": -2, "boring": -2, "bad": -2,
    "poor": -2, "hated": -3, "hate": -3, "slow": -1, "rude": -2,
    "useless": -3, "broken": -2, "overrated": -2, "frustrating": -2, "failed": -2,
    "fail": -1, "average": 0, "okay": 0, "ok": 0, "meh": 0, "mediocre": 0,
    "decent": 0, "not good": -1, "not bad": 1, "not worth": -2, "no": -1,
    "never": -1, "refund": -2, "delay": -1, "late": -1, "damaged": -2,
}

TEXT_COLUMN_KEYWORDS = [
    "review", "text", "comment", "feedback", "message", "tweet", "post",
    "description", "statement", "body", "content", "opinion", "remarks",
]

SEGMENT_COLORS = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#f39c12"}


def detect_encoding(path):
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as fh:
                fh.read(2000)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def decode_bytes(raw, enc):
    for candidate in (enc, "utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            return raw.decode(candidate)
        except (UnicodeDecodeError, AttributeError):
            continue
    return raw.decode("latin-1")


def load_data_from_path(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        enc = detect_encoding(path)
        return pd.read_csv(path, encoding=enc)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {ext}. Use a .csv or .xlsx file.")


def load_data_from_upload(uploaded):
    raw = uploaded.getvalue()
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        text = decode_bytes(raw, None)
        return pd.read_csv(io.StringIO(text))
    if name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(raw))
    if name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))
    raise ValueError("Unsupported file type. Upload a .csv or .xlsx file.")


def find_text_column(df):
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for kw in TEXT_COLUMN_KEYWORDS:
        for key, col in lowered.items():
            if kw in key:
                return col
    best_col = None
    best_len = 0
    for col in df.columns:
        sample = df[col].dropna().astype(str)
        if sample.empty:
            continue
        joined = " ".join(sample.head(50))
        if len(joined) > best_len:
            best_len = len(joined)
            best_col = col
    return best_col


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\S+", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lexicon_score(text):
    words = re.findall(r"[a-z']+", text.lower())
    score = 0
    joined = " ".join(words)
    for phrase in ("not good", "not bad", "not worth", "must-watch"):
        if phrase in joined:
            score += FALLBACK_LEXICON[phrase]
    for word in words:
        if word in FALLBACK_LEXICON:
            score += FALLBACK_LEXICON[word]
    return max(-1.0, min(1.0, score / 10.0))


def classify(score):
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def _score_one(text, analyzer):
    cleaned = clean_text(text)
    vs = analyzer.polarity_scores(cleaned)
    compound = vs["compound"]
    if abs(compound) < 0.01 and lexicon_score(cleaned) != 0:
        compound = lexicon_score(cleaned)
    return cleaned, round(compound, 4)


def analyze_dataframe(df, text_col, progress_cb=None):
    analyzer = SentimentIntensityAnalyzer()
    out = df.copy()
    cleaned = []
    scores = []
    sentiments = []
    total = len(df)
    for i, text in enumerate(df[text_col]):
        c, s = _score_one(text, analyzer)
        cleaned.append(c)
        scores.append(s)
        sentiments.append(classify(s))
        if progress_cb is not None and i % max(1, total // 20) == 0:
            progress_cb(i / total)
    out["cleaned_text"] = cleaned
    out["sentiment_score"] = scores
    out["sentiment"] = sentiments

    counts = {
        seg: sum(1 for s in sentiments if s == seg)
        for seg in ("positive", "negative", "neutral")
    }
    total = max(len(sentiments), 1)
    stats = pd.DataFrame(
        {
            "segment": list(counts.keys()),
            "count": list(counts.values()),
        }
    )
    stats["percentage"] = (stats["count"] / total * 100).round(2)
    return out, stats


def segment_frames(out_df):
    return {
        seg: out_df[out_df["sentiment"] == seg]
        for seg in ("positive", "negative", "neutral")
    }
