import argparse
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sentiment_engine import (
    analyze_dataframe,
    find_text_column,
    load_data_from_path,
)


def build_chart(stats, base, chart_out):
    colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#f39c12"}
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(stats["segment"], stats["count"], color=[colors[s] for s in stats["segment"]])
    for bar, cnt, pct in zip(bars, stats["count"], stats["percentage"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{cnt} ({pct}%)", ha="center", fontsize=11, fontweight="bold")
    ax.set_title(f"Sentiment Segmentation - {base}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of reviews")
    ax.set_ylim(0, max(stats["count"]) * 1.15 + 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(chart_out, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis for reviews from CSV/Excel files "
                    "(works for movies, products, services, any domain)."
    )
    parser.add_argument("input", help="Path to input .csv or .xlsx file")
    parser.add_argument("--column", help="Name of the column holding the review text "
                                         "(auto-detected if not given)")
    parser.add_argument("--output-dir", default="output", help="Directory for result files")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"File not found: {args.input}")
        sys.exit(1)

    print(f"Loading data from: {args.input}")
    df = load_data_from_path(args.input)
    print(f"Rows: {len(df)} | Columns: {list(df.columns)}")

    text_col = args.column or find_text_column(df)
    if text_col is None:
        print("Could not detect a text/review column. Use --column <name>.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    print(f"Using text column: {text_col}")

    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(
        os.path.dirname(os.path.abspath(args.input)), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    out, stats = analyze_dataframe(df, text_col)

    base = os.path.splitext(os.path.basename(args.input))[0]
    csv_out = os.path.join(output_dir, f"{base}_sentiment.csv")
    xlsx_out = os.path.join(output_dir, f"{base}_sentiment.xlsx")
    chart_out = os.path.join(output_dir, f"{base}_segmentation.png")
    stats_out = os.path.join(output_dir, f"{base}_summary.csv")

    out.to_csv(csv_out, index=False, encoding="utf-8-sig")
    out.to_excel(xlsx_out, index=False)
    stats.to_csv(stats_out, index=False, encoding="utf-8-sig")
    build_chart(stats, base, chart_out)

    print("\n=== Sentiment Summary ===")
    for _, row in stats.iterrows():
        print(f"{row['segment']:<10} {int(row['count']):>6}  {row['percentage']:>6.2f}%")
    print("\nSaved files:")
    for path in (csv_out, xlsx_out, stats_out, chart_out):
        print(f"  {path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
