import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sentiment_compare import finbert_score, sent_label


INPUT = ROOT / "results" / "data" / "sentiment_test.csv"
OUTPUT = ROOT / "results" / "data" / "sentiment_threshold_result.csv"

THRESHOLDS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]


def model_label(score, threshold):
    label = sent_label(score, threshold)

    return {
        "pos": "positive",
        "neg": "negative",
        None: "neutral",
    }[label]


def load_groups():
    with INPUT.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    groups = {}

    for row in rows:
        event_id = row["event_id"]

        groups.setdefault(event_id, {
            "human_label": row["human_label"],
            "headlines": [],
        })

        groups[event_id]["headlines"].append(
            row["headline"]
        )

    return groups


def main():
    groups = load_groups()
    scored = []

    for event_id, group in groups.items():
        headlines = group["headlines"][:10]

        article_scores = [
            finbert_score(headline)
            for headline in headlines
        ]

        average_score = (
            sum(article_scores) / len(article_scores)
            if article_scores else 0
        )

        scored.append({
            "event_id": event_id,
            "human_label": group["human_label"],
            "headlines": headlines,
            "article_scores": article_scores,
            "average_score": average_score,
        })

    results = []

    for threshold in THRESHOLDS:
        correct = 0

        for event in scored:
            predicted = model_label(
                event["average_score"],
                0.15,
            )

            status = (
                "O"
                if predicted == event["human_label"]
                else "X"
            )

            print(
                f"{status} | "
                f"score={event['average_score']:+.3f} | "
                f"human={event['human_label']:8} | "
                f"model={predicted:8} | "
                f"{event['headlines'][0]}"
            )
            
            if predicted == event["human_label"]:
                correct += 1

        accuracy = (
            correct / len(scored) * 100
            if scored else 0
        )

        results.append({
            "threshold": threshold,
            "correct": correct,
            "total": len(scored),
            "accuracy": round(accuracy, 1),
        })

        print(
            f"threshold={threshold:.2f} | "
            f"{correct}/{len(scored)} | "
            f"accuracy={accuracy:.1f}%"
        )

    with OUTPUT.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "threshold",
                "correct",
                "total",
                "accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n저장: {OUTPUT}")


if __name__ == "__main__":
    main()