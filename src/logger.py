import csv
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "benchmark.csv"


def log_query(result):
    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "mode",
                "question",
                "retrieval",
                "evidence",
                "generation",
                "total",
                "raw_sources",
                "reranked_sources",
                "confidence",
            ])

        timings = result.get("timings", {})

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            result["mode"],
            result["question"],
            timings.get("retrieval", 0),
            timings.get("evidence_extraction", 0),
            timings.get("answer_generation", 0),
            result["total_time"],
            result["raw_sources"],
            len(result["results"]),
            result["confidence"],
        ])