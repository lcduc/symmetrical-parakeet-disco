"""Evaluation harness for the Step 15 labeled set.

Runs the live /matches/run pipeline for every job_id in the labeled set
and compares its output against the human-labeled rankings and hard-filter
decisions. Requires the API server to be running (see main.py).

Usage: python eval/run_eval.py [--api-base-url http://localhost:8002]
"""

import argparse
import json
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DEFAULT_LABELED_SET_PATH = Path(__file__).parent / "labeled_set.jsonl"
DEFAULT_API_BASE_URL = "http://localhost:8002"


def load_labeled_set(path: Path | str) -> list[dict]:
    """Read the Step 15 labeled evaluation set from a JSONL file."""
    entries = []
    with open(path, encoding="utf-8") as labeled_file:
        for line in labeled_file:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    logger.info("Loaded %d labeled entries from %s", len(entries), path)
    return entries


class MatchClient:
    """Thin HTTP client for the running API's /matches/run endpoint."""

    def __init__(self, base_url: str = DEFAULT_API_BASE_URL):
        self._base_url = base_url.rstrip("/")

    def run_matches(self, job_id: str) -> list[dict]:
        """Trigger matching for `job_id` and return the ranked MatchResponse list."""
        logger.info("Running matches for job_id=%s", job_id)
        response = requests.post(f"{self._base_url}/matches/run", params={"job_id": job_id}, timeout=180)
        response.raise_for_status()
        return response.json()


def _spearman_correlation(human_ranking: list[str], system_ranking: list[str]) -> float | None:
    """Spearman rank correlation between two candidate_id orderings.

    Restricted to candidates present in both lists. Each list is first
    filtered down to that shared set (preserving its own relative order),
    then re-ranked 1..n within the subset — the closed-form formula below
    is only valid when ranks are a dense 1..n permutation, so reusing raw
    positions from the original (unfiltered, longer) lists would be wrong.
    Returns None if fewer than 2 candidates are shared, since correlation
    is undefined for n < 2.
    """
    system_position = {candidate_id: position for position, candidate_id in enumerate(system_ranking)}
    common_ids_by_human_order = [candidate_id for candidate_id in human_ranking if candidate_id in system_position]

    n = len(common_ids_by_human_order)
    if n < 2:
        return None

    human_rank = {candidate_id: rank for rank, candidate_id in enumerate(common_ids_by_human_order)}
    common_ids_by_system_order = sorted(common_ids_by_human_order, key=lambda cid: system_position[cid])
    system_rank = {candidate_id: rank for rank, candidate_id in enumerate(common_ids_by_system_order)}

    sum_squared_rank_diff = sum(
        (human_rank[candidate_id] - system_rank[candidate_id]) ** 2 for candidate_id in common_ids_by_human_order
    )
    return 1 - (6 * sum_squared_rank_diff) / (n * (n**2 - 1))


def evaluate(labeled_set: list[dict], system_results: dict[str, list[dict]]) -> dict:
    """Score the live pipeline's output against the labeled set.

    system_results maps job_id -> the ranked list returned by /matches/run
    (each item a dict with at least candidate_id and passed_hard_filters).

    Returns a dict with mean_ranking_spearman (averaged only over jobs with
    at least 2 candidates shared between human and system rankings),
    hard_filter_precision, and hard_filter_recall (pooled across all
    labeled candidate/job pairs where the candidate appears in the system
    results).
    """
    spearman_scores: list[float] = []
    true_positives = false_positives = false_negatives = 0

    for entry in labeled_set:
        job_id = entry["job_id"]
        human_ranking = entry["human_ranking"]
        job_results = system_results.get(job_id, [])
        system_ranking = [row["candidate_id"] for row in job_results]

        correlation = _spearman_correlation(human_ranking, system_ranking)
        if correlation is None:
            logger.warning(
                "Job %s: fewer than 2 candidates shared between human and system rankings; "
                "excluding from mean_ranking_spearman",
                job_id,
            )
        else:
            spearman_scores.append(correlation)

        system_passed_by_id = {row["candidate_id"]: row["passed_hard_filters"] for row in job_results}
        for label in entry["hard_filter_labels"]:
            candidate_id = label["candidate_id"]
            if candidate_id not in system_passed_by_id:
                logger.warning(
                    "Job %s: candidate %s absent from system results; excluding from hard-filter precision/recall",
                    job_id,
                    candidate_id,
                )
                continue

            human_passed = label["human_passed"]
            system_passed = system_passed_by_id[candidate_id]
            if system_passed and human_passed:
                true_positives += 1
            elif system_passed and not human_passed:
                false_positives += 1
            elif not system_passed and human_passed:
                false_negatives += 1

    mean_ranking_spearman = sum(spearman_scores) / len(spearman_scores) if spearman_scores else float("nan")
    hard_filter_precision = (
        true_positives / (true_positives + false_positives) if (true_positives + false_positives) else float("nan")
    )
    hard_filter_recall = (
        true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else float("nan")
    )

    return {
        "mean_ranking_spearman": mean_ranking_spearman,
        "hard_filter_precision": hard_filter_precision,
        "hard_filter_recall": hard_filter_recall,
    }


def main() -> None:
    """Load the labeled set, run the live pipeline for each job, and print the eval report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-set", default=str(DEFAULT_LABELED_SET_PATH))
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    args = parser.parse_args()

    labeled_set = load_labeled_set(args.labeled_set)
    client = MatchClient(args.api_base_url)

    system_results: dict[str, list[dict]] = {}
    for entry in labeled_set:
        job_id = entry["job_id"]
        if job_id not in system_results:
            system_results[job_id] = client.run_matches(job_id)

    report = evaluate(labeled_set, system_results)

    print("=== Evaluation Report ===")
    print(f"mean_ranking_spearman: {report['mean_ranking_spearman']:.4f}")
    print(f"hard_filter_precision: {report['hard_filter_precision']:.4f}")
    print(f"hard_filter_recall: {report['hard_filter_recall']:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    main()
