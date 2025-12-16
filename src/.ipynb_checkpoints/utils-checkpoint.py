# src/utils.py

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Iterable, Tuple

import json
import re
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_TRAIN_DIR = DATA_DIR / "bioasq_raw"
RAW_TEST_DIR = DATA_DIR / "bioasq_test"
RUNS_DIR = DATA_DIR / "runs"
INDEX_DIR = PROJECT_ROOT / "index" / "whoosh_index"


def create_project_directories() -> None:
    for path in [DATA_DIR, RAW_TRAIN_DIR, RAW_TEST_DIR, RUNS_DIR, INDEX_DIR, PROJECT_ROOT / "utils"]:
        path.mkdir(parents=True, exist_ok=True)


# ---------- Naming / filtering helpers ----------

def safe_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def extract_pmid(url: str) -> str:
    return url.rsplit("/", 1)[-1] if "/" in url else url


def is_ad_question(question_text: str, mesh_terms: Iterable[str] | None = None) -> bool:
    if not question_text:
        return False

    lowered = question_text.lower()

    keyword_patterns = [
        r"alzheimer(’|')?s?",
        r"alzheimers",
        r"alzheimer disease",
        r"\bdementia\b",
        r"\bmci\b",
        r"amyloid",
        r"tauopathy",
        r"tau protein",
    ]
    if any(re.search(pattern, lowered) for pattern in keyword_patterns):
        return True

    if mesh_terms:
        lowered_mesh = [term.lower() for term in mesh_terms]
        ad_mesh_terms = {"alzheimer disease", "dementia", "mild cognitive impairment"}
        if any(term in ad_mesh_terms for term in lowered_mesh):
            return True

    return False


# ---------- BioASQ loading ----------

def _normalize_question(raw_question: Dict[str, Any]) -> Dict[str, Any]:
    body = raw_question.get("body") or raw_question.get("question", "")
    concepts = raw_question.get("concepts") or raw_question.get("mesh_terms") or []
    documents = safe_list(raw_question.get("documents"))
    snippets = safe_list(raw_question.get("snippets"))

    normalized = {
        "id": raw_question.get("id") or raw_question.get("question_id"),
        "type": raw_question.get("type") or "summary",
        "body": body,
        "concepts": concepts,
        "documents": documents,
        "snippets": snippets,
        "ideal_answer": safe_list(raw_question.get("ideal_answer")),
        "exact_answer": safe_list(raw_question.get("exact_answer")),
    }
    return normalized


def _load_json_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [ _normalize_question(question) for question in data.get("questions", []) ]


def load_bioasq_tree(root: Path) -> List[Dict[str, Any]]:
    json_paths = sorted(root.rglob("*.json"))
    all_questions: List[Dict[str, Any]] = []
    for json_path in json_paths:
        all_questions.extend(_load_json_file(json_path))
    return all_questions


def filter_ad_questions(all_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only AD / dementia questions with gold PMIDs extracted."""
    ad_subset: List[Dict[str, Any]] = []

    for question in all_questions:
        mesh_terms = question.get("concepts", [])
        body = question.get("body", "")
        if not is_ad_question(body, mesh_terms):
            continue

        gold_pmids = [
            extract_pmid(url)
            for url in question.get("documents", [])
            if url
        ]

        ad_subset.append(
            {
                "id": question.get("id"),
                "type": question.get("type") or "summary",
                "body": body,
                "concepts": mesh_terms,
                "gold_pmids": sorted(set(gold_pmids)),
                "snippets": question.get("snippets", []),
                "ideal_answer": question.get("ideal_answer", []),
                "exact_answer": question.get("exact_answer", []),
            }
        )

    return ad_subset


# ---------- Corpus building ----------

def build_snippet_corpus(ad_questions: List[Dict[str, Any]]) -> Dict[str, str]:
    pmid_to_snippets: Dict[str, List[str]] = defaultdict(list)

    for question in ad_questions:
        for snippet in question.get("snippets", []):
            document_url = snippet.get("document", "")
            pmid = extract_pmid(document_url)
            snippet_text = " ".join(
                [
                    snippet.get("text", ""),
                    snippet.get("beginSection") or "",
                    snippet.get("endSection") or "",
                ]
            ).strip()
            if snippet_text:
                pmid_to_snippets[pmid].append(snippet_text)

    snippet_corpus = {
        pmid: " \n".join(text_list)
        for pmid, text_list in pmid_to_snippets.items()
    }
    return snippet_corpus


# ---------- Metrics ----------

def recall_at_k(gold_pmids: Iterable[str], ranked_pmids: List[str], k: int = 10) -> float:
    gold_set = set(gold_pmids)
    if not gold_set:
        return 0.0
    retrieved = set(ranked_pmids[:k])
    return len(gold_set & retrieved) / len(gold_set)


def average_precision(gold_pmids: Iterable[str], ranked_pmids: List[str]) -> float:
    gold_set = set(gold_pmids)
    if not gold_set:
        return 0.0

    hits = 0
    precisions: List[float] = []
    for rank_index, pmid in enumerate(ranked_pmids, start=1):
        if pmid in gold_set:
            hits += 1
            precisions.append(hits / rank_index)

    return float(np.mean(precisions)) if precisions else 0.0


def ndcg_at_k(gold_pmids: Iterable[str], ranked_pmids: List[str], k: int = 10) -> float:
    gold_set = set(gold_pmids)
    gains = [
        1.0 if pmid in gold_set else 0.0
        for pmid in ranked_pmids[:k]
    ]
    dcg = sum(gain / np.log2(index + 2) for index, gain in enumerate(gains))

    ideal_size = min(len(gold_set), k)
    if ideal_size == 0:
        return 0.0
    ideal_gains = [1.0] * ideal_size
    idcg = sum(gain / np.log2(index + 2) for index, gain in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def recall_cap_at_k(num_gold_docs: int, k: int) -> float:
    if num_gold_docs == 0:
        return 0.0
    return min(1.0, k / num_gold_docs)


def compute_cap_normalized_r10(
    questions: List[Dict[str, Any]],
    retrieve_top_pmids,
    topk: int = 20,
) -> Tuple[float, float]:
    raw_scores: List[float] = []
    normalized_scores: List[float] = []

    for question in questions:
        ranked = retrieve_top_pmids(question["body"], topk)
        gold_pmids = question["gold_pmids"]
        num_gold = len(gold_pmids)
        if num_gold == 0:
            continue

        cap = recall_cap_at_k(num_gold, 10)
        hits_at_10 = len(set(ranked[:10]) & set(gold_pmids)) / num_gold
        raw_scores.append(hits_at_10)
        normalized_scores.append(hits_at_10 / cap if cap > 0 else 0.0)

    mean = lambda values: sum(values) / len(values) if values else 0.0
    return mean(raw_scores), mean(normalized_scores)
