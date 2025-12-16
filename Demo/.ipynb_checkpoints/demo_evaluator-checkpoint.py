from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any

from time import sleep
import json
import re
import time as _time

import torch
from Bio import Entrez
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoTokenizer, AutoModelForCausalLM

from demo_utils import (
    DEMO_RUNS_DIR,
    build_snippet_corpus,
    recall_at_k,
    average_precision,
    ndcg_at_k,
    compute_cap_normalized_r10,
)
from demo_retriever import RetrievalPipeline, CorpusData


# ---------- PubMed enrichment for demo ----------

def fetch_pubmed_records(pmids: Iterable[str], email: str, tool_name: str = "alz-evidence-qa-demo") -> Dict[str, Dict[str, str]]:
    Entrez.email = email
    Entrez.tool = tool_name

    pmid_list = list(pmids)
    records: Dict[str, Dict[str, str]] = {}

    if not pmid_list:
        return records

    for start_index in range(0, len(pmid_list), 100):
        chunk = pmid_list[start_index: start_index + 100]
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(chunk),
            rettype="abstract",
            retmode="xml",
        )
        parsed = Entrez.read(handle)

        for article in parsed.get("PubmedArticle", []):
            pmid = str(article["MedlineCitation"]["PMID"])
            article_data = article["MedlineCitation"]["Article"]
            title = str(article_data.get("ArticleTitle", "") or "")
            abstract_segments = article_data.get("Abstract", {}).get("AbstractText", [])
            abstract_text = " ".join(map(str, abstract_segments)).strip()
            records[pmid] = {
                "title": title.strip(),
                "abstract": abstract_text,
            }

        # be gentle with NCBI
        sleep(0.35)

    return records


def build_full_corpus(
    train_questions: List[Dict[str, Any]],
    extra_questions: List[Dict[str, Any]] | None,
    email: str,
) -> CorpusData:
    if extra_questions is None:
        all_questions = list(train_questions)
    else:
        all_questions = list(train_questions) + list(extra_questions)

    snippet_corpus = build_snippet_corpus(all_questions)

    all_gold_pmids = set()
    for question in all_questions:
        all_gold_pmids.update(question["gold_pmids"])

    missing_pmids = sorted(all_gold_pmids - set(snippet_corpus.keys()))
    pubmed_records = fetch_pubmed_records(missing_pmids, email=email)

    texts: Dict[str, str] = {}
    titles: Dict[str, str] = {}

    for pmid in all_gold_pmids:
        snippet_text = snippet_corpus.get(pmid, "")
        record = pubmed_records.get(pmid, {"title": "", "abstract": ""})
        title = record["title"]
        abstract = record["abstract"]

        combined_parts = [snippet_text, title, abstract]
        combined_text = " \n".join([part for part in combined_parts if part]).strip()
        if not combined_text:
            continue

        texts[pmid] = combined_text
        titles[pmid] = title or "(No title fetched)"

    return CorpusData(texts=texts, titles=titles)


# ---------- Retrieval evaluation ----------

@dataclass
class RetrievalMetrics:
    recall_at_k: float
    ndcg_at_10: float
    map_score: float
    cap_norm_r10: float
    raw_r10: float


def evaluate_retrieval_variant(
    questions: List[Dict[str, Any]],
    retrieve_fn,
    variant_name: str,
    topk: int = 20,
) -> RetrievalMetrics:
    recalls: List[float] = []
    ndcgs: List[float] = []
    maps: List[float] = []

    for question in questions:
        ranked = retrieve_fn(question["body"], topk=topk)
        gold = question["gold_pmids"]

        recalls.append(recall_at_k(gold, ranked, k=topk))
        ndcgs.append(ndcg_at_k(gold, ranked, k=10))
        maps.append(average_precision(gold, ranked))

    mean = lambda values: sum(values) / len(values) if values else 0.0
    raw_r10, cap_norm_r10 = compute_cap_normalized_r10(
        questions,
        retrieve_top_pmids=lambda query, topk_val=20: retrieve_fn(query, topk=topk_val),
        topk=topk,
    )

    metrics = RetrievalMetrics(
        recall_at_k=mean(recalls),
        ndcg_at_10=mean(ndcgs),
        map_score=mean(maps),
        cap_norm_r10=cap_norm_r10,
        raw_r10=raw_r10,
    )

    print(f"== {variant_name} ==")
    print(
        f"Recall@{topk}: {metrics.recall_at_k:.3f}  "
        f"nDCG@10: {metrics.ndcg_at_10:.3f}  "
        f"MAP: {metrics.map_score:.3f}"
    )
    print(
        f"Raw R@10: {metrics.raw_r10:.3f}  "
        f"Cap-normalized R@10: {metrics.cap_norm_r10:.3f}"
    )
    print()
    return metrics


# ---------- QA model + answer generation (demo) ----------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
AD_TERMS = ("alzheimer", "dementia", "mci", "cognitive impairment", "amyloid", "tau")


def _is_reasonable_sentence(text: str, min_len: int = 60, max_len: int = 300) -> bool:
    stripped = text.strip()
    if not (min_len <= len(stripped) <= max_len):
        return False
    if "sections." in stripped.lower():
        return False
    letters = sum(character.isalpha() for character in stripped)
    if letters < max(10, int(0.2 * len(stripped))):
        return False
    return True


def _mentions_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _query_terms(query_text: str, max_terms: int = 6) -> List[str]:
    tokens = re.findall(r"[A-Za-z]{4,}", query_text.lower())
    seen = set()
    important_terms: List[str] = []

    for token in tokens:
        if token not in seen:
            seen.add(token)
            important_terms.append(token)
    return important_terms[:max_terms]


def select_top_sentences_for_pmids(
    query_text: str,
    pmids: List[str],
    pipeline: RetrievalPipeline,
    max_sentences: int = 8,
) -> List[Tuple[str, str]]:
    query_terms = _query_terms(query_text)
    sentences: List[str] = []
    sentence_pmids: List[str] = []

    for pmid in pmids:
        document_text = pipeline.get_document_text(pmid) or ""
        for sentence in _SENT_SPLIT.split(document_text):
            if _is_reasonable_sentence(sentence):
                sentences.append(sentence.strip())
                sentence_pmids.append(pmid)

    if not sentences:
        return []

    vectorizer = TfidfVectorizer(
        max_features=30_000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    sentence_matrix = vectorizer.fit_transform(sentences)
    query_vector = vectorizer.transform([query_text])
    similarity_scores = linear_kernel(query_vector, sentence_matrix).ravel()
    ranked_indices = similarity_scores.argsort()[::-1]

    chosen: List[Tuple[str, str]] = []

    # Phase 1: require co-mention of query term and AD term
    for index in ranked_indices:
        if len(chosen) >= max_sentences:
            break
        sentence_text = sentences[index]
        if _mentions_any(sentence_text, query_terms) and _mentions_any(sentence_text, AD_TERMS):
            chosen.append((sentence_text, sentence_pmids[index]))

    if not chosen:
        return []

    # Phase 2: fill remaining slots by similarity
    for index in ranked_indices:
        if len(chosen) >= max_sentences:
            break
        candidate = (sentences[index], sentence_pmids[index])
        if candidate not in chosen:
            chosen.append(candidate)

    return chosen[:max_sentences]


def build_qa_prompt(
    question_text: str,
    evidence_lines: List[Tuple[str, str]],
    question_type: str = "summary",
) -> Tuple[str, str]:
    system_message = "You are a careful biomedical QA assistant. Use only the provided evidence."

    evidence_block = "\n".join(
        f"[{pmid}] {sentence}"
        for sentence, pmid in evidence_lines
    )

    type_hint_map = {
        "yesno": "If yes/no, start with 'Yes.' or 'No.' in sentence 1, then justify briefly.",
        "factoid": "If a fact is asked, state the fact first, then justify briefly.",
        "list": "If a list is asked, list key items concisely in one sentence, then justify.",
    }
    type_hint = type_hint_map.get(question_type.lower(), "Write a concise biomedical answer.")

    user_message = f"""Question: {question_text}

Evidence lines (each starts with PMID):
{evidence_block}

Rules:
1) Write EXACTLY 3 sentences. No bullets, no numbering.
2) Each sentence MUST end with a bracketed PMID (e.g., [12345678]).
3) Use only PMIDs that appear in the evidence lines above. Do NOT invent PMIDs.
4) Avoid repeating the same phrase or PMID unless necessary.
5) Do NOT mention instructions or rules in your answer.
6) If evidence is insufficient, output exactly: Insufficient evidence from retrieved abstracts.

Additional guidance: {type_hint}

Now write the 3 sentences."""
    return system_message, user_message


def _force_three_sentences_with_pmids(
    generated_text: str,
    allowed_pmids: List[str],
    pipeline: RetrievalPipeline,
) -> Tuple[str, List[str]]:
    sentences = [sentence.strip() for sentence in _SENT_SPLIT.split(generated_text) if sentence.strip()]
    sentences = sentences[:3] if sentences else []

    if not sentences:
        return "Insufficient evidence from retrieved abstracts.", []

    evidence_sentences: List[str] = []
    evidence_pmids: List[str] = []

    for pmid in allowed_pmids:
        document_text = pipeline.get_document_text(pmid) or ""
        for sentence in _SENT_SPLIT.split(document_text):
            if _is_reasonable_sentence(sentence):
                evidence_sentences.append(sentence.strip())
                evidence_pmids.append(pmid)

    if not evidence_sentences:
        return "Insufficient evidence from retrieved abstracts.", []

    vectorizer = TfidfVectorizer(max_features=20_000, stop_words="english")
    evidence_matrix = vectorizer.fit_transform(evidence_sentences)
    prediction_matrix = vectorizer.transform(sentences)
    similarity_scores = linear_kernel(prediction_matrix, evidence_matrix)

    fixed_sentences: List[str] = []
    used_pmids: List[str] = []

    for row_index, sentence in enumerate(sentences):
        row_scores = similarity_scores[row_index]
        sorted_indices = row_scores.argsort()[::-1]
        chosen_pmid = None

        for index in sorted_indices:
            candidate_pmid = evidence_pmids[index]
            if candidate_pmid in allowed_pmids and candidate_pmid not in used_pmids:
                chosen_pmid = candidate_pmid
                break

        if chosen_pmid is None:
            chosen_pmid = allowed_pmids[0]

        cleaned_sentence = re.sub(r"\s+", " ", sentence).strip()
        cleaned_sentence = re.sub(
            r"\b(evidence lines?|rules?|sentences?)\b",
            "",
            cleaned_sentence,
            flags=re.IGNORECASE,
        )
        cleaned_sentence = re.sub(r"\s{2,}", " ", cleaned_sentence).strip()
        if len(cleaned_sentence) > 320:
            cleaned_sentence = cleaned_sentence[:320].rstrip(",;: ")

        if not cleaned_sentence.endswith(f"[{chosen_pmid}]"):
            cleaned_sentence = cleaned_sentence.rstrip(". ").strip() + f" [{chosen_pmid}]"

        fixed_sentences.append(cleaned_sentence)
        used_pmids.append(chosen_pmid)

    final_answer = " ".join(fixed_sentences)
    return final_answer, used_pmids[:5]


@dataclass
class QaModelBundle:
    tokenizer: Any
    model: Any
    device: str


def load_qa_model(model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> QaModelBundle:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    ).to(device)

    return QaModelBundle(tokenizer=tokenizer, model=model, device=device)


def answer_with_pmids(
    question_text: str,
    question_type: str,
    candidate_pmids: List[str],
    pipeline: RetrievalPipeline,
    qa_model: QaModelBundle,
    max_new_tokens: int = 220,
) -> Tuple[str, List[str]]:
    evidence = select_top_sentences_for_pmids(
        query_text=question_text,
        pmids=candidate_pmids,
        pipeline=pipeline,
        max_sentences=8,
    )
    if not evidence:
        return "Insufficient evidence from retrieved abstracts.", []

    system_message, user_message = build_qa_prompt(
        question_text,
        evidence,
        question_type=question_type,
    )

    full_prompt = f"<|system|>\n{system_message}\n<|user|>\n{user_message}\n<|assistant|>\n"

    inputs = qa_model.tokenizer(full_prompt, return_tensors="pt").to(qa_model.device)
    outputs = qa_model.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.05,
        eos_token_id=qa_model.tokenizer.eos_token_id,
    )
    decoded = qa_model.tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    ).split("<|assistant|>")[-1].strip()

    allowed_pmids = list(dict.fromkeys(candidate_pmids))
    final_answer, cited_pmids = _force_three_sentences_with_pmids(
        decoded,
        allowed_pmids=allowed_pmids,
        pipeline=pipeline,
    )
    if not cited_pmids:
        return "Insufficient evidence from retrieved abstracts.", []

    return final_answer, cited_pmids


def evaluate_qa(
    questions: List[Dict[str, Any]],
    pipeline: RetrievalPipeline,
    qa_model: QaModelBundle,
    dense_name: str,
    topk: int = 20,
    save_prefix: str = "demo_answers20_llm",
) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores: List[float] = []
    exact_scores: List[float] = []
    answers: List[Dict[str, Any]] = []

    for question in questions:
        question_type = (question.get("type") or "summary").lower()

        ranked_pmids = pipeline.retrieve_with_dense_rerank(
            question["body"],
            dense_name=dense_name,
            topk=topk,
            pool_k=300,
        )

        answer_text, cited_pmids = answer_with_pmids(
            question_text=question["body"],
            question_type=question_type,
            candidate_pmids=ranked_pmids,
            pipeline=pipeline,
            qa_model=qa_model,
        )

        ideal_list = question.get("ideal_answer", []) or []
        ideal_text = " ".join(ideal_list)
        if ideal_text:
            rouge = scorer.score(ideal_text, answer_text)["rougeL"].fmeasure
            rouge_scores.append(rouge)

        flat_exact: List[str] = []
        for candidate in question.get("exact_answer", []):
            if isinstance(candidate, list):
                flat_exact.extend(candidate)
            else:
                flat_exact.append(candidate)
        flat_exact = [item.strip().lower() for item in flat_exact if item]
        exact_hit = 1.0 if flat_exact and any(term in answer_text.lower() for term in flat_exact) else 0.0
        exact_scores.append(exact_hit)

        answers.append(
            {
                "id": question["id"],
                "question": question["body"],
                "answer": answer_text,
                "pmids": cited_pmids,
                "pmid_titles": {pmid: pipeline.get_title(pmid) for pmid in cited_pmids},
                "topk_used": ranked_pmids[:topk],
                "topk_titles": {pmid: pipeline.get_title(pmid) for pmid in ranked_pmids[:topk]},
                "gold_ideal": ideal_list,
                "gold_exact": question.get("exact_answer", []),
            }
        )

    mean = lambda values: sum(values) / len(values) if values else 0.0
    rouge_mean = mean(rouge_scores)
    exact_mean = mean(exact_scores)

    DEMO_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _time.strftime("%Y%m%d-%H%M%S")

    answers_path = DEMO_RUNS_DIR / f"{save_prefix}_{timestamp}.jsonl"
    metrics_path = DEMO_RUNS_DIR / f"{save_prefix}_metrics_{timestamp}.json"

    with answers_path.open("w") as handle:
        for item in answers:
            handle.write(json.dumps(item) + "\n")

    with metrics_path.open("w") as handle:
        json.dump(
            {"rougeL_mean": rouge_mean, "exact_mean": exact_mean},
            handle,
            indent=2,
        )

    print("ROUGE-L (vs ideal):", round(rouge_mean, 3), "  Exact:", round(exact_mean, 3))
    print("Saved:", answers_path.name, metrics_path.name)

    return {"rougeL_mean": rouge_mean, "exact_mean": exact_mean}


def preview_question_with_titles(
    question: Dict[str, Any],
    pipeline: RetrievalPipeline,
    qa_model: QaModelBundle,
    dense_name: str,
    topk: int = 10,
) -> None:
    ranked_pmids = pipeline.retrieve_with_dense_rerank(
        question["body"],
        dense_name=dense_name,
        topk=topk,
        pool_k=300,
    )

    print("Question:", question["body"])
    print("\nTop PMIDs and titles:")
    for pmid in ranked_pmids[:topk]:
        title = pipeline.get_title(pmid)
        print(f"- {pmid}: {title}")

    answer_text, cited_pmids = answer_with_pmids(
        question_text=question["body"],
        question_type=(question.get("type") or "summary").lower(),
        candidate_pmids=ranked_pmids,
        pipeline=pipeline,
        qa_model=qa_model,
    )

    print("\nLLM answer:")
    print(answer_text)
    print("\nCited PMIDs:", cited_pmids)
