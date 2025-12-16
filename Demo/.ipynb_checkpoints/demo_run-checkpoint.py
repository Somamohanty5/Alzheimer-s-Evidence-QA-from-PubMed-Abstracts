from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

from demo_utils import (
    DEMO_ROOT,
    create_demo_directories,
    load_bioasq_tree,
    filter_ad_questions,
)
from demo_retriever import RetrievalPipeline
from demo_evaluator import (
    build_full_corpus,
    evaluate_retrieval_variant,
    load_qa_model,
    evaluate_qa,
    preview_question_with_titles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Alzheimer’s Evidence QA demo on a small AD subset from demo/demo_data/."
    )
    parser.add_argument(
        "--demo-train-dir",
        type=Path,
        default=Path("demo_data/train"),
        help="Directory with demo train JSONs (subset of BioASQ train).",
    )
    parser.add_argument(
        "--demo-test-dir",
        type=Path,
        default=Path("demo_data/test"),
        help="Directory with demo test JSONs (subset of BioASQ golden test).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Number of documents to retrieve per question for QA and metrics.",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="somamohanty58@gmail.com",
        help="Contact email for NCBI E-utilities when fetching PubMed abstracts.",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="If set, run retrieval metrics only and skip TinyLlama QA.",
    )
    return parser.parse_args()


def load_ad_questions_from_dir(dir_path: Path) -> List[Dict[str, Any]]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Demo directory not found: {dir_path}")

    all_questions = load_bioasq_tree(dir_path)
    ad_questions = filter_ad_questions(all_questions)

    print(f"{dir_path}: loaded {len(ad_questions)} AD questions (after filtering).")
    return ad_questions


def main() -> None:
    args = parse_args()

    project_root = DEMO_ROOT
    print("Demo root:", project_root)

    create_demo_directories()

    # 1) Load AD-only questions
    train_root = project_root / args.demo_train_dir
    test_root = project_root / args.demo_test_dir

    ad_train_questions = load_ad_questions_from_dir(train_root)
    ad_test_questions = load_ad_questions_from_dir(test_root)

    if not ad_train_questions:
        raise RuntimeError("No AD questions found in demo train directory.")
    if not ad_test_questions:
        print("Warning: no AD questions found in demo test directory.")

    # 2) Build corpus from AD-TRAIN ONLY 
    corpus = build_full_corpus(
        train_questions=ad_train_questions,
        extra_questions=None,
        email=args.email,
    )
    print("Corpus PMIDs (from demo AD-train):", len(corpus.texts))

    # 3) Build retrieval pipeline (BM25 + TF-IDF)
    pipeline = RetrievalPipeline(corpus=corpus)
    pipeline.build()
    print("Whoosh BM25 index and TF-IDF matrix built for demo corpus.")

    # 4) Add SPECTER dense retriever and BGE reranker
    dense_name = "specter"
    print(f"Adding dense retriever: {dense_name}")
    pipeline.add_dense_model(dense_name, "sentence-transformers/allenai-specter")
    pipeline.load_bge_reranker()

    def retrieve_baseline(query_text: str, topk: int = 20):
        """BM25(relaxed) + TF-IDF(relaxed) baseline."""
        return pipeline.retrieve_bm25_tfidf(query_text, topk=topk)

    def retrieve_with_specter(query_text: str, topk: int = 20):
        """BM25 + TF-IDF + SPECTER + cross-encoder rerank."""
        return pipeline.retrieve_with_dense_rerank(
            query_text,
            dense_name=dense_name,
            topk=topk,
            pool_k=300,
        )

    # 5) Retrieval evaluation on demo AD-TRAIN and AD-TEST
    print("\n=== Demo AD-TRAIN retrieval metrics ===")
    demo_train_baseline_metrics = evaluate_retrieval_variant(
        questions=ad_train_questions,
        retrieve_fn=retrieve_baseline,
        variant_name="BM25(relaxed) + TF-IDF(relaxed) (demo-train)",
        topk=args.topk,
    )
    demo_train_dense_metrics = evaluate_retrieval_variant(
        questions=ad_train_questions,
        retrieve_fn=retrieve_with_specter,
        variant_name="BM25 + TF-IDF + SPECTER + reranker (demo-train)",
        topk=args.topk,
    )

    print("\n=== Demo AD-TEST retrieval metrics ===")
    demo_test_baseline_metrics = evaluate_retrieval_variant(
        questions=ad_test_questions,
        retrieve_fn=retrieve_baseline,
        variant_name="BM25(relaxed) + TF-IDF(relaxed) (demo-test)",
        topk=args.topk,
    )
    demo_test_dense_metrics = evaluate_retrieval_variant(
        questions=ad_test_questions,
        retrieve_fn=retrieve_with_specter,
        variant_name="BM25 + TF-IDF + SPECTER + reranker (demo-test)",
        topk=args.topk,
    )

    print("\nDemo retrieval summary:")
    print("  AD-TRAIN baseline:", demo_train_baseline_metrics)
    print("  AD-TRAIN +dense  :", demo_train_dense_metrics)
    print("  AD-TEST  baseline:", demo_test_baseline_metrics)
    print("  AD-TEST  +dense  :", demo_test_dense_metrics)

    if args.skip_qa:
        print("\n--skip-qa is set; skipping TinyLlama QA stage.")
    else:
        print("\n=== Loading QA model (TinyLlama) ===")
        qa_model = load_qa_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print("QA model loaded on device:", qa_model.device)

        print("\n=== Demo AD-TRAIN QA evaluation ===")
        demo_train_qa_metrics = evaluate_qa(
            questions=ad_train_questions,
            pipeline=pipeline,
            qa_model=qa_model,
            dense_name=dense_name,
            topk=args.topk,
            save_prefix="train_demo_answers20_llm",
        )
        print("Demo AD-TRAIN QA metrics:", demo_train_qa_metrics)

        print("\n=== Demo AD-TEST QA evaluation ===")
        demo_test_qa_metrics = evaluate_qa(
            questions=ad_test_questions,
            pipeline=pipeline,
            qa_model=qa_model,
            dense_name=dense_name,
            topk=args.topk,
            save_prefix="test_demo_answers20_llm",
        )
        print("Demo AD-TEST QA metrics:", demo_test_qa_metrics)

        if ad_test_questions:
            print("\n=== Example QA preview from demo AD-TEST ===")
            preview_question_with_titles(
                question=ad_test_questions[0],
                pipeline=pipeline,
                qa_model=qa_model,
                dense_name=dense_name,
                topk=args.topk,
            )


if __name__ == "__main__":
    main()
