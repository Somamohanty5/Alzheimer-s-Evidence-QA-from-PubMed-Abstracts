from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from collections import OrderedDict
from pathlib import Path

from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh.writing import AsyncWriter
from whoosh.qparser import MultifieldParser, OrGroup, FuzzyTermPlugin
from whoosh.query import Or, Term, Prefix
from whoosh import scoring

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from demo_utils import DEMO_INDEX_DIR


@dataclass
class CorpusData:
    texts: Dict[str, str]
    titles: Dict[str, str]


class RetrievalPipeline:

    def __init__(self, corpus: CorpusData, index_dir: Path | None = None, device: str | None = None) -> None:
        self.corpus = corpus
        self.index_dir = index_dir or DEMO_INDEX_DIR

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.searcher = None
        self.query_parser = None

        self.pmid_list: List[str] = []
        self.text_list: List[str] = []

        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None

        self.dense_models: Dict[str, SentenceTransformer] = {}
        self.dense_corpus: Dict[str, np.ndarray] = {}

        self.reranker: CrossEncoder | None = None
        self.max_reranker_text: int = 2000

    # ---------- Public API ----------

    def build(self) -> None:
        """Build index, BM25 searcher, and TF-IDF matrix."""
        self._prepare_corpus_lists()
        self._build_whoosh_index()
        self._build_tfidf_matrix()

    def add_dense_model(self, name: str, model_id: str, batch_size: int = 64) -> None:
        """Add a dense retriever model and pre-compute embeddings for the corpus."""
        model = SentenceTransformer(model_id, device=self.device)
        embeddings = model.encode(
            self.text_list,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self.dense_models[name] = model
        self.dense_corpus[name] = np.asarray(embeddings, dtype=np.float32)

    def load_bge_reranker(self, model_id: str = "BAAI/bge-reranker-base") -> None:
        self.reranker = CrossEncoder(model_id, device=self.device)

    def retrieve_bm25(self, query_text: str, k: int = 80) -> List[str]:
        query = self._build_boosted_query(query_text)
        results = self.searcher.search(query, limit=k)
        return [hit["pmid"] for hit in results]

    def retrieve_tfidf(self, query_text: str, k: int = 80) -> List[str]:
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("TF-IDF matrix has not been built. Call build() first.")

        query_vector = self.tfidf_vectorizer.transform([query_text])
        cosine_scores = linear_kernel(query_vector, self.tfidf_matrix).ravel()
        indices = cosine_scores.argsort()[::-1][:k]
        return [self.pmid_list[index] for index in indices]

    def retrieve_dense(self, model_name: str, query_text: str, topk: int = 150) -> List[str]:
        if model_name not in self.dense_models:
            raise ValueError(f"Dense model '{model_name}' not available.")

        model = self.dense_models[model_name]
        corpus_embeddings = self.dense_corpus[model_name]

        query_embedding = model.encode([query_text], normalize_embeddings=True)
        scores = (query_embedding @ corpus_embeddings.T).ravel()

        num_items = min(topk, len(scores) - 1)
        indices = np.argpartition(-scores, kth=num_items)[:topk]
        indices = indices[np.argsort(-scores[indices])]
        return [self.pmid_list[index] for index in indices]

    def retrieve_bm25_tfidf(self, query_text: str, topk: int = 20) -> List[str]:
        """
        Baseline variant: BM25(relaxed) + TF-IDF(relaxed) without dense retrieval.
        """
        bm25_ids = self.retrieve_bm25(query_text, k=max(topk, 80))
        tfidf_ids = self.retrieve_tfidf(query_text, k=max(topk, 80))

        ordered: OrderedDict[str, None] = OrderedDict()
        for pmid in bm25_ids + tfidf_ids:
            ordered.setdefault(pmid, None)
        return list(ordered.keys())[:topk]

    def retrieve_with_dense_rerank(
        self,
        query_text: str,
        dense_name: str,
        topk: int = 20,
        pool_k: int = 300,
    ) -> List[str]:
        """
        Full variant: fused BM25 + TF-IDF + dense + BGE reranker.
        """
        if self.reranker is None:
            raise RuntimeError("Cross-encoder reranker not loaded. Call load_bge_reranker().")

        pool_pmids = self._build_fused_pool(query_text, dense_name, pool_k)
        text_pairs = [
            (query_text, self._get_truncated_text(pmid))
            for pmid in pool_pmids
        ]
        scores = self.reranker.predict(text_pairs, batch_size=32)
        sorted_pmids = [
            pmid for _, pmid in sorted(zip(scores, pool_pmids), reverse=True)
        ]
        return sorted_pmids[:topk]

    def get_title(self, pmid: str) -> str:
        return self.corpus.titles.get(pmid, "")

    def get_document_text(self, pmid: str) -> str:
        return self.corpus.texts.get(pmid, "")

    # ---------- Internal helpers ----------

    def _prepare_corpus_lists(self) -> None:
        self.pmid_list = list(self.corpus.texts.keys())
        self.text_list = [self.corpus.texts[pmid] for pmid in self.pmid_list]

    def _build_whoosh_index(self) -> None:
        if self.index_dir.exists():
            import shutil
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        schema = Schema(
            pmid=ID(stored=True, unique=True),
            title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            text=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        )

        index = create_in(self.index_dir, schema)
        writer = AsyncWriter(index)

        for pmid in self.pmid_list:
            writer.add_document(
                pmid=str(pmid),
                title=self.corpus.titles.get(pmid, ""),
                text=self.corpus.texts.get(pmid, ""),
            )
        writer.commit()

        self.searcher = index.searcher(weighting=scoring.BM25F())
        parser = MultifieldParser(["title", "text"], schema=index.schema, group=OrGroup)
        parser.add_plugin(FuzzyTermPlugin())
        self.query_parser = parser

    def _build_tfidf_matrix(self) -> None:
        vectorizer = TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),
            stop_words="english",
        )
        matrix = vectorizer.fit_transform(self.text_list)
        self.tfidf_vectorizer = vectorizer
        self.tfidf_matrix = matrix

    def _build_boosted_query(self, query_text: str):
        base_query = self.query_parser.parse(query_text)
        bonus_terms = Or(
            [
                Prefix("text", "alzheimer"),
                Term("text", "alzheimer's"),
                Term("text", "dementia"),
                Term("text", "amyloid"),
                Term("text", "tau"),
            ]
        )
        return Or([base_query, bonus_terms])

    def _build_fused_pool(
        self,
        query_text: str,
        dense_name: str,
        pool_k: int,
        bm25_k: int = 80,
        tfidf_k: int = 80,
        dense_k: int = 150,
    ) -> List[str]:
        bm25_ids = self.retrieve_bm25(query_text, k=bm25_k)
        tfidf_ids = self.retrieve_tfidf(query_text, k=tfidf_k)
        dense_ids = self.retrieve_dense(dense_name, query_text, topk=dense_k)

        ordered: OrderedDict[str, None] = OrderedDict()
        for pmid in bm25_ids + tfidf_ids + dense_ids:
            ordered.setdefault(pmid, None)

        return list(ordered.keys())[:pool_k]

    def _get_truncated_text(self, pmid: str) -> str:
        full_text = self.corpus.texts.get(pmid, "")
        return full_text[: self.max_reranker_text]
