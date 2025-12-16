# Alzheimer’s Evidence QA from PubMed Abstracts

This project builds a small **retrieval-augmented question answering (QA)** system for Alzheimer’s disease (AD).

Given a natural-language question from the **BioASQ** benchmark, the system:

- Retrieves relevant PubMed abstracts about Alzheimer’s disease  
- Ranks them using a hybrid of keyword search and neural embeddings  
- Generates a short, citation-style answer using a lightweight language model  

The repository contains:

- A **full Alzheimer’s experiment** (using the AD subset of BioASQ Tasks 9–13B), and  
- A **demo setup** with a much smaller sample that can be run quickly on a single T4 GPU  

For quick experimentation, all you need is  the `demo/` folder.

---

## 1. Dataset

### Full dataset (for completeness)

The full BioASQ Task B datasets (Tasks 9–13B) can be downloaded from the  
BioASQ participants area (registration required):

- **BioASQ Task B datasets**:  
  <https://participants-area.bioasq.org/datasets/>

In this project, We extracted the **Alzheimer’s / dementia questions** from Tasks 9–13B and
built an AD-specific corpus of PubMed abstracts.

### Demo dataset (used for grading)

For the demo, We provide a **small sampled subset** of the AD questions and their
supporting PubMed evidence:

- `demo/demo_data/train/` – sampled AD training questions (BioASQ JSON format)  
- `demo/demo_data/test/` – sampled AD test questions (BioASQ JSON format)  

This demo subset is enough to run the full pipeline (retrieval + QA) and reproduce the
qualitative behavior reported in the report, but on a much smaller, faster-to-run dataset.

---

## 2. Repository Structure

High-level layout (only the important pieces):

- `README.md` – this file. High-level description, how to run, and expected outputs.  
- `requirements.txt` – Python dependencies (Whoosh, scikit-learn, sentence-transformers,  
  transformers, Biopython, rouge-score, etc.).

### Core source code (`src/`)

These files implement the main logic used in both the full run and the demo.

#### `src/utils.py`

- Loads BioASQ JSON files and normalizes questions.  
- Filters questions down to **Alzheimer’s / dementia** using simple keyword rules.  
- Builds a snippet-only corpus for each PubMed ID (PMID).  
- Implements retrieval metrics such as **Recall@k, MAP, nDCG@10** and  
  **cap-normalized R@10** used in the report.

#### `src/retriever.py`

- Defines `CorpusData`, a small container for `pmid → text` and `pmid → title`.  
- Implements `RetrievalPipeline`, which:
  - Builds a **Whoosh BM25** index over the Alzheimer’s corpus  
  - Builds a **TF-IDF** matrix with scikit-learn  
  - Adds one or more **dense embedding models** (SPECTER in this project)  
  - Optionally loads a **BGE cross-encoder reranker**  
- Provides methods for:
  - `retrieve_bm25_tfidf(...)` – keyword baseline  
  - `retrieve_dense(...)` – dense retrieval  
  - `retrieve_with_dense_rerank(...)` – full BM25 + TF-IDF + SPECTER + BGE pipeline  
- Also exposes helper methods to get titles and full document text by PMID.

#### `src/evaluator.py`

- Fetches PubMed titles and abstracts via **NCBI E-utilities** and merges them with
  BioASQ snippets to build the full Alzheimer’s evidence corpus.  
- Computes retrieval metrics for a given retrieval function over a set of questions.  
- Selects the **best dense model** (SPECTER, after a small development experiment).  
- Wraps the QA model (**TinyLlama-1.1B-Chat**) and builds prompts that:
  - Show the question and a set of evidence sentences tagged by PMID  
  - Ask for **exactly three sentences**, each ending with a PMID  
- Runs QA evaluation (**ROUGE-L** and an **exact-match heuristic**) and saves:
  - A `*.jsonl` file with each question, answer, and cited PMIDs  
  - A `*_metrics.json` file with averaged ROUGE-L and exact-match scores  
- Provides a `preview_question_with_titles(...)` helper to print:
  - The question  
  - Top retrieved PMIDs + titles  
  - A sample LLM answer and its cited PMIDs  

---

## 3. Demo Code (`Demo/`)

Everything needed for **quick run** is in this folder.

- `Demo/demo_data/train/`  
  Sampled AD training questions (BioASQ JSON).

- `Demo/demo_data/test/`  
  Sampled AD test questions (BioASQ JSON).

- `Demo/demo_run.py`  
  A **self-contained script** that:
  1. Loads the demo train/test questions from `demo_data/`.  
  2. Builds the Alzheimer’s evidence corpus for the demo train questions  
     (snippets + PubMed abstracts via NCBI).  
  3. Constructs a `RetrievalPipeline` with:
     - BM25 + TF-IDF (keyword baseline)  
     - SPECTER dense embeddings  
     - BGE cross-encoder reranker  
  4. Evaluates:
     - Baseline vs. hybrid retrieval on demo train and demo test  
       (Recall@10, nDCG@10, MAP, cap-normalized R@10).  
  5. Loads TinyLlama and evaluates QA on demo train and test  
     (ROUGE-L and exact-match), saving outputs as JSON/JSONL in `Demo/runs/`.  
  6. Prints a human-readable preview for one test question, including the  
     *“losartan and brain atrophy”* example described in the report.

- `Demo/demo.ipynb`  
  A notebook version of the demo script.  
  This is what We used in Google Colab with a T4 GPU:
  1. Upload the zipped `Demo/` folder and `requirements.txt`.  
  2. Unzip in the Colab workspace.  
  3. Open `Demo/demo.ipynb`.  
  4. Run all cells **sequentially** (they mirror the logic in `demo_run.py`).  

---

## 4. How to Run the Demo (Command Line)

You can reproduce the demo results with this single command line workflow.

```bash
# 1. Clone the repository and move into it
git clone https://github.com/Somamohanty5/Alzheimer-s-Evidence-QA-from-PubMed-Abstracts.git
cd Alzheimer-s-Evidence-QA-from-PubMed-Abstracts

# 2. (Optional) create and activate a virtual environment

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the demo (retrieval + QA) on the sampled demo dataset
python Demo/demo_run.py --topk 20
```
Argument

--topk – number of documents to keep per question for metrics and QA
(default: 20; smaller values like --topk 10 will run slightly faster).

Expected console output
Retrieval metrics for demo train and demo test:

Recall@10

nDCG@10

MAP

Cap-normalized R@10

QA metrics for demo train and demo test:

Mean ROUGE-L

Mean exact-match

A short preview for one AD question:

Question text

Top PMIDs + titles

A three-sentence TinyLlama answer with PMIDs in brackets

Expected files
The script writes outputs under:

Demo/runs/*.jsonl – one line per question with its answer, cited PMIDs, and titles.

Demo/runs/*_metrics_*.json – summary ROUGE-L and exact-match scores.

For grading, it is enough to clone the repo, ensure demo/ is present, and run
either:
python Demo/demo_run.py --topk 20
or open the notebook:
Demo/demo.ipynb
on a T4 GPU and run all cells.
