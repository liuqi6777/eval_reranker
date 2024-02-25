# Evaluate your reranker quickly!

## Usage

Install [Pyserini](https://github.com/castorini/pyserini) and [ColBERT](https://github.com/stanford-futuredata/ColBERT), please refer to the original installation guide.

Run the following command to evaluate your reranker:

```bash
python eval_reranker.py --dataset $dataset --reranker $model
```

Supported datasets are listed in the file. Now, the eval code only supports BM25 as the first-stage retriever and only supports ColBERT and cross-encoder as the reranker.

