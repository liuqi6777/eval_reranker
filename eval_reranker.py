import argparse
import tempfile
import os
import json
import shutil
import numpy
import time
import torch

from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import colbert_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trec_eval import EvalFunction
from pyserini.search import LuceneSearcher, get_topics, get_qrels


INDEX = {
    'bm25': {
        'msmarco-dev': 'msmarco-v1-passage',
        'dl19': 'msmarco-v1-passage',
        'dl20': 'msmarco-v1-passage',
        'covid': 'beir-v1.0.0-trec-covid.flat',
        'arguana': 'beir-v1.0.0-arguana.flat',
        'touche': 'beir-v1.0.0-webis-touche2020.flat',
        'news': 'beir-v1.0.0-trec-news.flat',
        'scifact': 'beir-v1.0.0-scifact.flat',
        'fiqa': 'beir-v1.0.0-fiqa.flat',
        'scidocs': 'beir-v1.0.0-scidocs.flat',
        'nfc': 'beir-v1.0.0-nfcorpus.flat',
        'quora': 'beir-v1.0.0-quora.flat',
        'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
        'fever': 'beir-v1.0.0-fever.flat',
        'robust04': 'beir-v1.0.0-robust04.flat',
        'signal': 'beir-v1.0.0-signal1m.flat',
        'nq': 'beir-v1.0.0-nq.flat',
        'cfever': 'beir-v1.0.0-climate-fever.flat',
        'hotpotqa': 'beir-v1.0.0-hotpotqa.flat'
    },
    'splade++ed': {
        'msmarco-dev': 'msmarco-v1-passage-splade-pp-ed-text',
        "dl19": "msmarco-v1-passage-splade-pp-ed-text",
        "dl20": "msmarco-v1-passage-splade-pp-ed-text",
        "covid": "beir-v1.0.0-trec-covid.splade-pp-ed",
        "arguana": "beir-v1.0.0-arguana.splade-pp-ed",
        "touche": "beir-v1.0.0-webis-touche2020.splade-pp-ed",
        "news": "beir-v1.0.0-trec-news.splade-pp-ed",
        "scifact": "beir-v1.0.0-scifact.splade-pp-ed",
        "fiqa": "beir-v1.0.0-fiqa.splade-pp-ed",
        "scidocs": "beir-v1.0.0-scidocs.splade-pp-ed",
        "nfc": "beir-v1.0.0-nfcorpus.splade-pp-ed",
        "quora": "beir-v1.0.0-quora.splade-pp-ed",
        "dbpedia": "beir-v1.0.0-dbpedia-entity.splade-pp-ed",
        "fever": "beir-v1.0.0-fever.splade-pp-ed",
        "robust04": "beir-v1.0.0-robust04.splade-pp-ed",
        "signal": "beir-v1.0.0-signal1m.splade-pp-ed",
        'nq': 'beir-v1.0.0-nq.splade-pp-ed',
        'cfever': 'beir-v1.0.0-climate-fever.splade-pp-ed',
        'hotpotqa': 'beir-v1.0.0-hotpotqa.splade-pp-ed'
    }
}

TOPICS = {
    'msmarco-dev': 'msmarco-passage-dev-subset',
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'covid': 'beir-v1.0.0-trec-covid-test',
    'arguana': 'beir-v1.0.0-arguana-test',
    'touche': 'beir-v1.0.0-webis-touche2020-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'scidocs': 'beir-v1.0.0-scidocs-test',
    'nfc': 'beir-v1.0.0-nfcorpus-test',
    'quora': 'beir-v1.0.0-quora-test',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
    'fever': 'beir-v1.0.0-fever-test',
    'robust04': 'beir-v1.0.0-robust04-test',
    'signal': 'beir-v1.0.0-signal1m-test',
    'nq': 'beir-v1.0.0-nq-test',
    'cfever': 'beir-v1.0.0-climate-fever-test',
    'hotpotqa': 'beir-v1.0.0-hotpotqa-test'
}


def run_retriever(topics, searcher, qrels=None, topk=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=topk)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=topk)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks


@torch.no_grad()
def run_cross_rerank(retrieval_results, model, tokenizer):
    model.eval()
    model.to('cuda')
    rerank_results = []
    all_queries = [hit['query'] for hit in retrieval_results]
    s = time.time()
    for i in tqdm(range(0, len(all_queries))):
        all_passages = [hit['content'] for hit in retrieval_results[i]['hits']]
        if len(all_passages) == 0:
            continue
        inputs = tokenizer(
            [(all_queries[i], passage) for passage in all_passages],
            return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        scores = model(**inputs).logits.flatten().cpu().numpy().tolist()
        ranking = numpy.argsort(scores)[::-1]
        rerank_results.append({'query': retrieval_results[i]['query'], 'hits': []})
        for j in range(0, len(ranking)):
            hit = retrieval_results[i]['hits'][ranking[j]]
            hit['score'] = scores[ranking[j]]
            rerank_results[-1]['hits'].append(hit)
    t = time.time() - s
    print(f"reranking latency per query: {t / len(all_queries):.4f} seconds.")
    return rerank_results


@torch.no_grad()
def run_colbert_rerank(retrieval_results, model):
    rerank_results = []
    all_queries = [hit['query'] for hit in retrieval_results]
    s = time.time()
    for i in tqdm(range(0, len(all_queries))):
        all_passages = [hit['content'] for hit in retrieval_results[i]['hits']]
        if len(all_passages) == 0:
            continue
        Q = model.queryFromText([all_queries[i]])
        D = model.docFromText(all_passages, bsize=32)[0]
        D_mask = torch.ones(D.shape[:2], dtype=torch.long)
        scores = colbert_score(Q, D, D_mask).flatten().cpu().numpy().tolist()
        ranking = numpy.argsort(scores)[::-1]
        rerank_results.append({'query': retrieval_results[i]['query'], 'hits': []})
        for j in range(0, len(ranking)):
            hit = retrieval_results[i]['hits'][ranking[j]]
            hit['score'] = scores[ranking[j]]
            rerank_results[-1]['hits'].append(hit)
    t = time.time() - s
    print(f"reranking latency per query: {t / len(all_queries):.4f} seconds.")
    return rerank_results


def write_retrival_results(rank_results, file, output_content=True):
    with open(file, 'w') as f:
        for item in rank_results:
            if not output_content:
                item['hits'] = [{'qid': hit['qid'], 'docid': hit['docid'], 'score': hit['score']} 
                                for hit in item['hits']]
            f.write((json.dumps(item) + '\n'))
    return True


def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True


def eval_dataset(dataset, retriver, reranker, topk=100):
    print('#' * 20)
    print(f'Evaluation on {dataset}')
    print('#' * 20)

    
    retrieval_results_file = f'results/{dataset}_retrival_{retriver}_top{topk}.jsonl'
    if os.path.exists(retrieval_results_file):
        with open(retrieval_results_file) as f:
            retrieval_results = [json.loads(line) for line in f]
    else:
        # Retrieve passages using pyserini BM25.
        # TODO: Add support for other retrievers, including SPLADE++ED and Dense retrieval.
        try:
            searcher = LuceneSearcher.from_prebuilt_index(INDEX[retriver][dataset])
            topics = get_topics(TOPICS[dataset] if dataset != 'dl20' else 'dl20')
            qrels = get_qrels(TOPICS[dataset])
            retrieval_results = run_retriever(topics, searcher, qrels, topk=topk)
            write_retrival_results(
                retrieval_results, 
                f'results/{dataset}_retrival_bm25_top{topk}.jsonl')
        except:
            print(f'Failed to retrieve passages for {dataset}')
            return
        
    # Evaluate nDCG@10
    output_file = tempfile.NamedTemporaryFile(delete=False).name
    write_eval_file(retrieval_results, output_file)
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', TOPICS[dataset], output_file])
    # Rename the output file to a better name
    shutil.move(output_file, f'results/eval_{dataset}_{retriver}.txt')
    
    # Rerank
    rerank_results_file = f'results/{dataset}_rerank_{reranker.split("/")[-1]}_top{topk}.jsonl'
    if os.path.exists(rerank_results_file):
        with open(rerank_results_file) as f:
            rerank_results = [json.loads(line) for line in f]
    else:
        if 'colbert' in reranker:
            colbert_config = ColBERTConfig(query_maxlen=32, doc_maxlen=512)
            colbert_reranker = Checkpoint(args.reranker, colbert_config=colbert_config)
            rerank_results = run_colbert_rerank(retrieval_results, colbert_reranker)
        else:
            tokenizer = AutoTokenizer.from_pretrained(reranker)
            model = AutoModelForSequenceClassification.from_pretrained(
                reranker, num_labels=1, trust_remote_code=True)
            rerank_results = run_cross_rerank(retrieval_results, model, tokenizer)
        write_retrival_results(
            rerank_results, 
            f'results/{dataset}_rerank_{reranker.split("/")[-1]}_top{topk}.jsonl',
            output_content=False
        )

    # Evaluate nDCG@10
    output_file = tempfile.NamedTemporaryFile(delete=False).name
    write_eval_file(rerank_results, output_file)
    EvalFunction.eval(['-c', '-m', 'ndcg_cut.10', TOPICS[dataset], output_file])
    # Rename the output file to a better name
    shutil.move(output_file, f'results/eval_{dataset}_{retriver}_{reranker.split("/")[-1]}.txt')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--retriver', type=str, default='bm25', choices=['bm25', 'splade++ed'])
    parser.add_argument('--reranker', type=str, default='jinaai/jina-colbert-v1-en')
    parser.add_argument('--topk', type=int, default=100)
    args = parser.parse_args()
    eval_dataset(args.dataset, args.retriver, args.reranker, args.topk)
