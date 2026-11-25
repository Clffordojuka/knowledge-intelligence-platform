import json
from random import Random
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.evaluation import retrieval_evaluator as evaluator


def test_normalize_text():
    assert evaluator.normalize_text(
        "  CREDIT   POLICY\n  APPROVAL "
    ) == "credit policy approval"


def test_calculate_metrics():
    relevant_id = uuid4()
    irrelevant_id = uuid4()
    metrics = evaluator.calculate_metrics(
        retrieved_ids=[irrelevant_id, relevant_id],
        relevant_ids={relevant_id},
    )
    assert metrics["precision_at_5"] == pytest.approx(0.2)
    assert metrics["recall_at_5"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["ndcg_at_5"] == pytest.approx(0.6309, abs=0.001)


def _aggregate_fixture():
    return [{
        "strategies": {
            strategy: {
                "metrics": {
                    "precision_at_5": 0.2,
                    "recall_at_5": 0.8,
                    "mrr": 0.5,
                    "ndcg_at_5": 0.6,
                },
                "latency": {
                    "mean_ms": 110.0,
                    "median_ms": 100.0,
                    "min_ms": 80.0,
                    "max_ms": 150.0,
                    "trials_ms": [80.0, 100.0, 150.0],
                },
                "latency_ms": 100.0,
            }
            for strategy in evaluator.STRATEGIES
        }
    }]


def test_calculate_aggregate_metrics():
    aggregate = evaluator.calculate_aggregate_metrics(_aggregate_fixture())
    assert set(aggregate) == set(evaluator.STRATEGIES)
    for strategy in evaluator.STRATEGIES:
        assert aggregate[strategy]["precision_at_5"] == pytest.approx(0.2)
        assert aggregate[strategy]["recall_at_5"] == pytest.approx(0.8)
        assert aggregate[strategy]["mrr"] == pytest.approx(0.5)
        assert aggregate[strategy]["ndcg_at_5"] == pytest.approx(0.6)
        assert aggregate[strategy]["mean_latency_ms"] == pytest.approx(110.0)
        assert aggregate[strategy]["median_latency_ms"] == pytest.approx(100.0)
        assert (
            aggregate[strategy]["overall_trial_mean_latency_ms"]
            == pytest.approx(110.0)
        )
        assert (
            aggregate[strategy]["overall_trial_median_latency_ms"]
            == pytest.approx(100.0)
        )


def test_calculate_aggregate_metrics_rejects_empty_results():
    with pytest.raises(evaluator.RetrievalEvaluationError, match="empty evaluation"):
        evaluator.calculate_aggregate_metrics([])


def test_summarize_latency():
    summary = evaluator._summarize_latency([10.0, 20.0, 30.0, 40.0, 50.0])
    assert summary["mean_ms"] == pytest.approx(30.0)
    assert summary["median_ms"] == pytest.approx(30.0)
    assert summary["min_ms"] == pytest.approx(10.0)
    assert summary["max_ms"] == pytest.approx(50.0)
    assert summary["trials_ms"] == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_summarize_latency_rejects_empty_samples():
    with pytest.raises(
        evaluator.RetrievalEvaluationError,
        match="empty latency sample",
    ):
        evaluator._summarize_latency([])

def test_load_dataset_validates_dataset(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "policy.txt").write_text("Sample policy document.", encoding="utf-8")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps({
            "dataset_name": "test_dataset",
            "version": "1.0",
            "corpus_files": ["policy.txt"],
            "queries": [{
                "id": "q001",
                "query": "What is the policy?",
                "relevant_evidence": [{
                    "filename": "policy.txt",
                    "phrase": "Sample policy",
                }],
            }],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluator, "CORPUS_DIR", corpus_dir)
    monkeypatch.setattr(evaluator, "DATASET_PATH", dataset_path)
    dataset = evaluator.load_dataset()
    assert dataset["dataset_name"] == "test_dataset"
    assert dataset["corpus_files"] == ["policy.txt"]
    assert len(dataset["queries"]) == 1


def test_load_dataset_rejects_missing_corpus_file(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps({
            "dataset_name": "test_dataset",
            "version": "1.0",
            "corpus_files": ["missing.txt"],
            "queries": [{
                "id": "q001",
                "query": "Test query",
                "relevant_evidence": [{
                    "filename": "missing.txt",
                    "phrase": "test",
                }],
            }],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluator, "CORPUS_DIR", tmp_path)
    monkeypatch.setattr(evaluator, "DATASET_PATH", dataset_path)
    with pytest.raises(evaluator.RetrievalEvaluationError, match="does not exist"):
        evaluator.load_dataset()


def _query_item():
    return {
        "id": "q001",
        "query": "What is the credit policy?",
        "topic": "credit",
        "relevant_evidence": [{
            "filename": "policy.txt",
            "phrase": "credit policy",
        }],
    }


def _patch_retrieval_operations(monkeypatch, document_id, relevant_id, irrelevant_id):
    calls = {"semantic": [], "bm25": [], "hybrid": []}

    def fake_semantic(**kwargs):
        calls["semantic"].append(kwargs)
        chunk_id = (
            relevant_id
            if kwargs.get("query_instruction") is None
            else irrelevant_id
        )
        return [SimpleNamespace(chunk_id=chunk_id, document_id=document_id)]

    def fake_bm25(**kwargs):
        calls["bm25"].append(kwargs)
        return [SimpleNamespace(chunk_id=relevant_id, document_id=document_id)]

    def fake_hybrid(**kwargs):
        calls["hybrid"].append(kwargs)
        if kwargs.get("rerank") or kwargs.get("query_instruction") is not None:
            chunk_id = irrelevant_id
        else:
            chunk_id = relevant_id
        return [SimpleNamespace(chunk_id=chunk_id, document_id=document_id)]

    monkeypatch.setattr(evaluator, "search_similar_chunks", fake_semantic)
    monkeypatch.setattr(evaluator, "search_bm25", fake_bm25)
    monkeypatch.setattr(evaluator, "search_hybrid_chunks", fake_hybrid)
    return calls


def test_evaluate_query_runs_all_strategies_with_warmups_and_trials(monkeypatch):
    document_id = uuid4()
    relevant_id = uuid4()
    irrelevant_id = uuid4()
    monkeypatch.setattr(
        evaluator, "resolve_relevant_chunks", lambda **kwargs: {relevant_id}
    )
    calls = _patch_retrieval_operations(
        monkeypatch, document_id, relevant_id, irrelevant_id
    )
    result = evaluator.evaluate_query(
        db=object(),
        query_item=_query_item(),
        document_ids={"policy.txt": document_id},
        reranker=object(),
        warmup_runs=2,
        measured_trials=3,
        rng=Random(123),
    )

    assert set(result["strategies"]) == set(evaluator.STRATEGIES)
    assert result["warmup_runs_per_strategy"] == 2
    assert result["measured_trials_per_strategy"] == 3
    assert len(result["trial_execution_orders"]) == 3
    assert len(calls["semantic"]) == 2 * (2 + 3)
    assert len(calls["bm25"]) == 2 + 3
    assert len(calls["hybrid"]) == 3 * (2 + 3)

    assert sum(
        call.get("query_instruction") == evaluator.QUERY_INSTRUCTION
        for call in calls["semantic"]
    ) == 2 + 3
    assert sum(
        call.get("query_instruction") == evaluator.QUERY_INSTRUCTION
        for call in calls["hybrid"]
    ) == 2 + 3

    for order in result["trial_execution_orders"]:
        assert len(order) == len(evaluator.STRATEGIES)
        assert set(order) == set(evaluator.STRATEGIES)

    for strategy_result in result["strategies"].values():
        assert len(strategy_result["latency"]["trials_ms"]) == 3
        assert strategy_result["latency"]["mean_ms"] >= 0
        assert strategy_result["latency"]["median_ms"] >= 0
        assert strategy_result["retrieved_count"] == 1

    assert result["strategies"]["semantic"]["metrics"]["mrr"] == 1.0
    assert result["strategies"]["semantic_instructed"]["metrics"]["mrr"] == 0.0
    assert result["strategies"]["hybrid"]["metrics"]["mrr"] == 1.0
    assert result["strategies"]["hybrid_instructed"]["metrics"]["mrr"] == 0.0


def test_evaluate_query_uses_deterministic_trial_orders(monkeypatch):
    document_id = uuid4()
    relevant_id = uuid4()
    irrelevant_id = uuid4()
    monkeypatch.setattr(
        evaluator, "resolve_relevant_chunks", lambda **kwargs: {relevant_id}
    )
    _patch_retrieval_operations(
        monkeypatch, document_id, relevant_id, irrelevant_id
    )

    first = evaluator.evaluate_query(
        db=object(),
        query_item=_query_item(),
        document_ids={"policy.txt": document_id},
        warmup_runs=0,
        measured_trials=4,
        rng=Random(42),
    )
    second = evaluator.evaluate_query(
        db=object(),
        query_item=_query_item(),
        document_ids={"policy.txt": document_id},
        warmup_runs=0,
        measured_trials=4,
        rng=Random(42),
    )
    assert first["trial_execution_orders"] == second["trial_execution_orders"]


@pytest.mark.parametrize(
    ("warmup_runs", "measured_trials"),
    [
        (-1, 1),
        (1, 0),
        (True, 1),
        (1, False),
        (1.5, 2),
        (1, "3"),
    ],
)
def test_evaluate_query_rejects_invalid_trial_configuration(
    warmup_runs, measured_trials
):
    with pytest.raises(evaluator.RetrievalEvaluationError):
        evaluator.evaluate_query(
            db=object(),
            query_item=_query_item(),
            document_ids={"policy.txt": uuid4()},
            warmup_runs=warmup_runs,
            measured_trials=measured_trials,
        )


def test_evaluate_query_rejects_results_outside_benchmark_scope(monkeypatch):
    benchmark_document_id = uuid4()
    outside_document_id = uuid4()
    monkeypatch.setattr(
        evaluator, "resolve_relevant_chunks", lambda **kwargs: {uuid4()}
    )
    monkeypatch.setattr(
        evaluator,
        "search_similar_chunks",
        lambda **kwargs: [
            SimpleNamespace(chunk_id=uuid4(), document_id=outside_document_id)
        ],
    )
    monkeypatch.setattr(evaluator, "search_bm25", lambda **kwargs: [])
    monkeypatch.setattr(evaluator, "search_hybrid_chunks", lambda **kwargs: [])

    with pytest.raises(
        evaluator.RetrievalEvaluationError,
        match="outside the benchmark corpus",
    ):
        evaluator.evaluate_query(
            db=object(),
            query_item=_query_item(),
            document_ids={"policy.txt": benchmark_document_id},
            warmup_runs=0,
            measured_trials=1,
        )