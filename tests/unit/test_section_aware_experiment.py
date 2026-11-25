
import hashlib

import pytest

from app.evaluation import section_aware_experiment as experiment


def test_create_line_ending_variant_preserves_text_and_changes_hash():
    original = (
        b"1. BORROWER ELIGIBILITY\n"
        b"Applicants must provide valid identification.\n"
        b"\n"
        b"2. REPAYMENT CAPACITY\n"
        b"Income must be assessed.\n"
    )

    variant = experiment.create_line_ending_variant(original)

    original_text = original.decode("utf-8")
    variant_text = variant.decode("utf-8")

    assert experiment._normalize_newlines(original_text) == (
        experiment._normalize_newlines(variant_text)
    )
    assert hashlib.sha256(original).digest() != hashlib.sha256(
        variant
    ).digest()
    assert b"\r\n" in variant


def test_create_line_ending_variant_handles_existing_crlf():
    original = (
        b"1. BORROWER ELIGIBILITY\r\n"
        b"Applicants must provide valid identification.\r\n"
        b"2. REPAYMENT CAPACITY\r\n"
        b"Income must be assessed.\r\n"
    )

    variant = experiment.create_line_ending_variant(original)

    assert b"\r\n" not in variant
    assert b"\n" in variant
    assert experiment._normalize_newlines(
        original.decode("utf-8")
    ) == experiment._normalize_newlines(
        variant.decode("utf-8")
    )
    assert hashlib.sha256(original).digest() != hashlib.sha256(
        variant
    ).digest()


def test_create_line_ending_variant_rejects_single_line_content():
    with pytest.raises(
        experiment.SectionAwareExperimentError,
        match="single-line",
    ):
        experiment.create_line_ending_variant(
            b"A document with no newline."
        )


def test_create_line_ending_variant_rejects_invalid_utf8():
    with pytest.raises(
        experiment.SectionAwareExperimentError,
        match="valid UTF-8",
    ):
        experiment.create_line_ending_variant(
            b"\xff\xfe\xfa"
        )


def test_build_metric_comparison_calculates_deltas():
    baseline = {
        strategy: {
            metric: 0.5
            for metric in experiment.evaluator.METRIC_NAMES
        }
        for strategy in experiment.evaluator.STRATEGIES
    }

    section_aware = {
        strategy: {
            metric: 0.7
            for metric in experiment.evaluator.METRIC_NAMES
        }
        for strategy in experiment.evaluator.STRATEGIES
    }

    comparison = experiment.build_metric_comparison(
        baseline_aggregate=baseline,
        experimental_aggregate=section_aware,
    )

    assert set(comparison) == set(experiment.evaluator.STRATEGIES)

    for strategy in experiment.evaluator.STRATEGIES:
        for metric in experiment.evaluator.METRIC_NAMES:
            assert comparison[strategy][metric] == {
                "baseline": 0.5,
                "section_aware": 0.7,
                "delta": 0.2,
            }


def test_build_metric_comparison_handles_negative_delta():
    baseline = {
        strategy: {
            metric: 0.8
            for metric in experiment.evaluator.METRIC_NAMES
        }
        for strategy in experiment.evaluator.STRATEGIES
    }

    section_aware = {
        strategy: {
            metric: 0.6
            for metric in experiment.evaluator.METRIC_NAMES
        }
        for strategy in experiment.evaluator.STRATEGIES
    }

    comparison = experiment.build_metric_comparison(
        baseline_aggregate=baseline,
        experimental_aggregate=section_aware,
    )

    assert comparison["hybrid"]["mrr"]["delta"] == -0.2