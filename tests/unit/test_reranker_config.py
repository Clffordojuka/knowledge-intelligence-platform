from app.core.config import Settings


def test_reranker_configuration_defaults():
    config = Settings(_env_file=None)

    assert config.reranker_model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.reranker_batch_size == 16
    assert config.reranker_max_length == 512


def test_reranker_configuration_can_be_overridden():
    config = Settings(
        _env_file=None,
        reranker_model_name="custom-reranker",
        reranker_batch_size=8,
        reranker_max_length=256,
    )

    assert config.reranker_model_name == "custom-reranker"
    assert config.reranker_batch_size == 8
    assert config.reranker_max_length == 256
