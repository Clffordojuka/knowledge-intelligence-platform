from app.database import dependencies


class FakeSession:
    """A lightweight session substitute for dependency testing."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_get_db_creates_and_closes_session(monkeypatch):
    fake_session = FakeSession()

    monkeypatch.setattr(
        dependencies,
        "SessionLocal",
        lambda: fake_session,
    )

    db_generator = dependencies.get_db()

    # The dependency provides the session.
    yielded_session = next(db_generator)

    assert yielded_session is fake_session
    assert fake_session.closed is False

    # Finishing the dependency should close the session.
    try:
        next(db_generator)
    except StopIteration:
        pass

    assert fake_session.closed is True
