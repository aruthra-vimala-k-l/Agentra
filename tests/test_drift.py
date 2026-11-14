import app.drift as drift


def setup_function():
    drift.reset()


def test_no_drift_reported_with_fewer_than_three_samples():
    result = drift.record_and_check("user-x", {"a": 1.0})
    assert result["drift_detected"] is False
    assert result["samples"] == 1


def test_drift_detected_when_a_signal_jumps_sharply():
    drift.record_and_check("user-y", {"a": 0.0, "b": 0.0})
    drift.record_and_check("user-y", {"a": 0.1, "b": 0.0})
    result = drift.record_and_check("user-y", {"a": 5.0, "b": 0.0})
    assert result["drift_detected"] is True
    assert "a" in result["drifted_signals"]
    assert "b" not in result["drifted_signals"]


def test_stable_signals_do_not_trigger_drift():
    for _ in range(4):
        result = drift.record_and_check("user-z", {"a": 1.0, "b": -0.5})
    assert result["drift_detected"] is False


def test_drift_history_is_isolated_per_user():
    drift.record_and_check("user-1", {"a": 10.0})
    drift.record_and_check("user-1", {"a": 10.0})
    result = drift.record_and_check("user-2", {"a": 0.0})
    assert result["samples"] == 1
