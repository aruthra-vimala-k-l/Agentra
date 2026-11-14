from app import sample_runs
from app.dashboard_service import build_dashboard


def test_sample_runs_fill_the_research_panels_from_real_pipeline_runs():
    before = build_dashboard()["total_evaluated_purchases"]
    out = sample_runs.run(n=6, seed=1)
    assert out["sessions_run"] >= 1
    d = build_dashboard()
    assert d["total_evaluated_purchases"] >= before + out["sessions_run"]
    assert d["irreversibility_distribution"]                      # irreversibility panel has data
    assert set(d["bias_flags"]) >= {"high_position_bias_flagged", "counterfactual_divergence_flagged"}
