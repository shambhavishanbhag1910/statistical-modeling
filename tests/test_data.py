from failure_risk.data import generate_synthetic_fleet, validate_dataset


def test_synthetic_data_is_reproducible_and_valid():
    df1, meta1 = generate_synthetic_fleet(n_assets=500, seed=123)
    df2, meta2 = generate_synthetic_fleet(n_assets=500, seed=123)
    validate_dataset(df1)
    assert df1.equals(df2)
    assert meta1["event_rate"] == meta2["event_rate"]
    assert 0.15 < df1["event"].mean() < 0.80
