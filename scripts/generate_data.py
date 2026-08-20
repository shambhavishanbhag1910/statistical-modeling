from failure_risk.config import DEFAULT_DATA_PATH
from failure_risk.data import generate_synthetic_fleet, save_dataset


def main() -> None:
    df, metadata = generate_synthetic_fleet(n_assets=8_000, seed=42)
    save_dataset(df, metadata, DEFAULT_DATA_PATH)
    print(f"Generated {len(df):,} rows at {DEFAULT_DATA_PATH}")
    print(f"Event rate: {metadata['event_rate']:.1%}; censoring rate: {metadata['censoring_rate']:.1%}")


if __name__ == "__main__":
    main()
