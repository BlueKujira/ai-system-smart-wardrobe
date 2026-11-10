from pathlib import Path
import json, joblib, mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from contracts import RawWardrobeSchema

RAW = Path("data/raw/data_cnn.csv")
OUT = Path("data/processed")
ART = Path("artifacts")
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

def main():
    targets = ["gender", "masterCategory", "subCategory","articleType", "baseColour", "season", "usage"]

    raw = pd.read_csv(RAW)
    validated = RawWardrobeSchema.validate(raw)
    clean = validated.drop_duplicates().dropna(subset=targets).copy()

    mappings = {}
    for attr in targets:
        unique = sorted(clean[attr].unique())
        mappings[attr] = {v: i for i, v in enumerate(unique)}

    for attr in targets: clean[attr] = clean[attr].map(mappings[attr])

    train_df, test_df = train_test_split(clean, test_size=0.2, random_state=42)

    cols = ["id"] + targets
    train_out = train_df[cols].reset_index(drop=True)
    test_out = test_df[cols].reset_index(drop=True)

    for col in targets:
        train_out[col] = train_out[col].astype(int)
        test_out[col] = test_out[col].astype(int)

    train_out.to_parquet(OUT / "train.parquet", index=False)
    test_out.to_parquet(OUT / "test.parquet", index=False)

    preprocessor = {
        "mappings": mappings,
        "targets": targets,
        "image_size": 96,
    }
    joblib.dump(preprocessor, ART / "preprocessor.joblib")

    mappings_path = Path("model/mappings.json") 
    mappings_path.parent.mkdir(parents=True, exist_ok=True)
    mappings_path.write_text(json.dumps(mappings, ensure_ascii=False, indent=2))

    report = {
        "raw_rows": len(raw), 
        "clean_rows": len(clean),
        "train_rows": len(train_out), 
        "test_rows": len(test_out),
        "dropped_rows": len(raw) - len(clean)
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2))

    mlflow.set_experiment("lab4-data-pipeline")
    with mlflow.start_run():
        mlflow.log_params({"split_strategy": "random", "test_size": 0.2})
        mlflow.log_metrics(report)
        mlflow.log_artifact(str(OUT / "validation_report.json"))
        mlflow.log_artifact(str(ART / "preprocessor.joblib"))

if __name__ == "__main__":
    main()