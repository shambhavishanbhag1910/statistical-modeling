# Architecture and Production Design

## Current portfolio implementation

```text
Synthetic dataset
      |
      v
Data validation
      |
      v
Temporal train / holdout split
      |
      v
Feature preprocessing
      |
 +----+----+
 |         |
 v         v
Cox PH   Weibull AFT
 |         |
 +----+----+
      |
Evaluation and model selection
      |
      v
model_bundle.pkl
   /        \
  v          v
FastAPI    Batch scorer
  |
Prometheus metrics
```

## Production target

```text
Fleet / warranty source systems
            |
            v
Governed ingestion pipeline
            |
            v
Validated feature tables
       /           \
      v             v
Training         Daily scoring
      |             |
Model registry      v
      |        Fleet risk table
      v             |
Approved model      v
      |        Maintenance workflow
      v
Online API
      |
Observability / drift / calibration
      |
Outcome feedback and retraining
```

## Reliability controls to add for an enterprise rollout

- schema validation and quarantine of invalid data
- model artifact checksum and version pinning
- authentication and authorization
- timeout, retry and circuit-breaker policies
- request and prediction audit IDs
- data drift alarms
- latency and error SLOs
- release approval before model promotion
- champion/challenger evaluation
- rollback to previous model version
- post-deployment calibration checks
