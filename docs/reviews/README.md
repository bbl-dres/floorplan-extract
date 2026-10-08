# Reviews

Technical reviews of the design documents and pilots: findings, recommendations and the evidence behind them. The newest review comes first.

| Date | Review | Scope |
|---|---|---|
| 2026-10-08 | [Second review of pilot v2: heads, retraining, merged evaluation code, exports](2026-10-08-pilot-v2-second-review.md) | The v2 boundary and interior heads in the room stage (CVC-FP rooms 0.64/0.71 → 0.73/0.77); renderer 3.0 and four training runs on RunPod; the `fpeval` package and a golden test; OCR once per sheet; Excel and IFC 4.3 exports; the workflow-app hook |
| 2026-10-07 | [Code review of pilot v2](2026-10-07-pilot-v2-code-review.md) | [pilot/v2-pipeline](../../pilot/v2-pipeline/README.md) against [motivation-goals.md](../motivation-goals.md): accuracy, robustness, performance and maintainability of every stage, the decision on the [centre-line report](../reports/2026-10-07-centre-line-bim-reconstruction.md); 30 recommendations implemented and measured on the public harness the same evening |
| 2026-10-07 | [Pipeline as canonical target: inputs, masking and scale](2026-10-07-pipeline-target-inputs-masking-scale.md) | [docs/pipeline.md](../pipeline.md) draft 0.2 against arbitrary uploads (JPG, PNG, TIFF, PDF, DWG): input normalisation, sheet layout and masking, scale per drawing, other sheet features, drawing types; implemented in draft 0.3 |
| 2026-10-07 | [Pipeline design and pilot v2](2026-10-07-pipeline-and-pilot-v2.md) | [docs/pipeline.md](../pipeline.md) draft 0.2, [pilot/v2-pipeline](../../pilot/v2-pipeline/README.md); CubiCasa5K zero-shot benchmark; findings of the solution and data catalogues; data needs, partner routes and legal questions |
