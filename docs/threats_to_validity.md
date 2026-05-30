# Threats to Validity — BugInsight AI

This document tracks all identified threats to validity for the BugInsight AI
research program. It is maintained from Day 1 and updated as new threats are
identified during experimentation.

---

## Internal Validity

Threats related to our experimental design and methodology.

### 1. Dataset Leakage
- **Threat**: Overlap between training and test sets could artificially inflate
  metrics.
- **Mitigation**: Stratified splitting is performed once per seed with strict
  separation.  No bug report appears in more than one split.

### 2. Hyperparameter Bias
- **Threat**: Hyperparameters are tuned to a specific dataset, producing results
  that do not generalise.
- **Mitigation**: All hyperparameters are declared in `configs/config.yaml` and
  kept constant across experiments unless explicitly stated. Future phases will
  use NAS to automate this.

### 3. Randomness
- **Threat**: Single-run results may be due to favourable random initialisations.
- **Mitigation**: All experiments are run across 5 random seeds (42, 123, 456,
  789, 999). We report Mean ± Std and apply paired t-tests / Wilcoxon
  signed-rank tests for statistical significance.

### 4. Class Imbalance
- **Threat**: Severity distributions are often skewed (many Minor/Trivial, few
  Critical), potentially biasing classifiers toward majority classes.
- **Mitigation**: We report macro-averaged metrics (Precision, Recall, F1) that
  weight all classes equally. Classical baselines use `class_weight="balanced"`.

---

## External Validity

Threats related to generalisability of our findings.

### 1. Language Bias
- **Threat**: Datasets may be dominated by Java projects (e.g., Defects4J),
  limiting claims about language-agnostic performance.
- **Mitigation**: Acknowledged as a limitation. Phase 9 Cross-Project Validation
  will evaluate on multiple software ecosystems.

### 2. Bug Tracker Bias
- **Threat**: Results may not transfer to bug trackers with different reporting
  conventions (e.g., GitHub Issues vs. Bugzilla vs. Jira).
- **Mitigation**: The dataset abstraction layer supports multiple formats. We
  will test on datasets from different trackers in Phase 9.

### 3. Temporal Bias
- **Threat**: Training on all historical data ignores temporal ordering. In
  practice, models would only have access to past bugs.
- **Mitigation**: Noted for future work. A temporal split experiment can be added
  to Phase 9.

---

## Construct Validity

Threats related to whether our measurements capture the intended concepts.

### 1. Severity Label Reliability
- **Threat**: Severity labels are assigned by human reporters and may be
  inconsistent, subjective, or incorrect.
- **Mitigation**: Acknowledged. This is a fundamental limitation of all
  severity prediction research. We normalise labels through a consistent
  mapping (documented in `config.yaml`).

### 2. Metric Selection
- **Threat**: F1-score may not fully capture real-world usefulness to developers.
- **Mitigation**: We report multiple metrics (Accuracy, Precision, Recall, F1)
  and generate confusion matrices for qualitative analysis.  Phase 3
  (Explainability) will add human evaluation.

### 3. Explainability Faithfulness
- **Threat**: Attention weights may not faithfully represent model reasoning.
- **Mitigation**: Phase 3 will include faithfulness metrics (token removal
  experiments) to quantify the reliability of explanations.

---

## Revision Log

| Date | Phase | Change |
|:-----|:------|:-------|
| 2026-05-30 | Phase 1 | Initial document created |
