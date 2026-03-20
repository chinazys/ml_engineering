# ML Engineering Labs

Minimal repository layout for the current Lab 1, based on assignment requirements.

## Repository Structure
- `docs/requirements/` - assignment PDFs.
- `labs/lab01/notebooks/` - Lab 1 notebook.
- `labs/lab01/configs/` - Lab 1 config files.
- `labs/lab01/src/` - Lab 1 reusable Python code.
- `labs/lab01/artifacts/` - saved models, logs, metrics.
- `labs/lab01/report/` - Lab 1 report.

## Lab 1 Requirements Checklist
- Choose dataset and split into train/validation/test.
- Implement full training pipeline: ingestion, training loop, evaluation, artifact collection.
- Evaluate with multiple metrics (accuracy, precision, recall, F1).
- Use configuration files (not hardcoded values).
- Use logging module instead of print for pipeline events.
- Use Poetry and code quality tools (mypy, ruff, black, isort).
- Keep clean commit history and include lab report.

## Current Lab 1 Assets
- Notebook: `labs/lab01/notebooks/ml_engineering_lab_01.ipynb`
- Requirements PDF: `docs/requirements/lab01_requirements.pdf`
