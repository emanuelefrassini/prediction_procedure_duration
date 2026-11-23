# Result-analysis tools

This directory contains the **original research analysis scripts**, retained with their historical assumptions and file paths.

| Script | Analysis |
|---|---|
| `read_results.py` | Aggregate model errors and training/inference times from trial folders |
| `calculate_ensemble.py` | Combine prediction files from different architectures |
| `plots_paper.py` | Generate comparative figures from Excel summary sheets |
| `accuracy_from_txt.py` | Re-read prediction text files and compute error/accuracy summaries |
| `scaling_predictions.py` | Explore rescaling of saved predictions |
| `besteld_time.py` | Analyze a separate spreadsheet of procedure booking/arrival times |

Several scripts contain fixed Windows paths and require packages such as pandas, matplotlib or seaborn beyond the primary model dependencies. Update local paths before running them. These exploratory tools are not executed by the repository's lightweight CI tests.

For a portable, dependency-free summary of the standard `PAPER_ETC.py` results, use [`../scripts/summarize_results.py`](../scripts/summarize_results.py), documented in [`../docs/results.md`](../docs/results.md).
