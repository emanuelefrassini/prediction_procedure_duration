# Corrected ETC evaluation protocol

This new runner uses the **full archived TensorFlow/Keras architectures**
from `utils/models.py` (not the smaller, independent `modern_etc` models) while
correcting training/validation/test isolation, masked-SMAPE training, the trial schedule and
ensemble definition, and unshuffled procedure batches. It does **not** establish that the 2025 publication's
reported results were generated using these exact settings. The original
`models/PAPER_ETC.py` is retained for historical provenance.

Full methodology and instructions: [corrected protocol guide](../docs/corrected-protocol.md).
An explicit response to the original repository criticisms is available in the
[paper-alignment checklist](../docs/paper-alignment-checklist.md).

Preview without clinical data or TensorFlow:

```bash
python -m corrected_protocol
python -m unittest discover -s tests -p 'test_*.py' -v
```

Train only after establishing a working Python 3.10 + TensorFlow 2.10 environment.
The default full run is intentionally a dry run; use `--execute` to access the
anonymized public data and train models. A first smoke run is:

```bash
python -m corrected_protocol --execute --trials 1 --epochs 2 \
  --intervals 60 --models inceptiontime lstm_fcn
```
