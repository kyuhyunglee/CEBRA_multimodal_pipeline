# CEBRA IBL Probe Pipeline

This project prepares IBL ONE API cache data for CEBRA. The current default
target is one lab folder:

```text
/content/drive/MyDrive/cebra_data/cache/churchlandlab/Subjects
```

Each valid session is discovered from:

```text
{subject}/{date}/{number}/alf/
  probe00/pykilosort/*/spikes.times.npy
  probe00/pykilosort/*/spikes.clusters.npy
  */_ibl_trials.table.pqt
  */_ibl_trials.stimOnTrigger_times.npy
```

`sampler.py` bins spikes into `probe.npy` and builds CEBRA behaviour labels for
the decision interval from stimulus onset to movement/response/feedback.

## Standardized Output

Running preparation creates one directory per subject/session:

```text
/content/drive/MyDrive/cebra_data/preprocessed/{subject}__{date}__{number}/
  probe.npy              # binned spike counts, shape: (time, clusters)
  continuous_labels.npy  # decision progress + trial variables
  discrete_labels.npy    # choice/reward class per bin
  timestamps.npy
  cluster_ids.npy
  metadata.json
```

`/content/drive/MyDrive/cebra_data/preprocessed/ibl_sessions.json` records all
sessions that were parsed. The config uses `target_subjects: auto`, so
validation/training scripts consume that summary automatically.

By default, completed sessions are skipped on reruns (`preprocessing.skip_existing: true`).
This makes Colab runs resumable and keeps large preprocessed arrays on Google
Drive instead of the temporary Colab disk.

## Colab Flow

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
pip install -r requirements.txt
python scripts/prepare_data.py --config configs/default_config.yaml
python scripts/validate_preprocessed.py --config configs/default_config.yaml
python scripts/train_unimodal.py --config configs/default_config.yaml --modality probe --labels auto
python scripts/train_joint.py --config configs/default_config.yaml --labels auto
python scripts/embed_subject.py --config configs/default_config.yaml --modality probe
```

For this IBL-only stage, `train_joint.py` means multi-session probe training
across all discovered Churchland sessions. Later, calcium sessions can be added
as another modality without concatenating features.

## Multi-Modality Training Wrappers

`scripts/train_discovery.py` and `scripts/train_hypothesis.py` are thin
wrappers around `scripts/train_joint.py` for multi-modality training runs. They
use `configs/default_config.yaml` and are intended for cases where the config
contains at least two modalities, such as `probe` and `calcium`.

- `train_discovery.py` runs joint training without behaviour labels
  (`use_labels=False`).
- `train_hypothesis.py` runs joint training with behaviour labels
  (`use_labels=True`).
