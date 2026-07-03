# CEBRA Multimodal Pipeline

This project prepares probe and calcium recordings as independent CEBRA
sessions, trains them jointly with shared labels, and saves modality-specific
embeddings in one learned latent space.

The intended setup follows the CEBRA cross-modality use case: Neuropixels and
two-photon calcium recordings do not need to be simultaneous or from the same
animal. They should share a sampling variable, such as DINO features extracted
from the same repeated video, behavioural variables, or time labels.

## Expected Data Layout

Each subject/session directory can contain one or more modality files:

```text
data/preprocessed/{subject_id}/
  probe.npy       # optional, shape: (time, probe_features)
  calcium.npy     # optional, shape: (time, calcium_features)
  continuous_labels.npy  # recommended, shape: (time, label_features)
  labels.npy             # accepted alias for continuous_labels.npy
  discrete_labels.npy    # optional, shape: (time,)
  timestamps.npy  # optional, shape: (time,)
  metadata.json   # optional
```

`probe.npy` and `calcium.npy` are loaded as separate sessions. They are not
merged into one feature vector. For calcium movies, extract traces or image
features first and save them as `calcium.npy`.

For CEBRA-Behaviour with video, save DINO frame features as
`continuous_labels.npy`. If you also have class-like behavioural variables,
save them as `discrete_labels.npy`; the training scripts will call CEBRA in the
corresponding form: `fit(s)`, `fit(s, c)`, `fit(s, k)`, or `fit(s, c, k)`.
For multi-session training, the scripts pass lists such as
`fit([s_np, s_2p], [c_np, c_2p])`.

With `cebra==0.4.0`, mixed continuous+discrete labels are supported for a
single session, but not for multi-session training. For the cross-modality
joint model, use either shared continuous labels such as DINO features or shared
discrete labels, not both at the same time.

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
python scripts/train_unimodal.py --config configs/default_config.yaml --modality calcium --labels auto
python scripts/train_joint.py --config configs/default_config.yaml --labels auto
python scripts/embed_subject.py --config configs/default_config.yaml --subject-id NP_MOUSE_001
```

The unimodal training commands are optional. They are useful for comparing
probe-only, calcium-only, and cross-modality joint latent spaces.

The main outputs are:

```text
models/checkpoints/joint_cebra_model.pt
models/checkpoints/joint_training_sessions.json
data/embeddings/{subject_id}/probe_embedding.npz
data/embeddings/{subject_id}/calcium_embedding.npz
```

`joint_training_sessions.json` records the order of sessions passed to CEBRA.
This is important when transforming a specific modality back through a trained
multi-session model.

## Manifest Option

If the raw Drive layout is not simply `{drive_base_path}/raw/{subject_id}/`, add
a manifest at `data/manifests/default_manifest.csv` with columns:

```text
subject_id,modality,path,target_name
```

Relative `path` values are resolved under `paths.drive_base_path`.
