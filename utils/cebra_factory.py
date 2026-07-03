from __future__ import annotations

from collections.abc import Sequence

from .config import configured_device


def _session_architectures(config: dict, session_metadata: Sequence[dict] | None) -> list[str] | None:
    if session_metadata is None:
        return None

    model_config = config["model"]
    default_architecture = model_config.get("architecture", "offset10-model")
    architecture_by_modality = model_config.get("architecture_by_modality", {})
    return [
        architecture_by_modality.get(metadata["modality"], default_architecture)
        for metadata in session_metadata
    ]


class ModalityAwareCEBRAMixin:
    """Mixin for CEBRA models that choose one architecture per training session."""

    def __init__(self, *args, session_architectures: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_architectures = session_architectures

    def _prepare_model(self, dataset, is_multisession: bool):
        if not is_multisession or self.session_architectures is None:
            return super()._prepare_model(dataset, is_multisession)

        if len(self.session_architectures) != dataset.num_sessions:
            raise ValueError(
                "session_architectures must match the number of CEBRA sessions: "
                f"got {len(self.session_architectures)} architectures for "
                f"{dataset.num_sessions} sessions."
            )

        import torch.nn as nn
        import cebra

        return nn.ModuleList(
            [
                cebra.models.init(
                    architecture,
                    num_neurons=session.input_dimension,
                    num_units=self.num_hidden_units,
                    num_output=self.output_dimension,
                )
                for architecture, session in zip(
                    self.session_architectures, dataset.iter_sessions()
                )
            ]
        ).to(self.device_)


def _modality_aware_cebra_class():
    import cebra

    class ModalityAwareCEBRAEstimator(ModalityAwareCEBRAMixin, cebra.CEBRA):
        pass

    ModalityAwareCEBRAEstimator.__module__ = __name__
    ModalityAwareCEBRAEstimator.__qualname__ = "ModalityAwareCEBRAEstimator"
    globals()["ModalityAwareCEBRAEstimator"] = ModalityAwareCEBRAEstimator
    return ModalityAwareCEBRAEstimator


def build_cebra_model(config: dict, session_metadata: Sequence[dict] | None = None):
    import cebra

    model_config = config["model"]
    training_config = config["training"]
    conditional = training_config.get("conditional", "time_delta")
    if conditional == "behavior":
        conditional = "time_delta"
    session_architectures = _session_architectures(config, session_metadata)
    cebra_kwargs = {
        "model_architecture": model_config.get("architecture", "offset10-model"),
        "output_dimension": model_config["output_dimension"],
        "num_hidden_units": model_config.get("hidden_dim", 32),
        "batch_size": training_config["batch_size"],
        "learning_rate": training_config["learning_rate"],
        "max_iterations": training_config["max_iterations"],
        "temperature": model_config.get("temperature", 1.0),
        "distance": model_config.get("distance", "cosine"),
        "conditional": conditional,
        "device": configured_device(config),
    }
    if session_architectures is None:
        return cebra.CEBRA(**cebra_kwargs)

    return _modality_aware_cebra_class()(
        **cebra_kwargs,
        session_architectures=session_architectures,
    )


def fit_cebra_model(
    model,
    feature_sessions: list,
    continuous_label_sessions: list | None = None,
    discrete_label_sessions: list | None = None,
) -> None:
    """Fit CEBRA with API-compatible single- or multi-session arguments."""
    has_continuous = continuous_label_sessions is not None
    has_discrete = discrete_label_sessions is not None
    is_single_session = len(feature_sessions) == 1

    if is_single_session:
        features = feature_sessions[0]
        continuous = None if not has_continuous else continuous_label_sessions[0]
        discrete = None if not has_discrete else discrete_label_sessions[0]
        if continuous is not None and discrete is not None:
            model.fit(features, continuous, discrete)
        elif continuous is not None:
            model.fit(features, continuous)
        elif discrete is not None:
            model.fit(features, discrete)
        else:
            model.fit(features)
        return

    if not has_continuous and not has_discrete:
        raise ValueError(
            "CEBRA multi-session/joint training needs labels so sessions can be "
            "aligned. Provide continuous_labels.npy/labels.npy or "
            "discrete_labels.npy for every modality session, or train a single "
            "session with --labels off."
        )
    if has_continuous and has_discrete:
        raise NotImplementedError(
            "CEBRA 0.4.0 does not support mixed continuous+discrete labels "
            "for multi-session training. Use continuous_labels.npy for the "
            "cross-modality joint model, or remove continuous labels to train "
            "with discrete labels only."
        )
    if has_continuous:
        model.fit(feature_sessions, continuous_label_sessions)
    elif has_discrete:
        model.fit(feature_sessions, discrete_label_sessions)
    else:
        model.fit(feature_sessions)


def load_cebra_model(model_path):
    """Load models saved by this project, including modality-aware torch saves."""
    import cebra

    _modality_aware_cebra_class()
    try:
        return cebra.CEBRA.load(model_path)
    except Exception as exc:
        message = str(exc)
        if "Weights only load failed" in message:
            return cebra.CEBRA.load(model_path, weights_only=False)
        raise
