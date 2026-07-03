from __future__ import annotations

from .config import configured_device


def build_cebra_model(config: dict):
    import cebra

    model_config = config["model"]
    training_config = config["training"]
    return cebra.CEBRA(
        model_architecture=model_config.get("architecture", "offset10-model"),
        output_dimension=model_config["output_dimension"],
        batch_size=training_config["batch_size"],
        learning_rate=training_config["learning_rate"],
        max_iterations=training_config["max_iterations"],
        temperature=model_config.get("temperature", 1.0),
        distance=model_config.get("distance", "cosine"),
        device=configured_device(config),
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
