"""Predictive-Machine Guard Applied ML & Deep Learning Engine package."""
from .ai_recommender import AIMaintenanceRecommender
from .datasets import (
    DATASET_CATALOG,
    ensure_all_12_datasets_generated,
    generate_or_load_ai4i2020,
    generate_or_load_cmapss_turbofan,
    generate_or_load_custom_plant_telemetry,
    generate_or_load_domain_dataset,
    normalize_custom_dataframe,
)
from .deep_learning import DeepLearningClassifierWrapper
from .pipeline import PredictiveMaintenanceEngine
from .preprocessing import DataQualityPreprocessor

__all__ = [
    "AIMaintenanceRecommender",
    "DATASET_CATALOG",
    "DataQualityPreprocessor",
    "DeepLearningClassifierWrapper",
    "PredictiveMaintenanceEngine",
    "ensure_all_12_datasets_generated",
    "generate_or_load_ai4i2020",
    "generate_or_load_cmapss_turbofan",
    "generate_or_load_custom_plant_telemetry",
    "generate_or_load_domain_dataset",
    "normalize_custom_dataframe",
]

