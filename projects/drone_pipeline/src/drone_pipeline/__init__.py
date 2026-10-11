"""Starter interfaces for the drone visual-inertial navigation project (Project 1)."""

from .data_paths import DatasetNotFoundError, find_dataset_dir
from .datasets import (
    ParameterFileError,
    default_dataset_path,
    default_drone_parameters,
    default_parameters_path,
    drone_data_dir,
    interpolate_vicon_truth,
    load_drone_dataset,
    load_drone_parameters,
    parse_drone_parameters,
)
from .filters import EKFNavigator, ParticleFilterNavigator, UKFNavigator
from .mechanization import PropagationParams, propagate_state
from .observation import (
    build_apriltag_corner_map,
    estimate_measurement_covariance,
    estimate_pose,
)
from .types import DroneDataset, DronePacket, DroneParameters, PoseObservation, TruthSample

__all__ = [
    "DatasetNotFoundError",
    "DroneDataset",
    "DronePacket",
    "DroneParameters",
    "EKFNavigator",
    "ParameterFileError",
    "ParticleFilterNavigator",
    "PoseObservation",
    "PropagationParams",
    "TruthSample",
    "UKFNavigator",
    "build_apriltag_corner_map",
    "default_dataset_path",
    "default_drone_parameters",
    "default_parameters_path",
    "drone_data_dir",
    "estimate_measurement_covariance",
    "estimate_pose",
    "find_dataset_dir",
    "interpolate_vicon_truth",
    "load_drone_dataset",
    "load_drone_parameters",
    "parse_drone_parameters",
    "propagate_state",
]
