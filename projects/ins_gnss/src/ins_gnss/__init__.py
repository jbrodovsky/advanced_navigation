"""Starter package for the INS/GNSS integration project (Project 3)."""

from .data_paths import DatasetNotFoundError, find_dataset_dir
from .datasets import (
    BASELINE_TIME_OFFSET_S,
    TRIAL_NOTES,
    default_mems_root,
    estimate_mount_rotation,
    list_mems_nav_trials,
    load_ins_gnss_data,
    load_mems_nav_trial,
    recommended_trials,
    repeated_fix_mask,
    rotate_imu_samples,
)
from .eskf import (
    INSGNSSFilter,
    discrete_process_noise,
    error_state_jacobian,
    gnss_position_residual,
    inject_error_state,
    joseph_update,
    reset_covariance,
)
from .strapdown import coarse_alignment, free_inertial, mechanize
from .types import GNSSSample, IMUNoise, IMUSample, MEMSNavTrial, MEMSReferenceState, NavState

__all__ = [
    "BASELINE_TIME_OFFSET_S",
    "DatasetNotFoundError",
    "GNSSSample",
    "IMUNoise",
    "IMUSample",
    "INSGNSSFilter",
    "MEMSNavTrial",
    "MEMSReferenceState",
    "NavState",
    "TRIAL_NOTES",
    "coarse_alignment",
    "default_mems_root",
    "discrete_process_noise",
    "error_state_jacobian",
    "estimate_mount_rotation",
    "find_dataset_dir",
    "free_inertial",
    "gnss_position_residual",
    "inject_error_state",
    "joseph_update",
    "list_mems_nav_trials",
    "load_ins_gnss_data",
    "load_mems_nav_trial",
    "mechanize",
    "recommended_trials",
    "repeated_fix_mask",
    "reset_covariance",
    "rotate_imu_samples",
]
