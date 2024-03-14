import torch
import numpy as np
from scipy.optimize import brute
import math


def calibrate_llpr_params(model, validation_loader, function="ssl", **kwargs):
    # This function optimizes the calibration parameters for LLPR on the validation set
    # Original author: F. Bigi (@frostedoyster) <https://github.com/frostedoyster/llpr>

    if function == "ssl":
        obj_function = _sum_squared_log
    elif function == "nll":
        obj_function = _avg_nll_regression
    else:
        raise RuntimeError("Unsupported objective function type for LLPR uncertainty calibration!")

    def obj_function_wrapper(x):
        x = _process_inputs(x)
        try:
            model.compute_inv_covariance(*x)
            obj_function_value = obj_function(model, validation_loader, **kwargs)
        except torch._C._LinAlgError:
            obj_function_value = 1e10
        # HACK:
        if math.isnan(obj_function_value):
            obj_function_value = 1e10
        return obj_function_value
    result = brute(obj_function_wrapper, ranges=[slice(-5, 5, 0.25), slice(-5, 5, 0.25)])

    # warn if we hit the edge of the parameter space
    if result[0] == -5 or result[0] == 5 or result[1] == -5 or result[1] == 5:
        raise Warning("Optimal parameters are found at the edge of the parameter space")

    print(f"Calibrated LLPR parameters:\tC = {10**result[0]:.4E}\tsigma = {10**result[1]:.4E}")
    model.compute_inv_covariance(*(_process_inputs(result)))


def _process_inputs(x):
    x = list(x)
    x = [10**single_x for single_x in x]
    return x


def _avg_nll_regression(model, dataloader, energy_shift=0.0, energy_scale=1.0):
    # This function calculates the negative log-likelihood on the energy for a dataset
    # Original author: F. Bigi (@frostedoyster) <https://github.com/frostedoyster/llpr>

    total_nll = 0.0
    total_datapoints = 0
    for batch in dataloader:
        batch_dict = batch.to_dict()
        y = batch_dict['energy']  # * energy_scale + energy_shift
        model_outputs = model(batch_dict)
        predictions = model_outputs['energy']  # * energy_scale + energy_shift
        estimated_variances = model_outputs['uncertainty']  # * energy_scale**2
        total_datapoints += len(y)
        total_nll += (
            (y-predictions)**2 / estimated_variances + torch.log(estimated_variances) + np.log(2*np.pi)
        ).sum().item() * 0.5

    return total_nll / total_datapoints


def _sum_squared_log(model, dataloader, n_samples_per_bin=1):
    # This function calculates the sum of squared log errors on the energy for a dataset
    # Original author: F. Bigi (@frostedoyster) <https://github.com/frostedoyster/llpr>

    actual_errors = []
    predicted_errors = []
    for batch in dataloader:
        batch_dict = batch.to_dict()
        y = batch_dict['energy']
        model_outputs = model(batch_dict)
        predictions = model_outputs['energy']
        estimated_variances = model_outputs['uncertainty']
        actual_errors.append((y-predictions)**2)
        predicted_errors.append(estimated_variances)

    actual_errors = torch.cat(actual_errors).flatten()
    predicted_errors = torch.cat(predicted_errors).flatten()
    sort_indices = torch.argsort(predicted_errors)
    actual_errors = actual_errors[sort_indices]
    predicted_errors = predicted_errors[sort_indices]

    n_samples = len(actual_errors)

    actual_error_bins = []
    predicted_error_bins = []

    # skip the last bin for incompleteness
    for i_bin in range(n_samples // n_samples_per_bin - 1):
        actual_error_bins.append(
            actual_errors[i_bin*n_samples_per_bin:(i_bin+1)*n_samples_per_bin]
        )
        predicted_error_bins.append(
            predicted_errors[i_bin*n_samples_per_bin:(i_bin+1)*n_samples_per_bin]
        )

    actual_error_bins = torch.stack(actual_error_bins)
    predicted_error_bins = torch.stack(predicted_error_bins)

    # calculate means:
    actual_error_means = actual_error_bins.mean(dim=1)
    predicted_error_means = predicted_error_bins.mean(dim=1)

    # calculate squared log errors:
    squared_log_errors = (
        torch.log(actual_error_means / predicted_error_means)**2
    )

    # calculate the sum of squared log errors:
    sum_squared_log = squared_log_errors.sum().item()

    return sum_squared_log
