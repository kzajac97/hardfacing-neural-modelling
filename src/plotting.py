from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_hardness(
    sample_label: str,
    data: pd.DataFrame,
    predictions: Optional[np.array] = None,
    marker_size: int = 120
) -> None:
    """
    Utility for plotting hardness data with optional predictions of a model

    :param sample_label: string label of plotted sample, example 1A1 etc.
    :param data: dataframe of measurements, must contain required columns: `Sample`, `Distance` and `Hardness`
    :param predictions: array of model predictions
    :param marker_size: scatter plot marker size
    """
    plot_data = data.loc[data["Sample"] == sample_label]
    _ = plt.scatter(plot_data["Distance"], plot_data["Hardness"], s=marker_size)

    if predictions is not None:
        _ = plt.scatter(plot_data["Distance"], predictions, s=marker_size)

    plt.grid()
    plt.legend([sample_label])
