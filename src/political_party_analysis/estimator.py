from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


class DensityEstimator:
    """Class to estimate Density/Distribution of the given data.
    1. Write a function to model the distribution of the political party dataset
    2. Write a function to randomly sample 10 parties from this distribution
    3. Map the randomly sampled 10 parties back to the original higher dimensional
    space as per the previously used dimensionality reduction technique.
    """

    def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names):
        self.data = data
        self.dim_reducer_model = dim_reducer.model
        self.feature_names = high_dim_feature_names

    # Alternative: KernelDensity from sklearn.neighbors for non-parametric estimation
    def fit_distribution(self, n_components: int = 3, random_state: int = 0) -> GaussianMixture:
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
        )
        model.fit(self.data.values)
        self.gmm_ = model
        return model

    def sample_parties(
        self, n_samples: int = 10, random_state: int = 0
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        if not hasattr(self, "gmm_"):
            raise RuntimeError("Call fit_distribution() before sampling")
        rng = np.random.RandomState(random_state)
        samples, labels = self.gmm_.sample(n_samples)
        sampled_df = pd.DataFrame(samples, columns=self.data.columns)
        return sampled_df, labels

    def invert_to_high_dim(self, low_dim_df: pd.DataFrame) -> pd.DataFrame:
        # Inverse transform back to original feature space if supported by the reducer
        if hasattr(self.dim_reducer_model, "inverse_transform"):
            high_dim = self.dim_reducer_model.inverse_transform(low_dim_df.values)
            return pd.DataFrame(high_dim, columns=self.feature_names)
        # Alternative: train a regressor per high-dim feature to map from low-dim → high-dim
        raise NotImplementedError("Dimensionality reducer does not support inverse_transform")
