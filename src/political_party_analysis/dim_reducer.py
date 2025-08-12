from typing import Literal, Optional, Union

import pandas as pd
from sklearn.decomposition import PCA


class DimensionalityReducer:
    """Class to model a dimensionality reduction method for the given dataset.
    1. Write a function to convert the high dimensional data to 2 dimensional.
    """

    def __init__(
        self,
        method: Union[Literal["PCA"], str],
        data: pd.DataFrame,
        n_components: int = 2,
        random_state: Optional[int] = 0,
    ):
        # Alternative: support "TSNE" or "UMAP" methods for non-linear embeddings
        self.method = method.upper()
        self.n_components = n_components
        self.data = data
        self.model = None
        if self.method == "PCA":
            self.model = PCA(n_components=self.n_components, random_state=random_state)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")

    ##### YOUR CODE GOES HERE #####
    def transform(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Dimensionality reduction model is not initialized")
        reduced = self.model.fit_transform(self.data)
        columns = ["component_1", "component_2"][: self.n_components]
        transformed_df = pd.DataFrame(reduced, index=self.data.index, columns=columns)
        return transformed_df
