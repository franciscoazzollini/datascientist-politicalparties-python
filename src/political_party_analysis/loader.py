from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import pandas as pd


class DataLoader:
    """Class to load the political parties dataset"""

    data_url: str = "https://www.chesdata.eu/s/CHES2019V3.dta"

    def __init__(self):
        self.party_data = self._download_data()
        self.non_features = []
        self.index = ["party_id", "party", "country"]

    def _download_data(self) -> pd.DataFrame:
        local_path = Path(__file__).parents[2].joinpath(*["data", "CHES2019V3.dta"])
        if local_path.exists():
            return pd.read_stata(local_path)
        # Alternative: ship a small CSV subset for offline tests
        data_path, _ = urlretrieve(self.data_url, local_path)
        return pd.read_stata(data_path)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to remove duplicates in a dataframe"""
        ##### YOUR CODE GOES HERE #####

        # Alternative: `df.drop_duplicates(subset=[...])` if only certain columns define uniqueness
        return df.drop_duplicates()

    def remove_nonfeature_cols(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Write a function to remove certain features cols and set certain cols as indices
        in a dataframe"""
        ##### YOUR CODE GOES HERE #####

        # Drop the provided non-feature columns if they exist
        cols_to_drop = [col for col in self.non_features if col in df.columns]
        df_clean = df.drop(columns=cols_to_drop, errors="ignore")

        # Set the specified columns as index (ignore missing gracefully)
        existing_index_cols = [col for col in self.index if col in df_clean.columns]
        if existing_index_cols:
            df_clean = df_clean.set_index(existing_index_cols)

        return df_clean

    # def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """Write a function to handle NaN values in a dataframe"""
    #     ##### YOUR CODE GOES HERE #####

    #     # Remove columns that are entirely NaN
    #     df_no_all_nan = df.dropna(axis=1, how="all")

    #     # Fill numeric columns with mean; non-numeric with mode when available
    #     numeric_cols = df_no_all_nan.select_dtypes(include=["number"]).columns
    #     non_numeric_cols = [c for c in df_no_all_nan.columns if c not in numeric_cols]

    #     df_filled = df_no_all_nan.copy()
    #     if len(numeric_cols) > 0:
    #         df_filled[numeric_cols] = df_no_all_nan[numeric_cols].apply(
    #             lambda s: s.fillna(s.mean())
    #         )

    #     for col in non_numeric_cols:
    #         if df_filled[col].isna().any():
    #             mode_vals = df_filled[col].mode(dropna=True)
    #             if not mode_vals.empty:
    #                 df_filled[col] = df_filled[col].fillna(mode_vals.iloc[0])
    #             # Alternative: drop rows with NaNs in non-numeric columns

    #     return df_filled

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        #     ##### YOUR CODE GOES HERE #####

        # Remove columns entirely NaN
        df_no_all_nan = df.dropna(axis=1, how="all").copy()

        # Fill numeric columns with mean
        for col in df_no_all_nan.select_dtypes(include="number"):
            df_no_all_nan[col] = df_no_all_nan[col].fillna(df_no_all_nan[col].mean())

        # Fill non-numeric columns with mode
        for col in df_no_all_nan.select_dtypes(exclude="number"):
            if df_no_all_nan[col].isna().any():
                mode_val = df_no_all_nan[col].mode(dropna=True)
                if not mode_val.empty:
                    df_no_all_nan[col].fillna(mode_val.iloc[0], inplace=True)

        # ALTERNATIVE
        # Drop rows with NaNs in non-numeric columns
        # non_numeric_cols = df_no_all_nan.select_dtypes(exclude="number").columns
        # df_no_all_nan = df_no_all_nan.dropna(subset=non_numeric_cols)
        return df_no_all_nan

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to normalise values in a dataframe. Use StandardScaler."""
        ##### YOUR CODE GOES HERE #####

        from sklearn.preprocessing import StandardScaler

        # Scale only numeric columns; preserve index
        numeric_cols = df.select_dtypes(include=["number"]).columns
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[numeric_cols])  # is a numpy array
        # Alternative: `sklearn.preprocessing.RobustScaler()` for outlier robustness
        scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=numeric_cols)

        # Make a copy of the original dataframe to avoid changing it directly
        df_scaled = df.copy()
        # Replace only the numeric columns with the scaled ones
        df_scaled[numeric_cols] = scaled_df

        return scaled_df

    def preprocess_data(self):
        """Write a function to combine all pre-processing steps for the dataset"""
        ##### YOUR CODE GOES HERE #####

        df = self.party_data
        df = self.remove_duplicates(df)
        df = self.remove_nonfeature_cols(df)
        df = self.handle_NaN_values(df)
        df = self.scale_features(df)
        self.party_data = df
        return self.party_data
