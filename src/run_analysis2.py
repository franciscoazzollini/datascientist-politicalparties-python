from pathlib import Path

from matplotlib import pyplot

from political_party_analysis.loader import DataLoader
from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.estimator import DensityEstimator
from political_party_analysis.visualization import (
    plot_density_estimation_results,
    plot_finnish_parties,
    scatter_plot,
)

if __name__ == "__main__":

    data_loader = DataLoader()
    # Data pre-processing step
    processed_df = data_loader.preprocess_data()

    # Dimensionality reduction step
    dim_reducer = DimensionalityReducer("PCA", processed_df, n_components=2)
    reduced_dim_data = dim_reducer.transform()

    # Uncomment this snippet to plot dim reduced data
    pyplot.figure()
    splot = pyplot.subplot()
    scatter_plot(
        reduced_dim_data,
        color="r",
        splot=splot,
        label="dim reduced data",
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "dim_reduced_data.png"]))

    # Density estimation/distribution modelling step
    estimator = DensityEstimator(reduced_dim_data, dim_reducer, processed_df.columns)
    gmm = estimator.fit_distribution(n_components=3)
    sampled_df, labels = estimator.sample_parties(n_samples=10)
    # Alternative: use KernelDensity for a non-parametric density estimate

    # Plot density estimation results here
    plot_density_estimation_results(
        reduced_dim_data,
        gmm.predict(reduced_dim_data),
        gmm.means_,
        gmm.covariances_,
        title="GMM on PCA(2) of parties",
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "density_estimation.png"]))

    # Plot left and right wing parties here
    pyplot.figure()
    splot = pyplot.subplot()
    # Plot left- and right-wing based on PC1 as a proxy (negative=left, positive=right)
    left_mask = reduced_dim_data.iloc[:, 0] < 0
    right_mask = ~left_mask
    scatter_plot(reduced_dim_data[left_mask], color="r", splot=splot, label="left")
    scatter_plot(reduced_dim_data[right_mask], color="b", splot=splot, label="right")
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "left_right_parties.png"]))
    pyplot.title("Lefty/righty parties")

    # Plot finnish parties here
    plot_finnish_parties(reduced_dim_data)

    print("Analysis Complete")
