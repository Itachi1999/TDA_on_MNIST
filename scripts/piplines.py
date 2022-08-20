from sklearn.pipeline import Pipeline, make_pipeline, make_union
import numpy as np 
from gtda.images import Binarizer
from gtda.images import RadialFiltration
from gtda.images import HeightFiltration
from gtda.homology import CubicalPersistence
#from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Scaler
from gtda.diagrams import Amplitude
from gtda.diagrams import PersistenceEntropy

def initial_pipeline(binarizer_threshold = 0.4, center = np.array([20, 6]), 
    metric_params = {'sigma' : 0.15, 'n_bins' : 60}):
    
    steps = [
    ("binarizer", Binarizer(threshold= binarizer_threshold)),
    ("filtration", RadialFiltration()),
    ("diagram", CubicalPersistence()),
    ("rescale", Scaler()),
    ("amplitude", Amplitude(metric="heat", metric_params= metric_params))
    ]

    heat_pipeline = Pipeline(steps)
    return heat_pipeline


def full_pipeline():
    direction_list = [[1, 0], [1, 1], [0, 1],
                      [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]


    center_list = [
        [13, 6],
        [6, 13],
        [13, 13],
        [20, 13],
        [13, 20],
        [6, 6],
        [6, 20],
        [20, 6],
        [20, 20],
    ]

    # Creating a list of all filtration transformer, we will be applying
    filtration_list = (
        [
            HeightFiltration(direction=np.array(direction), n_jobs=-1)
            for direction in direction_list
        ]
        + [RadialFiltration(center=np.array(center), n_jobs=-1)
        for center in center_list]
    )

# Creating the diagram generation pipeline
    diagram_steps = [
        [
            Binarizer(threshold=0.4, n_jobs=-1),
            filtration,
            CubicalPersistence(n_jobs=-1),
            Scaler(n_jobs=-1),
        ]
        for filtration in filtration_list
    ]

# Listing all metrics we want to use to extract diagram amplitudes
    metric_list = [
        {"metric": "bottleneck", "metric_params": {}},
        {"metric": "wasserstein", "metric_params": {"p": 1}},
        {"metric": "wasserstein", "metric_params": {"p": 2}},
        {"metric": "landscape", "metric_params": {
            "p": 1, "n_layers": 1, "n_bins": 100}},
        {"metric": "landscape", "metric_params": {
            "p": 1, "n_layers": 2, "n_bins": 100}},
        {"metric": "landscape", "metric_params": {
            "p": 2, "n_layers": 1, "n_bins": 100}},
        {"metric": "landscape", "metric_params": {
            "p": 2, "n_layers": 2, "n_bins": 100}},
        {"metric": "betti", "metric_params": {"p": 1, "n_bins": 100}},
        {"metric": "betti", "metric_params": {"p": 2, "n_bins": 100}},
        {"metric": "heat", "metric_params": {"p": 1, "sigma": 1.6, "n_bins": 100}},
        {"metric": "heat", "metric_params": {"p": 1, "sigma": 3.2, "n_bins": 100}},
        {"metric": "heat", "metric_params": {"p": 2, "sigma": 1.6, "n_bins": 100}},
        {"metric": "heat", "metric_params": {"p": 2, "sigma": 3.2, "n_bins": 100}},
    ]

    #
    feature_union = make_union(
        *[PersistenceEntropy(nan_fill_value=-1)]
        + [Amplitude(**metric, n_jobs=-1) for metric in metric_list]
    )

    tda_union = make_union(
        *[make_pipeline(*diagram_step, feature_union)
        for diagram_step in diagram_steps],
        n_jobs=-1
    )

    return tda_union
