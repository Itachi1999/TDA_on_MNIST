import imp
import numpy as np
#import matplotlib.pyplot as plt
#from gtda.plotting import plot_heatmap
from gtda.images import Binarizer
from gtda.images import RadialFiltration
from gtda.homology import CubicalPersistence
#from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Scaler
from gtda.diagrams import HeatKernel

def index_selection(number, X_train, y_train):
    idx_train = np.flatnonzero(y_train == number)[0]
    img_train = X_train[idx_train]
    #plt.savefig('/content/drive/MyDrive/Github/TDA_on_MNIST/figures/grayscale_8.png')
    return idx_train, img_train


def Binarizing(img_train):
    binarizer_model = Binarizer(threshold= 0.4)
    img_binarized = binarizer_model.fit_transform(img_train)
    return img_binarized


def radial_filter(img_binarized):
    num_x = img_binarized.shape[0]
    num_y = img_binarized.shape[1]
    img_binarized = img_binarized.reshape(1, num_x, num_y)
    radial_model = RadialFiltration(center=np.array([20, 6]))
    img_filtered = radial_model.fit_transform(img_binarized)
    return img_filtered



def CubeComplex(img_filtered):
    cubical_model = CubicalPersistence(homology_dimensions=(0, 1),n_jobs=-1)
    img_cubical = cubical_model.fit_transform(img_filtered)
    #fig = cubical_model.plot(img_cubical)
    return img_cubical


# def VRComplex(img_filtered):
#     VR_model = VietorisRipsPersistence(homology_dimensions=(0, 1),n_jobs=-1)
#     img_VR = VR_model.fit_transform(img_filtered)
#     fig = VR_model.plot(img_VR)
#     return fig


def Scaled(img_cubical):
    scalar_model = Scaler()
    img_scaled = scalar_model.fit_transform(img_cubical)
    fig = scalar_model.plot(img_scaled)
    return fig, img_scaled


def vectorization_of_persistence(img_scaled):
    heat_model = HeatKernel(sigma=0.15, n_bins=60, n_jobs=-1)
    img_heat = heat_model.fit_transform(img_scaled)

    fig = heat_model.plot(img_heat, homology_dimension_idx=1, colorscale='jet')
    return fig, img_heat