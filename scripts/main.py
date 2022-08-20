import pandas as pd
import matplotlib.pyplot as plt
import tda_analysis
import piplines
from sklearn import set_config 
#import data


X_train = pd.read_csv('../data/X_train.csv', index_col=False)
y_train = pd.read_csv('../data/y_train.csv', index_col=False)
X_test = pd.read_csv('../data/X_test.csv', index_col=False)

print(f'Shape of X_train: {X_train.shape}')
print(X_train.loc[1])

X_train = X_train.iloc[::, 1::]
X_test = X_test.iloc[::, 1::]
y_train = y_train["0"]
num_x = X_train.shape[0]
num_y = y_train.shape[0]
num_x_test = X_test.shape[0]
X_train = X_train.to_numpy().reshape((num_x, 28, 28))
X_test = X_test.to_numpy().reshape((num_x_test, 28, 28))
y_train = y_train.to_numpy().reshape((num_y, ))

print(f'Shape of X-train: {X_train.shape}, Shape of y: {y_train.shape}, Shape of X-test: {X_test.shape}')


#The grayscale number store
number = input('The number to input:')
index, img_train = tda_analysis.index_selection(int(number), X_train, y_train)
print(index)
plt.imshow(img_train, cmap='Greys')
plt.savefig('../figures/tda_analysis_mnist/digits/' + number + '/img' + number + 'grayscale.png')
#plt.show()


#Binarzing the number

img_binarized = tda_analysis.Binarizing(img_train)
plt.imshow(img_binarized, cmap='Greys')
plt.savefig('../figures/tda_analysis_mnist/digits/' + number + '/img' + number + 'binarized.png')
#plt.show()


#Radial Filtration of the Image
img_filtered = tda_analysis.radial_filter(img_binarized)
plt.imshow(img_filtered.reshape(28, 28), cmap= 'jet')
plt.savefig('../figures/tda_analysis_mnist/digits/' + number + '/img' + number + 'radialfiltered.png')
#plt.show()


#Cubical Complex of the radial filtered image
img_cubical = tda_analysis.CubeComplex(img_filtered)

#Scale the persistence Diagram
fig, img_scaled = tda_analysis.Scaled(img_cubical)
#fig.show()


#Vietoris Rips Complex
# img_VR = tda_analysis.VRComplex(img_filtered)
# img_VR.show()

fig1, img_heat = tda_analysis.vectorization_of_persistence(img_scaled)
#fig1.show()


#Using the initial pipeline
heat_pipeline = piplines.initial_pipeline()

img_train = img_train.reshape(1, 28, 28)
img_pipeline = heat_pipeline.fit_transform(img_train)
print(img_pipeline)


#The full pipeline
set_config(display='diagram')
tda_union = piplines.full_pipeline()
#Later set up the



#Topological FeatureExtraction
# X_train_tda = tda_union.fit_transform(X_train)
# print(X_train_tda.shape)

# X_train_tda_DF = pd.DataFrame(X_train_tda)
# X_train_tda_DF.to_csv('../data/tda_features/X_train_tda.csv')

X_test_tda = tda_union.fit_transform(X_test)
print(X_test_tda.shape)

X_test_tda_DF = pd.DataFrame(X_test_tda)
X_test_tda_DF.to_csv('../data/tda_features/X_test_tda.csv')

