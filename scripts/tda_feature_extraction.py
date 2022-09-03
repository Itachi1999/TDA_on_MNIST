import piplines
import numpy as np

X_train = np.load('../data/datasets/X_train.npy', allow_pickle=True)
X_test = np.load('../data/datasets/X_test.npy', allow_pickle=True)
y_train = np.load('../data/datasets/y_train.npy', allow_pickle=True)
y_test = np.load('../data/datasets/y_test.npy', allow_pickle=True)

#print(f'Shape of X: {X.shape}, Shape of y: {y.shape}')


#X_dummy = X[1].reshape(1, 28, 28)



# #The grayscale number store
# number = input('The number to input:')
# index, img_train = tda_analysis.index_selection(int(number), X_train, y_train)
# print(index)
# plt.imshow(img_train, cmap='Greys')
# plt.savefig('../figures/tda_analysis_mnist/digits/' + number + '/img' + number + 'grayscale.png')
# #plt.show()


# #Binarzing the number

# img_binarized = tda_analysis.Binarizing(img_train)
# plt.imshow(img_binarized, cmap='Greys')
# plt.savefig('../figures/tda_analysis_mnist/digits/' + number + '/img' + number + 'binarized.png')
# #plt.show()


# #Radial Filtration of the Image
# img_filtered = tda_analysis.radial_filter(img_binarized)
# plt.imshow(img_filtered.reshape(28, 28), cmap= 'jet')
# plt.savefig('../figures/tda_analysis_mnist/digits/' + number + '/img' + number + 'radialfiltered.png')
# #plt.show()


# #Cubical Complex of the radial filtered image
# img_cubical = tda_analysis.CubeComplex(img_filtered)

# #Scale the persistence Diagram
# fig, img_scaled = tda_analysis.Scaled(img_cubical)
# #fig.show()


# #Vietoris Rips Complex
# # img_VR = tda_analysis.VRComplex(img_filtered)
# # img_VR.show()

# fig1, img_heat = tda_analysis.vectorization_of_persistence(img_scaled)
# #fig1.show()


# #Using the initial pipeline
# heat_pipeline = piplines.initial_pipeline()

# img_train = img_train.reshape(1, 28, 28)
# img_pipeline = heat_pipeline.fit_transform(img_train)
# print(img_pipeline)


#The full pipeline
# set_config(display='diagram')
# tda_union = piplines.full_pipeline()
#Later set up the



#Topological FeatureExtraction
# X_train_tda = tda_union.fit_transform(X_train)
# print(X_train_tda.shape)

# X_train_tda_DF = pd.DataFrame(X_train_tda)
# X_train_tda_DF.to_csv('../data/tda_features/X_train_tda.csv')
tda_union = piplines.paper_pipeline()

print('Starting the tda process for X_train\n')
X_train_tda = tda_union.fit_transform(X_train)
print(f'Shape of X_train_tda: {X_train_tda.shape}')
print('Ending tda process for X_train\n')
print('Saving X_train\n')
np.save('../data/datasets/tda_features/X_train_tda.npy', X_train_tda)
print('Saved\n')

print('Starting the tda process for X_test\n')
X_test_tda = tda_union.fit_transform(X_test)
print(f'Shape of X_test_tda: {X_test_tda.shape}')
print('Ending tda process for X_test\n')
print('Saving X_test')
np.save('../data/datasets/tda_features/X_test_tda.npy', X_test_tda)
print('Saved\n')
#X_dummy_tda = tda_union.fit_transform(X_dummy)
#print(X_dummy_tda.shape)

# X_test_tda_DF = pd.DataFrame(X_test_tda)
# X_test_tda_DF.to_csv('../data/tda_features/X_test_tda.csv')

