"""
The goal of this assignment is to predict GPS coordinates from image features using k-Nearest Neighbors.
Specifically, have featurized 28616 geo-tagged images taken in Spain split into training and test sets (27.6k and 1k).

The assignment walks students through:
    * visualizing the data
    * implementing and evaluating a kNN regression model
    * analyzing model performance as a function of dataset size
    * comparing kNN against linear regression

Images were filtered from Mousselly-Sergieh et al. 2014 (https://dl.acm.org/doi/10.1145/2557642.2563673)
and scraped from Flickr in 2024. The image features were extracted using CLIP ViT-L/14@336px (https://openai.com/clip/).
"""

import matplotlib.pyplot as plt
import numpy as np
import imageio

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def plot_data(train_feats, train_labels):
    """
    Input:
        train_feats: Training set image features
        train_labels: Training set GPS (lat, lon)

    Output:
        Displays plot of image locations, and first two PCA dimensions vs longitude
    """
    # Plot image locations
    plt.scatter(train_labels[:, 1], train_labels[:, 0], marker=".")
    plt.title('Image Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    # Standardize the data and run PCA on training_feats
    transformed_feats = StandardScaler().fit_transform(train_feats)
    transformed_feats = PCA(n_components=2).fit_transform(transformed_feats)

    # Plot images by first two PCA dimensions
    plt.scatter(transformed_feats[:, 0],     # Select first column
                transformed_feats[:, 1],     # Select second column
                c=train_labels[:, 1],
                marker='.')
    plt.colorbar(label='Longitude')
    plt.title('Image Features by Longitude after PCA')
    plt.show()

def plot_3nn(test_image_filename, nn_filenames):
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))
    axes = axes.flatten()

    # Plot test image
    axes[0].imshow(imageio.imread(f'im2spain_images/{test_image_filename}'))
    axes[0].set_title('test image')
    axes[0].axis('off')

    # Plot nearest neighbors
    for i in range(3):
        axes[i+1].imshow(imageio.imread(f'im2spain_images/{nn_filenames[i]}'))
        axes[i+1].set_title(f'nearest neighbor {i+1}')
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

# TODO: You may find implementing this function useful...
def displacement_error(y, y_hat):
    """
    Input:
        y: True (lat, lon) coords
        y_hat: Predicted (lat, lon) coords

    Returns:
        Displacement error
    """
    return ...

def grid_search(train_features, train_labels, test_features, test_labels, is_weighted=False, verbose=True):
    """
    Input:
        train_features: Training set image features
        train_labels: Training set GPS (lat, lon) coords
        test_features: Test set image features
        test_labels: Test set GPS (lat, lon) coords
        is_weighted: Weight prediction by distances in feature space

    Output:
        Prints mean displacement error as a function of k
        Plots mean displacement error vs k

    Returns:
        Minimum mean displacement error
    """
    # Evaluate mean displacement error (in miles) of kNN regression for different values of k
    # Technically we are working with spherical coordinates and should be using spherical distances, but within a small
    # region like Spain we can get away with treating the coordinates as cartesian coordinates.
    knn = NearestNeighbors(n_neighbors=100).fit(train_features)

    if verbose:
        print(f'Running grid search for k (is_weighted={is_weighted})')

    ks = list(range(1, 11)) + [20, 30, 40, 50, 100]
    mean_errors = []
    for k in ks:
        distances, indices = knn.kneighbors(test_features, n_neighbors=k)

        errors = []
        for i, nearest in enumerate(indices):
            # Evaluate mean displacement error in miles for each test image
            # Assume 1 degree latitude is 69 miles and 1 degree longitude is 52 miles
            y = test_labels[i]

            ##### TODO(d): Your Code Here #####
            ##### TODO(f): Modify Your Code #####
            if is_weighted:
                weights = 1 / (distances[i] + 1e-8)
                yhat = np.average(train_labels[nearest], axis=0, weights=weights)
            else:
                yhat = np.mean(train_labels[nearest], axis=0)
            e = np.linalg.norm((y-yhat) * [69, 52])

            errors.append(e)
        
        mean_error = np.mean(np.array(errors))
        mean_errors.append(mean_error)
        if verbose:
            print(f'{k}-NN mean displacement error (miles): {mean_error:.1f}')

    # Plot error vs k for k Nearest Neighbors
    if verbose:
        plt.plot(ks, mean_errors)
        plt.xlabel('k')
        plt.ylabel('Mean Displacement Error (miles)')
        plt.title('Mean Displacement Error (miles) vs. k in kNN')
        plt.show()

    print(mean_errors.index(min(mean_errors)))
    print(min(mean_errors))
    return min(mean_errors)


def main():
    print("Predicting GPS from CLIP image features\n")

    # Import Data
    print("Loading Data")
    data = np.load('im2spain_data.npz')

    train_features: np.ndarray = data['train_features']  # [N_train, dim] array
    test_features: np.ndarray = data['test_features']    # [N_test, dim] array
    train_labels: np.ndarray = data['train_labels']      # [N_train, 2] array of (lat, lon) coords
    test_labels: np.ndarray = data['test_labels']        # [N_test, 2] array of (lat, lon) coords
    train_files: np.ndarray = data['train_files']        # [N_train] array of strings
    test_files: np.ndarray = data['test_files']          # [N_test] array of strings

    # Data Information
    print('Train Data Count:', train_features.shape[0])

    # Feature and label visualization
    # plot_data(train_features, train_labels)

    test_image_filename = '53633239060.jpg'
    # Part (b): Use knn to get the 3 nearest neighbors of test image 53633239060.jpg
    index = test_files.tolist().index(test_image_filename)
    nn = NearestNeighbors(n_neighbors=3)
    testimg_features = test_features[index]
    nn.fit(train_features)

    distances, indices = nn.kneighbors(testimg_features.reshape(1, -1), 3)
    indices = indices[0]
    nn_coords = [train_labels[x] for x in indices]
    nn_filenames = [train_files[x] for x in indices]
    
    testimg_coords = test_labels[index]
    print(f"Test Image Coords: {testimg_coords}")
    for i in range(3):
        print(f"Image {i} Coords: {nn_coords[i]}")

    # Visualize the images
    # plot_3nn(test_image_filename, nn_filenames)

    # Part (c): establish a naive baseline of predicting the mean of the training set
    ##### TODO(c): Your Code Here #####
    center = np.mean(train_labels, axis=0)
    naive_error = np.mean(np.linalg.norm((test_labels - center) * [69, 52], axis=1))

    print(f'\nNaive baseline mean displacement error (miles: {naive_error:.1f})')

    # Part (d): complete grid_search to find the best value of k
    grid_search(train_features, train_labels, test_features, test_labels)

    # Part (f): rerun grid search weighted by distance to find the best value of k
    grid_search(train_features, train_labels, test_features, test_labels, is_weighted=True)


    # Part (g): compare to linear regression for different # of training points
    mean_errors_lin = []
    mean_errors_nn = []
    ratios = np.arange(0.1, 1.1, 0.1)
    for r in ratios:
        num_samples = int(r * len(train_features))
        ##### TODO(g): Your Code Here #####
        X = train_features[:num_samples]
        y = train_labels[:num_samples]
        lr = LinearRegression().fit(X, y)
        pred = lr.predict(test_features)
        err = np.linalg.norm((test_labels - pred) * [69,52], axis=1)
        e_lin = np.mean(err)
        e_nn = grid_search(X, y, test_features, test_labels, is_weighted=True)

        mean_errors_lin.append(e_lin)
        mean_errors_nn.append(e_nn)

        print(f'\nTraining set ratio: {r:.1f} ({num_samples})')
        print(f'Linear Regression mean displacement error (miles): {e_lin:.1f}')
        print(f'kNN mean displacement error (miles): {e_nn:.1f}')

    # Plot error vs training set size
    plt.plot(ratios, mean_errors_lin, label='lin. reg.')
    plt.plot(ratios, mean_errors_nn, label='kNN')
    plt.xlabel('Training Set Ratio')
    plt.ylabel('Mean Displacement Error (miles)')
    plt.title('Mean Displacement Error (miles) vs. Training Set Ratio')
    plt.legend()
    plt.show()
       

if __name__ == '__main__':
    main()
