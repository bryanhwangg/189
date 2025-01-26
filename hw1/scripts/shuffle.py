import numpy as np

MNIST_DATA_LENGTH = 60000


def split_and_write_data(data: np.ndarray, training_percent=0.0, training_amount=0):
    """
    Splits 2d numPy array into training and testing data based on a given training amount (int) OR training percentage (float)
    Saves the split data into npy files in data directory
    """
    data_length = data.shape[0]
    if training_percent:
        index = int(data_length * training_percent)
        testing_data = data[:index]
        training_data = data[index:]
    else:
        training_data = data[:training_amount]
        testing_data = data[training_amount:]

    data_name = "spam"
    if data_length == MNIST_DATA_LENGTH:
        data_name = "mnist"

    np.save(f"../data/{data_name}_training_data.npy", training_data)
    np.save(f"../data/{data_name}_testing_data.npy", testing_data)


def main():
    mnist_full_data = np.load("../data/mnist-data.npz")
    spam_full_data = np.load("../data/spam-data.npz")

    # Set random seed for reproducibility
    np.random.seed(189)

    # Extract subsets of data that we need
    mnist_training_data, spam_training_data = (
        mnist_full_data["training_data"],
        spam_full_data["training_data"],
    )
    mnist_data_labels, spam_data_labels = (
        mnist_full_data["training_labels"],
        spam_full_data["training_labels"],
    )

    # Reshape and flatten our 28x28 matrix representation of each MNIST image into a vector
    # mnist_training_data_2d shape = (60,000, 784)
    # spam_training_data_2d shape = (4171, 32)
    mnist_training_data_2d, spam_training_data_2d = mnist_training_data.reshape(
        mnist_training_data.shape[0], -1
    ), spam_training_data.reshape(spam_training_data.shape[0], -1)

    # Join the 2d data together with given labels
    # mnist_training_data_2d shape = (60,000, 785)
    # spam_training_data_2d shape = (4171, 33)
    mnist_training_data_2d = np.column_stack(
        (mnist_training_data_2d, mnist_data_labels)
    )
    spam_training_data_2d = np.column_stack((spam_training_data_2d, spam_data_labels))

    # Shuffle joined data randomly
    np.random.shuffle(mnist_training_data_2d)
    np.random.shuffle(spam_training_data_2d)

    split_and_write_data(mnist_training_data_2d, training_amount=10000)
    split_and_write_data(spam_training_data_2d, training_percent=0.2)


if __name__ == "__main__":
    main()
    print("MNIST and spam data split successfully")
