import numpy as np
import matplotlib.pyplot as plt
import preprocessing

MAX_EPOCHS = 20      # Maximum number of epochs
WEIGHT_MIN = -0.05  # Initial weights minimum value
WEIGHT_MAX = 0.05   # Initial weights maximum value
ETA = 0.05
ALPHA = 0.05
KMIN = 50

def trainAutoEncoder(train_data):
    # m - the number of data points
    # k - the number of dimensions
    m = train_data.shape[0]
    k = train_data.shape[1]

    # Middle hidden layer size
    k_mid = int((k + KMIN) / 2)

    # Data to first hiddeen layer
    weights1 = np.random.rand(k, k_mid)*(WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN
    weights1_update_old = np.zeros((k, k_mid))
    # First to Second hidden layer - end of auto encoding
    weights2 = np.random.rand(k_mid, KMIN)*(WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN
    weights2_update_old = np.zeros((k_mid, KMIN))
    # First decoding layer
    weights3 = np.random.rand(KMIN, k_mid)*(WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN
    weights3_update_old = np.zeros((KMIN, k_mid))
    # Second decoding layer
    weights4 = np.random.rand(k_mid, k)*(WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN
    weights4_update_old = np.zeros((k_mid, k))

    # Train the algorithm
    perm = np.array(range(0, m))
    epoch_count = 0
    while epoch_count < MAX_EPOCHS + 1:
        # Increment epoch
        epoch_count += 1
        print("Doing epoch", epoch_count)

        # Permute the data
        perm = np.random.permutation(perm)
        train_data = train_data[perm, :]

        # Perform stochastic gradient descent
        for row in train_data:
            # Calculate the output of each layer
            hidden1 = np.reciprocal(1 + np.exp(-np.dot(row, weights1)))
            hidden2 = np.reciprocal(1 + np.exp(-np.dot(hidden1, weights2)))
            hidden3 = np.reciprocal(1 + np.exp(-np.dot(hidden2, weights3)))
            output = np.reciprocal(1 + np.exp(-np.dot(hidden3, weights4)))

            # Calculate the back propagation delta values for each layer
            delta4 = (output * (1 - output)) * (row - output)
            delta3 = hidden3 * (1 - hidden3) * np.dot(weights4, delta4)
            delta2 = hidden2 * (1 - hidden2) * np.dot(weights3, delta3)
            delta1 = hidden1 * (1 - hidden1) * np.dot(weights2, delta2)
            
            # Calculate the weight updates
            weights4_update = ETA * np.outer(hidden3, delta4)
            weights3_update = ETA * np.outer(hidden2, delta3)
            weights2_update = ETA * np.outer(hidden1, delta2)
            weights1_update = ETA * np.outer(row, delta1)

            # Update the weights
            weights1 = weights1 + weights1_update + ALPHA * weights1_update_old
            weights1_update_old = weights1_update
            weights2 = weights2 + weights2_update + ALPHA * weights2_update_old
            weights2_update_old = weights2_update
            weights3 = weights3 + weights3_update + ALPHA * weights3_update_old
            weights3_update_old = weights3_update
            weights4 = weights4 + weights4_update + ALPHA * weights4_update_old
            weights4_update_old = weights4_update

        if(epoch_count % 5 == 0):
            file1 = "weights1_" + str(epoch_count) + ".txt"
            np.savetxt(file1, weights1)
            file2 = "weights2_" + str(epoch_count) + ".txt"
            np.savetxt(file2, weights2)
            file3 = "weights3_" + str(epoch_count) + ".txt"
            np.savetxt(file3, weights3)
            file4 = "weights4_" + str(epoch_count) + ".txt"
            np.savetxt(file4, weights4)
            # Note: momentum is applied based on each update to the weights not per epoch
    return (weights1, weights2, weights3, weights4)

def trainAutoEncoder(train_data, epoch_decay = None, k_in = None, savefile1 = None, savefile2 = None, savefile3 = None, savefile4 = None):
    if(k_in != None):
        k_min = k_in
    else:
        k_min = KMIN
    # m - the number of data points
    # k - the number of dimensions
    m = train_data.shape[0]
    k = train_data.shape[1]

    # Middle hidden layer size
    k_mid = int((k + k_min) / 2)

    # Data to first hiddeen layer
    weights1 = np.random.rand(k, k_mid)*(WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN
    weights1_update_old = np.zeros((k, k_mid))
    # First to Second hidden layer - end of auto encoding
    weights2 = np.random.rand(k_mid, k_min)*(WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN
    weights2_update_old = np.zeros((k_mid, k_min))
    # First decoding layer
    weights3 = np.random.rand(k_min, k_mid)*(WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN
    weights3_update_old = np.zeros((k_min, k_mid))
    # Second decoding layer
    weights4 = np.random.rand(k_mid, k)*(WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN
    weights4_update_old = np.zeros((k_mid, k))

    # Train the algorithm
    perm = np.array(range(0, m))
    epoch_count = 0
    while epoch_count < MAX_EPOCHS + 1:
        if(epoch_decay != None and epoch_count > epoch_decay):
            alpha = ALPHA - ALPHA/(MAX_EPOCHS - epoch_decay) * (epoch_count - epoch_decay)
        else:
            alpha = ALPHA
        # Increment epoch
        epoch_count += 1
        print("Doing epoch", epoch_count)

        # Permute the data
        perm = np.random.permutation(perm)
        train_data = train_data[perm, :]

        # Perform stochastic gradient descent
        for row in train_data:
            # Calculate the output of each layer
            hidden1 = np.reciprocal(1 + np.exp(-np.dot(row, weights1)))
            hidden2 = np.reciprocal(1 + np.exp(-np.dot(hidden1, weights2)))
            hidden3 = np.reciprocal(1 + np.exp(-np.dot(hidden2, weights3)))
            output = np.reciprocal(1 + np.exp(-np.dot(hidden3, weights4)))

            # Calculate the back propagation delta values for each layer
            delta4 = (output * (1 - output)) * (row - output)
            delta3 = hidden3 * (1 - hidden3) * np.dot(weights4, delta4)
            delta2 = hidden2 * (1 - hidden2) * np.dot(weights3, delta3)
            delta1 = hidden1 * (1 - hidden1) * np.dot(weights2, delta2)
            
            # Calculate the weight updates
            weights4_update = ETA * np.outer(hidden3, delta4)
            weights3_update = ETA * np.outer(hidden2, delta3)
            weights2_update = ETA * np.outer(hidden1, delta2)
            weights1_update = ETA * np.outer(row, delta1)

            # Update the weights
            weights1 = weights1 + weights1_update + alpha * weights1_update_old
            weights1_update_old = weights1_update
            weights2 = weights2 + weights2_update + alpha * weights2_update_old
            weights2_update_old = weights2_update
            weights3 = weights3 + weights3_update + alpha * weights3_update_old
            weights3_update_old = weights3_update
            weights4 = weights4 + weights4_update + alpha * weights4_update_old
            weights4_update_old = weights4_update

        if(epoch_count % 5 == 0):
            if(savefile1 != None):
                file1 = savefile1
            else:
                file1 = "weights1_" + str(epoch_count) + ".txt"
            np.savetxt(file1, weights1)
            if(savefile2 != None):
                file2 = savefile2
            else:
                file2 = "weights2_" + str(epoch_count) + ".txt"
            np.savetxt(file2, weights2)
            if(savefile3 != None):
                file3 = savefile3
            else:
                file3 = "weights3_" + str(epoch_count) + ".txt"
            np.savetxt(file3, weights3)
            if(savefile4 != None):
                file4 = savefile4
            else:
                file4 = "weights4_" + str(epoch_count) + ".txt"
            np.savetxt(file4, weights4)
            # Note: momentum is applied based on each update to the weights not per epoch
    return (weights1, weights2, weights3, weights4)

def trainOnMNIST():
    # Pre-process the training data
    train_data = np.loadtxt("mnist_train.csv", delimiter=",", skiprows=1)
    perm = np.array(range(0, train_data.shape[0]))
    perm = np.random.permutation(perm)
    train_data = train_data[perm, :]
    train_data = train_data[0:1000, :]

    train_label = train_data[:, 0]
    train_data = train_data / 255
    train_data[: , 0] = 1

    # Pre-process the test data
    test_data = np.loadtxt("mnist_test.csv", delimiter=",", skiprows=1)
    perm = np.array(range(0, test_data.shape[0]))
    perm = np.random.permutation(perm)
    test_data = test_data[perm, :]
    test_data = test_data[0:1000, :]

    test_label = test_data[:, 0]
    test_data = test_data / 255
    test_data[: , 0] = 1

    # Save the randomly selected training and test data
    np.savetxt("train_subset.txt", train_data)
    np.savetxt("train_label.txt", train_label)
    np.savetxt("test_subset.txt", test_data)
    np.savetxt("test_label.txt", test_label)

    # Train auto-encoder
    trainAutoEncoder(train_data)

def evalOnMNIST():
    train_data = np.loadtxt("train_subset.txt")
    train_label = np.loadtxt("train_label.txt")
    test_data = np.loadtxt("test_subset.txt")
    test_label = np.loadtxt("test_label.txt")

    fig0, ax0 = plt.subplots()
    ax0.imshow(train_data[0, 1:].reshape((28, 28)), origin='lower', cmap='gray')
    ax0.invert_xaxis()
    plt.show()

    epochs = []
    error = []
    for num in range(5, MAX_EPOCHS + 1, 5):
        file1 = "weights1_" + str(num) + ".txt"
        weights1 = np.loadtxt(file1)
        file2 = "weights2_" + str(num) + ".txt"
        weights2 = np.loadtxt(file2)
        file3 = "weights3_" + str(num) + ".txt"
        weights3 = np.loadtxt(file3)
        file4 = "weights4_" + str(num) + ".txt"
        weights4 =  np.loadtxt(file4)

        # Evaluate the data
        eval = np.copy(train_data)
        
        for index, row in enumerate(eval):
            tmp = np.reciprocal(1 + np.exp(-np.dot(row, weights1)))
            tmp = np.reciprocal(1 + np.exp(-np.dot(tmp, weights2)))
            tmp = np.reciprocal(1 + np.exp(-np.dot(tmp, weights3)))
            tmp = np.reciprocal(1 + np.exp(-np.dot(tmp, weights4)))
            eval[index, :] = tmp

        
        fig1, ax1 = plt.subplots()
        ax1.set_title("Epoch = " + str(num))
        ax1.imshow(eval[0, 1:].reshape((28, 28)), origin='lower', cmap='gray')
        ax1.invert_xaxis()
        plt.show()

        # Test the encoded data in nearest neighbor
        encode_train = np.zeros((1000, KMIN))
        for index, row in enumerate(train_data):
            tmp = np.reciprocal(1 + np.exp(-np.dot(row, weights1)))
            tmp = np.reciprocal(1 + np.exp(-np.dot(tmp, weights2)))
            encode_train[index, :] = tmp

        encode_test = np.zeros((1000, KMIN))
        for index, row in enumerate(test_data):
            tmp = np.reciprocal(1 + np.exp(-np.dot(row, weights1)))
            tmp = np.reciprocal(1 + np.exp(-np.dot(tmp, weights2)))
            encode_test[index, :] = tmp

        # Check encoding in nearest neighbor
        epochs.append(num)
        error.append(kNearestNeighbor(encode_train, train_label, encode_test, test_label, 11))
    
    baseError = kNearestNeighbor(train_data, train_label, test_data, test_label, 11)
    baseErrorArr = baseError * np.ones(len(error))

    fig2, ax2 = plt.subplots()
    ax2.set_title("Nearest Neighbor Error")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    plt.plot(epochs, error, label="Encoded Data")
    plt.plot(epochs, baseErrorArr, label="Base Testing Data")
    plt.legend()
    plt.show()
    
def trainOnStudent():
    # Set to true once all data is made
    trained = False

    train_data, train_label, test_data, test_label = preprocessing.main()
    # unonehot the labels
    #train_label = np.argmax(train_label, axis=1)
    test_label = np.argmax(test_label, axis=1)

    # you can also mess with this too
    k_min = [2, 4, 6, 10, 14, 25]
    n_vals = [2, 3, 4, 6, 10, 14]

    # m - the number of data points
    # k - the number of dimensions
    m = train_data.shape[0]
    k = train_data.shape[1]
    epoch_decay = 3

    # Validate fitting method
    baseError = neuralNet(train_data, train_label, test_data, test_label, ETA, ALPHA, 20, 5)
    print(baseError)

    # Error Values for plotting
    epochsAll = []
    errorAll = []
    for index_k, k_val in enumerate(k_min):
        epochs = []
        error = []
        print("k = " + str(k_val) + ":")
        # Middle hidden layer size
        k_mid = int((k + k_val) / 2)

        # Data to first hiddeen layer
        weights1 = np.random.rand(k, k_mid)*(WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN
        weights1_update_old = np.zeros((k, k_mid))
        # First to Second hidden layer - end of auto encoding
        weights2 = np.random.rand(k_mid, k_val)*(WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN
        weights2_update_old = np.zeros((k_mid, k_val))
        # First decoding layer
        weights3 = np.random.rand(k_val, k_mid)*(WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN
        weights3_update_old = np.zeros((k_val, k_mid))
        # Second decoding layer
        weights4 = np.random.rand(k_mid, k)*(WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN
        weights4_update_old = np.zeros((k_mid, k))

        # Train the algorithm
        perm = np.array(range(0, m))
        epoch_count = 0
        while epoch_count < MAX_EPOCHS + 1:
            if(not trained):
                # Alpha decay
                if(epoch_count > epoch_decay):
                    alpha = ALPHA - ALPHA/(MAX_EPOCHS - epoch_decay) * (epoch_count - epoch_decay)
                else:
                    alpha = ALPHA
                
                # Increment epoch
                epoch_count += 1
                print("Doing epoch", epoch_count)

                # Permute the data
                perm = np.random.permutation(perm)
                train_data = train_data[perm, :]

                # Perform stochastic gradient descent
                for row in train_data:
                    # Calculate the output of each layer
                    hidden1 = np.reciprocal(1 + np.exp(-np.dot(row, weights1)))
                    hidden2 = np.reciprocal(1 + np.exp(-np.dot(hidden1, weights2)))
                    hidden3 = np.reciprocal(1 + np.exp(-np.dot(hidden2, weights3)))
                    output = np.reciprocal(1 + np.exp(-np.dot(hidden3, weights4)))

                    # Calculate the back propagation delta values for each layer
                    delta4 = (output * (1 - output)) * (row - output)
                    delta3 = hidden3 * (1 - hidden3) * np.dot(weights4, delta4)
                    delta2 = hidden2 * (1 - hidden2) * np.dot(weights3, delta3)
                    delta1 = hidden1 * (1 - hidden1) * np.dot(weights2, delta2)
                    
                    # Calculate the weight updates
                    weights4_update = ETA * np.outer(hidden3, delta4)
                    weights3_update = ETA * np.outer(hidden2, delta3)
                    weights2_update = ETA * np.outer(hidden1, delta2)
                    weights1_update = ETA * np.outer(row, delta1)

                    # Update the weights
                    weights1 = weights1 + weights1_update + alpha * weights1_update_old
                    weights1_update_old = weights1_update
                    weights2 = weights2 + weights2_update + alpha * weights2_update_old
                    weights2_update_old = weights2_update
                    weights3 = weights3 + weights3_update + alpha * weights3_update_old
                    weights3_update_old = weights3_update
                    weights4 = weights4 + weights4_update + alpha * weights4_update_old
                    weights4_update_old = weights4_update
            if(epoch_count % 5 == 0):
                if(not trained):
                    file1 = "student1_k" + str(k) + "_" + str(epoch_count) + ".txt"
                    np.savetxt(file1, weights1)
                    file2 = "student2_k" + str(k) + "_" + str(epoch_count) + ".txt"
                    np.savetxt(file2, weights2)
                else:
                    weights1 = np.loadtxt(file1)
                    weights2 = np.loadtxt(file2)
            
                # Test the encoded data in nearest neighbor
                encode_train = np.zeros((train_data.shape[0], k_val))
                for index, row in enumerate(train_data):
                    tmp = np.reciprocal(1 + np.exp(-np.dot(row, weights1)))
                    tmp = np.reciprocal(1 + np.exp(-np.dot(tmp, weights2)))
                    encode_train[index, :] = tmp

                encode_test = np.zeros((test_data.shape[0],k_val))
                for index, row in enumerate(test_data):
                    tmp = np.reciprocal(1 + np.exp(-np.dot(row, weights1)))
                    tmp = np.reciprocal(1 + np.exp(-np.dot(tmp, weights2)))
                    encode_test[index, :] = tmp

                # Check encoding in nearest neighbor
                epochs.append(epoch_count)
                # Uncomment for neareest neighbor
                # error.append(kNearestNeighbor(encode_train, train_label, encode_test, test_label, 11))
                error.append(neuralNet(encode_train, train_label, encode_test, test_label, ETA, ALPHA, n_vals[index_k], 5)) 
        epochsAll.append(epochs)
        errorAll.append(error)
    baseErrorArr = baseError * np.ones(len(error))

    fig2, ax2 = plt.subplots()
    ax2.set_title("Neeural Net Error")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    for ind in range(0, len(k_min)):
        plt.plot(epochsAll[ind], errorAll[ind], label="Encoded Data $k = " + str(k_min[ind]) +"$")
    plt.plot(epochsAll[0], baseErrorArr, label="Base Testing Data")
    plt.legend()
    plt.show()
    

def kNearestNeighbor(training_data, training_label, test_data, test_label, k):
    # The number of test data points
    m = test_data.shape[0]
    
    # Calculate the initial distances
    label = np.zeros(m)
    for ind_i in range(0, m):
        temp = training_data - np.array(test_data[ind_i, :])
        temp = np.multiply(temp, temp)
        dist = np.sum(temp, axis=1)
        smallest = dist.argsort()
        labelsToPoll = training_label[smallest[0:k]]
        counts = np.bincount(labelsToPoll.astype(int))
        label[ind_i] = np.argmax(counts)
    return np.sum(label == test_label) / float(m)

def neuralNet(train_data, train_target, test_data, test_target, eta, alpha, n, max_epoch):
    print("Starting new training run.")
    # m = 60,000 for MNIST (m the number of examples)
    m = train_data.shape[0]
    m_star = test_data.shape[0]

    # Initialize weights of hidden layer: k -> n - k is the row size, n is the number of hidden nodes
    # k from the encoded data
    # n should be smaller than k
    k = train_data.shape[1]
    weights1 = np.random.rand(k, n)*(WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN
    weights1_update_old = np.zeros((k, n))

    # Initialize weights of output layer: n -> o - o is the output size, 3 in this case
    o = train_target.shape[1]
    weights2 = np.random.rand(n, o)*(WEIGHT_MAX - WEIGHT_MIN) + WEIGHT_MIN
    weights2_update_old = np.zeros((n, o))

    # Train the algorithm
    perm = np.array(range(0, m))
    epoch_count = 0
    while epoch_count < max_epoch:
        # Increment epoch
        epoch_count += 1

        #if epoch_count % 10 == 0:
        print("Doing epoch", epoch_count)

        # Permute the data
        perm = np.random.permutation(perm)
        train_data = train_data[perm, :]
        train_target = train_target[perm, :]

        # Perform stochastic gradient descent
        for index, row in enumerate(train_data):
            # Calculate the output of each layer
            hidden = np.reciprocal(1 + np.exp(-np.dot(row, weights1)))
            output = np.reciprocal(1 + np.exp(-np.dot(hidden, weights2)))

            # Calculate the back propagation delta values for each layer
            delta_output = (output * (1 - output)) * (train_target[index, :] - output)
            delta_hidden = hidden * (1 - hidden) * np.dot(weights2, delta_output)
            
            # Calculate the weight updates
            weights2_update = eta * np.outer(hidden, delta_output)
            weights1_update = eta * np.outer(row, delta_hidden)

            # Update the weights
            weights1 = weights1 + weights1_update + alpha * weights1_update_old
            weights1_update_old = weights1_update
            weights2 = weights2 + weights2_update + alpha * weights2_update_old
            weights2_update_old = weights2_update
            # Note: momentum is applied based on each update to the weights not per epoch

    label = np.zeros(test_target.shape[0])
    for index, row in enumerate(test_data):
        # Calculate the output of each layer
        hidden = np.reciprocal(1 + np.exp(-np.dot(row, weights1)))
        output = np.reciprocal(1 + np.exp(-np.dot(hidden, weights2)))
        label[index] = np.argmax(output)
    return np.sum(label.astype(int) == test_target) / float(m_star)



if __name__ == "__main__":
    #trainOnMNIST() # done
    #evalOnMNIST() # done
    trainOnStudent()