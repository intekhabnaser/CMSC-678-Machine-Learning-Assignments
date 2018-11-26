'''
Name: Intekhab Naser
Campus ID : ZC11577
E-mail: intek1@umbc.edu
Machine Learning HW1: Decision trees and KNN

'''
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter

#Loading the data
M = loadmat('MNIST_digit_data.mat')
images_train,images_test,labels_train,labels_test= M['images_train'],M['images_test'],M['labels_train'],M['labels_test']

#just to make all random sequences on all computers the same.
np.random.seed(8)

#randomly permute data points
inds = np.random.permutation(images_train.shape[0])
images_train = images_train[inds]
labels_train = labels_train[inds]

inds = np.random.permutation(images_test.shape[0])
images_test = images_test[inds]
labels_test = labels_test[inds]

#if you want to use only the first 1000 data points.
images_train = images_train[0:10000,:]
labels_train = labels_train[0:10000,:]


#show the 10'th train image
#i=10
#im = images_train[i,:].reshape((28,28),order='F')
#plt.imshow(im)
#plt.title('Class Label:'+str(labels_train[i][0]))
#plt.show()


# Function for finding kNN to test data, predicting the labels for it and calculating classwise and average accuracy 

def kNearestNeighbors(images_train, labels_train, images_test, labels_test, k):
    
    # Dictionary for arranging the training data classwise
    train_datadic = defaultdict(list)
    
    # Iterating 'labels_train' and 'images_train' to store the training data for each class in dictionary
    for idx, i in enumerate(labels_train):
        train_datadic[i[0]].append(images_train[idx].tolist())
    
    # Variables for storing number of total test images and number of correct predictions done
    total, true = 0, 0


    testLabelDic = defaultdict(int) # Dictionary for storing number of total corresponding test labels
    predLabelDic = defaultdict(int) # Dictionary for storing number of correctly predicted corresponding test labels
    
    for idx, test_img in enumerate(images_test):
        d = []  # list for storing all the distances of train dataset images from test dataset image

        testLabelDic[labels_test[idx][0]] += 1
        
        for num in train_datadic:
            for train_img in train_datadic[num]:
                euclidean = np.linalg.norm(np.array(train_img) - np.array(test_img))
                d.append([euclidean, num])
        
        nearestK = [i[1] for i in sorted(d)[:k]]    # List of K nearest distance train dataset images
        predict_result = Counter(nearestK).most_common(1)[0][0]     # Most common image label from the k images
        
        if predict_result == labels_test[idx][0]:
            true = true + 1            
            predLabelDic[predict_result] += 1

        total = total + 1
        
    accuracy = [] # list of accuracy for each label
    for label in testLabelDic:
        accuracy.append(predLabelDic[label] / testLabelDic[label])
        
    acc_av = true/total # Average accuracy across all classes/labels
    return accuracy, acc_av


# Function to measure and plot accuracy results for different training data sizes
def dataSizeAccuracy():
    datapoints = np.logspace(np.log10(31.0),np.log10(10000.0),num = 10,base=10.0,dtype='int')
    acc_av = []
    for dp in datapoints:
        acc_av.append(kNearestNeighbors(images_train[:dp], labels_train[:dp], images_test[:500], labels_test[:500], 1)[1])
    plt.xlabel("Datapoints")
    plt.ylabel("Accuracy")
    plt.plot(datapoints,acc_av,'-o')
    plt.savefig('DataSize_Accuracy.png')
    plt.show()


# Function to measure and plot accuracy results for different training data sizes along with different sizes of K
def kSizeDataSizeAccuracy():
    datapoints = np.logspace(np.log10(31.0),np.log10(10000.0),num = 10,base=10.0,dtype='int')
    K = [1, 2, 3, 5, 10]
    acc_av = []

    for k in K:
        acc_av = []
        for dp in datapoints:
            acc_av.append(kNearestNeighbors(images_train[:dp], labels_train[:dp], images_test[:500], labels_test[:500], k)[1])
        plt.plot(datapoints,acc_av,label='k='+str(k))
    plt.legend(loc = 'best')
    plt.xlabel("Datapoints")
    plt.ylabel("Accuracy")
    plt.savefig('KSize_DataSize_Accuracy.png')
    plt.show()


# Function to measure and plot accuracy results for different sizes of K
def kSizeAccuracy():
    K = [1, 2, 3, 5, 10]
    acc_av = []
    images_tr = images_train[:1000]
    images_te = images_train[1000:2000]
    labels_tr = labels_train[:1000]
    labels_te = labels_train[1000:2000]
    for k in K:
        acc_av.append(kNearestNeighbors(images_tr, labels_tr, images_te, labels_te, k)[1])
    plt.plot(K,acc_av,'-o')
    plt.xlabel("K-values")
    plt.ylabel("Accuracy")
    plt.savefig('kSize_Accuracy.png')
    plt.show()

# Main Function
def main():
    print("1.Show accuracy for each label and the average accuracy\n2.Accuracy plot for varying dataset sizes\n3. Accuracy plot for varying dataset sizes and K size in [1,2,3,5,10]\n4. Accuracy plot for Best K in [1,2,3,5,10] for 1000 Training and 1000 Validation data")
    choice = int(input("***Enter your choice***\n"))
    if choice == 1:
        accuracy = kNearestNeighbors(images_train[:10000], labels_train[:10000], images_test[:1000], labels_test[:1000], 5)
        print("Accuracy vector: ",accuracy[0])
        print("Average Accuracy: ",accuracy[1])
    elif choice == 2:
        print("Plot for the dataset sizes in range 30 to 10000")
        dataSizeAccuracy()
    elif choice == 3:
        print("Plot for the dataset sizes in range 30 to 10000 and K=[1,2,3,5,10]")
        kSizeDataSizeAccuracy()
    elif choice == 4:
        print("Plot for K=[1,2,3,5,10] with 1000 Training data & 1000 validation data")
        kSizeAccuracy()
    else:
        print("Exiting")

if __name__ == "__main__":
    main()
#################################################################################################