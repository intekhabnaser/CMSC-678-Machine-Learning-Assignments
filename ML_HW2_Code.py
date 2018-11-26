"""
Name: Intekhab Naser
Campus ID : ZC11577
E-mail: intek1@umbc.edu
Machine Learning HW2: Perceptron and SVM
"""
#===================================================================================================#
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from operator import itemgetter
from random import random,shuffle,sample

#===================================================================================================#
def extract_data(train_images,train_labels,test_images,test_labels,count,num1,num2,shuf):
    images_train, labels_train, images_test, labels_test = [],[],[],[]
    
    #1 denoted by num1 and -1 denoted by num2
    #Training data
    num1_count,num2_count = 0,0
    
    for i in range(len(train_labels)):
        #extract only num1's
        if num1_count < count and train_labels[i] == num1:
            images_train.append(train_images[i])
            labels_train.append([1])
            num1_count += 1
    for i in range(len(train_labels)):
        #extract only num2's
        if num2_count < count and train_labels[i] == num2:
            images_train.append(train_images[i])
            labels_train.append([-1])
            num2_count += 1

    #Testing data
    num1_count,num2_count = 0,0
    
    for i in range(len(test_labels)):
        if num1_count < count and test_labels[i] == num1:
            images_test.append(test_images[i])
            labels_test.append([1])
            num1_count += 1
    for i in range(len(test_labels)):
        #extract only num2's
        if num2_count < count and test_labels[i] == num2:
            images_test.append(test_images[i])
            labels_test.append([-1])
            num2_count += 1
    
    if shuf == 1:
	    #shuffle training data
	    train = list(zip(images_train,labels_train))
	    shuffle(train)
	    images_train, labels_train = zip(*train)

	    #shuffle testing data
	    test = list(zip(images_test,labels_test))
	    shuffle(test)
	    images_test, labels_test = zip(*test)

    return images_train,labels_train,images_test,labels_test

#===================================================================================================#
def perceptron(train_data,train_label,test_data,test_label,iteration):
    
    weights = np.zeros(28*28)

    bias = 0
    for i in range(0,iteration):
        for j in range(0,len(train_data)):
            predict = prediction(train_data[j],weights,bias)
            if predict != train_label[j][0]:
                weights += np.multiply(train_data[j],train_label[j])
                bias += train_label[j][0]

    return accuracy(test_data,test_label,weights,bias),weights

#===================================================================================================#
def svm(train_images,train_labels,test_images,test_labels,epochs, num1 = None, num2 = None):
    weight = np.zeros(28*28)
    for epoch in range(1,epochs):
        for i in range(1,len(train_images)):
            learningRate = 1/i
            C = 0.00001
            if train_labels[i][0]*np.dot(train_images[i], weight) < 1:
                weight += learningRate*((train_images[i]*train_labels[i]) + (-2 * C * weight))
            else:
                weight += learningRate * (-2 * C * weight)
    if num1 is not None and num2 is not None:
        correct_num1, correct_num2, total_num1, total_num2 = get_count(test_images, test_labels, weight, num1, num2)
        return correct_num1, correct_num2, total_num1, total_num2
    return accuracy(test_images,test_labels,weight, 0), weight

#===================================================================================================#
def get_count(test_images, test_labels, weight, num1, num2):
    #count for correct predictions in num1
    total_num1, total_num2, correct_num1, correct_num2 = 0 ,0 , 0 , 0
    for i in range(len(test_labels)):
        if test_labels[i] == [1]:
            total_num1 += 1
            if prediction(test_images[i], weight, 0) == 1:
                correct_num1 += 1
        else:
            total_num2 += 1
            if prediction(test_images[i], weight, 0) == -1:
                correct_num2 += 1
    return correct_num1, correct_num2, total_num1, total_num2

#===================================================================================================#
def prediction(inputs,weights,bias):
    activation = 0
    for input,weight in zip(inputs,weights):
        activation += input*weight + bias
    if activation > 0:
        return 1
    else:
        return -1
#===================================================================================================#
def accuracy(testdata,actual_label,weights,bias):
    correct = 0
    for i in range(len(testdata)):
        pred = prediction(testdata[i],weights,bias)
        if pred == actual_label[i][0]: correct += 1
    return correct/float(len(testdata))*100

#===================================================================================================#
def accuracy_iteration(train_data,train_label,test_data,test_label,iteration,num1,num2,svm1):

    x = [i for i in range(iteration)]
    y = []

    if svm1 == 1:
        for i in range(0,iteration):
	        y.append(svm(train_data,train_label,test_data,test_label,i)[0])

        plt.ylim(0,100)
        plt.plot(x,y)
        plot_name = "svm_accuracy_iteration%s_%s%s.png" %(iteration,num1,num2)
        plt.savefig(plot_name)
        plt.close()

    else:
	    for i in range(0,iteration):
		    y.append(perceptron(train_data,train_label,test_data,test_label,i)[0])

	    plt.ylim(0,100)
	    plt.plot(x,y)
	    plot_name = "accuracy_iteration%s_%s%s.png" %(iteration,num1,num2)
	    plt.savefig(plot_name)
	    plt.close()

#===================================================================================================#
def visualize_weight_vector(weights,num1,num2):
    weight_pos, weight_neg = [], []

    for i in range(len(weights)):
        if weights[i] >= 0:
            weight_pos.append(weights[i])
        else:
            weight_pos.append(0)

    for i in range(len(weights)):
        if weights[i] <= 0:
            weight_neg.append(abs(weights[i]))
        else:
            weight_neg.append(0)

    pos_weight,neg_weight = [],[]

    pos_weight = np.asarray(weight_pos).reshape((28,28), order = 'F')
    neg_weight = np.asarray(weight_neg).reshape((28,28), order = 'F')

    plt.imshow(pos_weight,'gray_r')
    plot_name = "weight_%s.png"%(num1)
    plt.savefig(plot_name)
    plt.close()

    plt.imshow(neg_weight,'gray_r')
    plot_name = "weight_%s.png"%(num2)
    plt.savefig(plot_name)
    plt.close()

#===================================================================================================#
def get_score(train_images,train_labels,test_images,test_labels,iteration,num1,num2):
    weights =  perceptron(train_images,train_labels,test_images,test_labels,iteration)[1]
    weight_pos, weight_neg = [], []

    for i in range(len(weights)):
        if weights[i] > 0:
            weight_pos.append(weights[i])
            weight_neg.append(0)
        elif weights[i] < 0:
            weight_pos.append(0)
            weight_neg.append(weights[i])
        elif weights[i] == 0:
        	weight_pos.append(weights[i])
        	weight_neg.append(weights[i])

    pos_test, neg_test = [], []

    for i in range(len(test_labels)):
        if test_labels[i][0] == num1:
            pos_test.append(test_images[i])
        else:
            neg_test.append(test_images[i])
    #calculate score for num1 images
    score_pos = []
    for i in range(len(pos_test)):
        score = 0
        for j in range(len(pos_test[i])):
            score += abs(weight_pos[j] - pos_test[i][j])
        score_pos.append(score)
    #calculate score for num2 images
    score_neg = []
    for i in range(len(neg_test)):
        score = 0
        for j in range(len(neg_test[i])):
            score += abs(weight_neg[j] - neg_test[i][j])
        score_neg.append(score)

    pos_test_score = list(zip(pos_test,score_pos))
    pos_test_score = sorted(pos_test_score,key=itemgetter(1),reverse = True)
    #20 best num1
    best_pos = []
    best_pos_arr = pos_test_score[0:20]
    for i in range(len(best_pos_arr)):
        best_pos.append(best_pos_arr[i][0])

    worst_pos_arr = pos_test_score[-21:-1]
    worst_pos = []
    for i in range(len(worst_pos_arr)):
        worst_pos.append(worst_pos_arr[i][0])

    neg_test_score = list(zip(neg_test,score_neg))
    neg_test_score = sorted(neg_test_score,key = itemgetter(1),reverse = True)
    #20 best num2
    best_neg = []
    best_neg_arr = neg_test_score[0:20]
    for i in range(len(best_neg_arr)):
        best_neg.append(best_neg_arr[i][0])
    #20 worst num2
    worst_neg = []
    worst_neg_arr = neg_test_score[-21:-1]
    for i in range(len(worst_neg_arr)):
        worst_neg.append(worst_neg_arr[i][0])

    #image plot for num1 best 20
    for i in range(len(best_pos)):
        temp = np.asarray(best_pos[i]).reshape((28,28), order = 'F')
        plt.subplot(4, 5, i + 1)
        plt.imshow(temp,'gray_r')
    plot_name = "best_20_%s.png"%(num1)
    plt.savefig(plot_name)
    plt.close()

    #image plot for num1 worst 20
    for i in range(len(worst_pos)):
        temp = np.asarray(worst_pos[i]).reshape((28,28), order = 'F')
        plt.subplot(4,5,i+1)
        plt.imshow(temp,'gray_r')
    plot_name = "worst_20_%s.png"%(num1)
    plt.savefig(plot_name)
    plt.close()

    #image plot for num2 best 20
    for i in range(len(best_neg)):
        temp = np.asarray(best_neg[i]).reshape((28,28), order = 'F')
        plt.subplot(4, 5, i + 1)
        plt.imshow(temp,'gray_r')
    plot_name = "best_20_%s.png"%(num2)
    plt.savefig(plot_name)
    plt.close()

    #image plot for num2 worst 20
    for i in range(len(worst_neg)):
        temp = np.asarray(worst_neg[i]).reshape((28,28), order = 'F')
        plt.subplot(4,5,i+1)
        plt.imshow(temp,'gray_r')
    plot_name = "worst_20_%s.png"%(num2)
    plt.savefig(plot_name)
    plt.close()

#===================================================================================================#
def random_flip(train_data,train_label,test_data,test_label,iteration):
    index = sample(range(1000), 100)

    for i in index:
        if train_label[i] == [1]:
            train_label[i][0] = -1
        else:
            train_label[i][0] = 1
    x = [i for i in range(iteration)]
    y = []
    for i in range(iteration):
        y.append(perceptron(train_data,train_label,test_data,test_label,i)[0])
    plt.ylim(0,100)
    plt.plot(x,y)
    plt.savefig('accuracy_random_flip.png')
    plt.close()

    get_score(train_data,train_label,test_data,test_label,20,1,6)
    accuracy = perceptron(train_data,train_label,test_data,test_label,iteration)[0]
    weight = perceptron(train_data,train_label,test_data,test_label,iteration)[1]

    return accuracy,weight
    # return perceptron(train_data,train_label,test_data,test_label,iteration)[0]

#===================================================================================================#
def data_visualization(train_data,train_label,test_data,test_label,iteration,num1,num2 = None):
	x = [i for i in range(iteration)]
	y = []

	if num2 is None:
		for i in range(0,iteration):
			y.append(svm(train_data,train_label,test_data,test_label,i)[0])
		print("The values of y are: ", y)
		plt.ylim(0,100)
		plt.plot(x,y)
		plot_name = "svm_multiclass_accuracy_iteration%s.png" %(num1)
		plt.savefig(plot_name)
		plt.close()
	else:
		for i in range(0,iteration):
			y.append(perceptron(train_data,train_label,test_data,test_label,i)[0])
		print("The values of y are: ", y)
		plt.ylim(0,100)
		plt.plot(x,y)
		plot_name = "sorted_accuracy_iteration%s%s.png" %(num1,num2)
		plt.savefig(plot_name)
		plt.close()

#===================================================================================================#
def extract_multiclass_data(train_images,train_labels,test_images,test_labels,count,num1):
    images_train, labels_train, images_test, labels_test = [],[],[],[]
    
    #1 denoted by num1 and -1 denoted by other 
    #Training data
    num1_count,num2_count = 0,0
    
    for i in range(len(train_labels)):
        #extract only num1's
        if num1_count < count and train_labels[i] == num1:
            images_train.append(train_images[i])
            labels_train.append([1])
            num1_count += 1
        #extract only other's
        if num2_count < count and train_labels[i] != num1:
            images_train.append(train_images[i])
            labels_train.append([-1])
            num2_count += 1

    #Testing data
    num1_count,num2_count = 0,0
    
    for i in range(len(test_labels)):
        if num1_count < count//2 and test_labels[i] == num1:
            images_test.append(test_images[i])
            labels_test.append([1])
            num1_count += 1
        #extract only other's
        if num2_count < count//2 and test_labels[i] != num1:
            images_test.append(test_images[i])
            labels_test.append([-1])
            num2_count += 1
    
    #shuffle training data
    train = list(zip(images_train,labels_train))
    shuffle(train)
    images_train, labels_train = zip(*train)

    #shuffle testing data
    test = list(zip(images_test,labels_test))
    shuffle(test)
    images_test, labels_test = zip(*test)

    return images_train,labels_train,images_test,labels_test

#===================================================================================================#
def getConfusionMatrix(M):
    confusion = [[0 for j in range(10)] for i in range(10)]
    for i in range(10):
        for j in range(10):
            if i != j:
                train_images,train_labels,test_images,test_labels = extract_data(M['images_train'],M['labels_train'],M['images_test'],M['labels_test'],500,i,j,1)
                train_images = np.asarray(train_images)
                test_images = np.asarray(test_images)
                correct_num1, correct_num2, total_num1, total_num2 = svm(train_images, train_labels, test_images, test_labels, 10, i, j)
                confusion[i][i] += correct_num1
                confusion[i][j] += total_num1 - correct_num1
    for con in confusion:
        temp = sum(con)
        for i in range(len(con)):
            con[i] = round(con[i] / temp,3)
    return confusion

#===================================================================================================#
def getPrecision(confusion):
    relevant,retrieved = 0,0
    for i in range(len(confusion)):
        for j in range(len(confusion[i])):
            retrieved += confusion[i][j]
            if i == j:
                relevant += confusion[i][j]
    return round(relevant/retrieved,5)

#===================================================================================================#
def getIncorrectImages(test_images,test_labels, weight):
    incorrect_images = []
    for i in range(len(test_images)):
        if test_labels[i] == [1]:
            if prediction(test_images[i], weight, 0) == -1:
                incorrect_images.append(test_images[i])
    return incorrect_images

#===================================================================================================#
def getMaxIncorrectImage(incorrect_images, weight):
    scores = []
    for image in incorrect_images:
        score = 0
        for i in range(len(image)):
            score += abs(weight[i] - image[i])
        scores.append(score)
    ind = scores.index(max(scores))
    return incorrect_images[ind]
#===================================================================================================#
def main():

	while 1:
	    #Loading the data
	    M = loadmat('MNIST_digit_data.mat')

	    images_train,labels_train,images_test,labels_test = extract_data(M['images_train'],M['labels_train'],M['images_test'],M['labels_test'],500,1,6,1)
	    images_train = np.asarray(images_train)
	    images_test = np.asarray(images_test)

	    print("\n(1) Perceptron\n(2) SVM\n(Enter any other key to Exit!)\n")
	    model = input("#####=====Select Your Model=====#####\n")
	    if model == '1':
	    	while 1:
	    		#===================================================================================================#
	    		print("\n(a) Plot Accuracy for Classifying digits 1 and 6 w.r.t Number of Iterations\n(b) Visualize learned model in image form for digits 1 and 6\n(c) Visualize 20 best and worst scoring images for each class\n(d) Randomly flip label for 10%\ of training data\n(e) Visualize Accuracy for sorted training data\n(f) Accuracy plot with 10 training data and 1000 training data\n(Press Any other key to Exit)")
	    		#===================================================================================================#
	    		choice = input("*****=====Choose an Option=====*****\n")
	    		if choice == 'a':
	    			accuracy1 = perceptron(images_train,labels_train,images_test,labels_test,20)[0]
	    			print("Accuracy for classifying digits 1 and 6 is:",accuracy1)
	    		#===================================================================================================#
	    			accuracy_iteration(images_train,labels_train,images_test,labels_test,20,1,6,0)
	    			print("Accuracy-iteration plot for digits 1 and 6 ploted!")
	    		#===================================================================================================#
	    		elif choice == 'b':
	    			weight = perceptron(images_train,labels_train,images_test,labels_test,20)[1]
	    			visualize_weight_vector(weight,1,6)
	    			print("Learned model for digits 1 and 6 plotted!")
	    		#===================================================================================================#
	    		elif choice == 'c':
	    			print("Image plot for best and worst 20 images")
	    			get_score(images_train,labels_train,images_test,labels_test,20,1,6)
	    		#===================================================================================================#
	    		elif choice == 'd':
	    			accuracy_random, weight = random_flip(images_train,labels_train,images_test,labels_test,20)
	    			print("Accuracy for classifying digits 1 and 6 with 10%\ random flip",accuracy_random)
	    			print("Accuracy plot with 10%\ error plotted!")
	    			visualize_weight_vector(weight,1,6)
	    			print("Learned model with 10%\ error is plotted for digits 1 and 6!")

	    		#===================================================================================================#
	    		elif choice == 'e':
	    			images_train_s,labels_train_s,images_test_s,labels_test_s = extract_data(M['images_train'],M['labels_train'],M['images_test'],M['labels_test'],500,1,6,0)
	    			images_train_s = np.asarray(images_train_s)
	    			images_test_s = np.asarray(images_test_s)
	    			data_visualization(images_train_s,labels_train_s,images_test_s,labels_test_s,20,1,6)
	    			print("Accuracy plot with sorted data plotted!")
	    		#===================================================================================================#
	    		elif choice == 'f':
	    			images_train,labels_train,images_test,labels_test = extract_data(M['images_train'],M['labels_train'],M['images_test'],M['labels_test'],5,1,6,1)
	    			images_train = np.asarray(images_train)
	    			images_test = np.asarray(images_test)
	    			accuracy1 = perceptron(images_train,labels_train,images_test,labels_test,20)[0]
	    			print("Accuracy with 10 training examples",accuracy1)
	    			accuracy_iteration(images_train,labels_train,images_test,labels_test,25,1,6,0)
	    			print("Plot for 10 training examples plotted!!")

	    			images_train,labels_train,images_test,labels_test = extract_data(M['images_train'],M['labels_train'],M['images_test'],M['labels_test'],500,1,6,1)
	    			images_train = np.asarray(images_train)
	    			images_test = np.asarray(images_test)
	    			accuracy2 = perceptron(images_train,labels_train,images_test,labels_test,10)[0]
	    			print("Accuracy with 1000 training examples",accuracy2)
	    			accuracy_iteration(images_train,labels_train,images_test,labels_test,30,1,6,0)
	    			print("Plot for 1000 training examples plotted!!")
	    		#===================================================================================================#
	    		else:
	    			break
	    elif model == '2':
	    	while 1:
	    		#===================================================================================================#
	    		print("\n(a) Plot Accuracy for Classifying digits 1 and 6 w.r.t Number of Iterations\n(b) 1-vs-all multi-class SVM\n(c) Confusion Matrix for all 10 digits in MNIST dataset\n(d) Top mistakes along with ground truth labels and predicted labels\n(e) Weight Vector visualization for 1 and 6\n(Press Any other key to Exit)")
	    		#===================================================================================================#
	    		choice = input("*****=====Choose an Option=====*****\n")
	    		if choice == 'a':
	    			accuracy_iteration(images_train,labels_train,images_test,labels_test,20,1,6,1)
	    			print("Accuracy-iteration plot for digits 1 and 6 ploted!")
	    		#===================================================================================================#
	    		elif choice == 'b':
	    			train_images_m,train_labels_m,test_images_m,test_labels_m = extract_multiclass_data(M['images_train'],M['labels_train'],M['images_test'],M['labels_test'],1000,1)
	    			train_images_m = np.asarray(train_images_m)
	    			test_images_m = np.asarray(test_images_m)
	    			data_visualization(train_images_m,train_labels_m,test_images_m,test_labels_m,20,1)
	    			print("Accuracy-iteration plot for 1 vs all multi-class SVM!")
	    		#===================================================================================================#
	    		elif choice == 'c':
	    			confusion = getConfusionMatrix(M)
	    			for con in confusion:
	    				print(con)
	    			print("Average Precision is:",getPrecision(confusion))
	    		#===================================================================================================#
	    		elif choice == 'd':
	    			confusion = getConfusionMatrix(M)
	    			print(confusion)
	    			maximum_incorrect = []
	    			for i in range(0,10):
	    				conf = confusion[i][0:i] + confusion[i][i+1:]
	    				maxi = max(conf)
	    				maximum_incorrect.append(conf.index(maxi))
	    			print(maximum_incorrect)
	    			for i in range(len(maximum_incorrect)):
	    				train_images,train_labels,test_images,test_labels = extract_data(M['images_train'],M['labels_train'],M['images_test'],M['labels_test'],500,i,maximum_incorrect[i],1)
	    				train_images = np.asarray(train_images)
	    				test_images = np.asarray(test_images)
	    				weight = svm(train_images,train_labels,test_images,test_labels,20)[1]
	    				incorrect_images = getIncorrectImages(test_images,test_labels, weight)
	    				Incorrect_image = getMaxIncorrectImage(incorrect_images, weight)
	    				
	    				temp = np.asarray(Incorrect_image).reshape((28,28), order = 'F')

	    				pl = plt.subplot(2,5,i+1)
	    				plt.axis('off')
	    				s = str(i) + ',' + str(maximum_incorrect[i])
	    				pl.set_title(s)
	    				plt.imshow(temp,'gray_r')
	    			plt.savefig('Worst_mistakes.png')
	    			plt.close()
	    			print("Worst mistakes are plotted!")
	    		#===================================================================================================#
	    		elif choice == 'e':
	    			weight = svm(images_train,labels_train,images_test,labels_test,20)[1]
	    			visualize_weight_vector(weight,1,6)
	    			print("Weight visualization images have been plotted!")
	    		#===================================================================================================#
	    		else:
	    			break
	    		#===================================================================================================#
	    else:
	    	break

if __name__ =="__main__":
    main()