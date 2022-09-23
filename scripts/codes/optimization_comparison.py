import warnings
warnings.filterwarnings("ignore")

# p_list generator
from distutils.log import error
from math import exp
import math
import random
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support

# Make list of 25 binary lists of dim 512x1
def make_random_p_list(D, N = 25):
    p_list = [[0 for i in range(D)] for j in range(N)]
    for p in p_list:
        for i in range(len(p)):
            if (np.random.rand() > 0.8):
                p[i] = 1

    # print(sum(p_list[5]))

    with open('p_list.txt', 'w') as file:
        for p in p_list:
            file.write(str(p))
            file.write('\n')
            # print(sum(p))
            # print('\n')

    return p_list


# p_list = make_random_p_list()

# generate subset of p_list
def generate_subset(p_list, N = 25):
    for i in range(N):
        data = np.load('../data/datasets/tda_features/X_train_tda.npy', allow_pickle=True)
        data = pd.DataFrame(data=data)
        #data = data.iloc[:, 1:]
        data.columns = [1 for i in range(728)]
        sub_data = data.iloc[: , data.columns == p_list[i]]
        sub_data.to_csv('subsets/subset_'+str(i)+'.csv')

    return sub_data


# generate test subset
def generate_test_subset(p_list, N = 25):
    for i in range(N):
        data = np.load('../data/datasets/tda_features/X_test_tda.npy', allow_pickle= True)
        data = pd.DataFrame(data=data)
        #data = data.iloc[:, 1:]
        data.columns = [1 for i in range(728)]
        sub_data = data.iloc[: , data.columns == p_list[i]]
        sub_data.to_csv('test_subsets/test_subset_'+str(i)+'.csv')

    return sub_data

#  calculating accuracy
def calculate_accuracy(index):
    #accuracy_vector = []
    #score_list = []

    train_feature = pd.read_csv('subsets/subset_' + str(index) + '.csv', header=None)
        # train_feature = pd.read_csv('data.csv', header=None)
    train_feature = train_feature.iloc[1:, 1:]
        # Discover, visualize, and preprocess data using pandas if needed.
    train_feature = train_feature.to_numpy()

    test_feature = pd.read_csv('test_subsets/test_subset_' + str(index) + '.csv', header=None)
        # test_feature = pd.read_csv('test.csv', header=None)
    test_feature = test_feature.iloc[1:, 1:]
        # Discover, visualize, and preprocess data using pandas if needed.
    test_feature = test_feature.to_numpy()


    train_label=np.load('../data/datasets/y_train.npy', allow_pickle= True)
        # for i in range(77):
        #     train_label.append(1)
        # for k in range(77,156):
        #     train_label.append(2)
        
    test_label= np.load('../data/datasets/y_test.npy', allow_pickle=True)
        # for i in range(8):
        #     test_label.append(1)
        # for k in range(8,27):
        #     test_label.append(2)

    print(f'Train Feature Shape: {train_feature.shape}, Test Feature Shape: {test_feature.shape}' + '\n')
    print(f'Train label shape: {train_label.shape}, Test Label Shape: {test_label.shape}')

    rbf_svc = svm.SVC(kernel='rbf')   
    rbf_svc.fit(train_feature, train_label)
    predict=rbf_svc.predict(test_feature)
        # for i in range(len(train_label)):
        #     print(predict[i]-train_label[i])

        # score is mean accuracy
    #print(test_label.shape)
    print(f'Predict vector shape: {predict.shape}' + '\n')
    accuracy = rbf_svc.score(test_feature, test_label)
        # with open('accuracy.txt', 'a') as file:
        #     file.write(str(accuracy))
        #     file.write('\n')

    performance = precision_recall_fscore_support(test_label, predict, average='macro')

    # with open('accuracy.txt', 'a') as file:
    #     file.write('best ' + str(max(accuracy_vector)))
    #     file.write('\n')
    #     file.write('worst ' + str(min(accuracy_vector)))
    #     file.write('\n\n')
    print(f'Score : {accuracy}' + '\n')
    return accuracy, performance

# Difference of two lists
def list_difference_and_square(a, b):
    diff_and_square = []

    for i in range(len(a)):
        diff_and_square.append((a[i] - b[i])**2)

    return diff_and_square


def check_p_list_is_zero(p_list):
    _list = []
    for index in range(len(p_list)):
        if sum(p_list[index]) == 0:
            _list.append(index)

    return _list


# Optimize
def optimize(N, D, p_list, max_iter=10, Rpower=1, FCheck=True):

    F_best = 0 #The highest fitness value in all iteration
    F_best_performance = 0
    for x in range(max_iter):
        p_list_copy = p_list.copy()

        subset = generate_subset(p_list, N)
        test_subset = generate_test_subset(p_list, N)
        
        performance = [0] * N
        #score_list = np.zeros(N)
        #accuracy_vector = np.zeros(N)
        fitness_vector = np.zeros(N)
        best = 0
        worst = 0

        for i in range(N):
            fitness_vector[i], performance[i] = calculate_accuracy(i)
        
        #fitness_vector = accuracy_vector.copy()
        best = max(fitness_vector)
        worst = min(fitness_vector)

        max_index = np.argmax(fitness_vector)
        max_performance = performance[max_index]
            

        if best > F_best:
            F_best = best
            best_feature_subset = p_list[max_index]
            F_best_performance = performance[max_index]
            with open('bestFS.txt', 'w') as file:
                file.write('Best Feature Subset Found at: \n')
                file.write(f'Iteration {x} and population index {max_index}' + '\n')
                file.write(f'Fitness Value (Accuracy): {F_best}' + '\n')
                file.write('Performance:' + str(F_best_performance) + '\n')
                file.write('Feature Subset: ' + str(best_feature_subset))

            

        print("best: ", best, "worst: ", worst, "FBest: ", F_best)

        # calculation of charge
        print('Charge calculation initiation ...')
        q = np.zeros(N)
        qi_sum = 0
        
        for qi in range(len(q)):
            q[qi] = math.exp((fitness_vector[qi]-worst)/(best-worst))
            qi_sum += q[qi]

        for qi in q:
            qi = qi / qi_sum

        print('Charge calculation finished.')

        print('Coulombs constant, electric field calculation initiated ...')
        # calculation of Coulomb's Constant
        s = 500 * exp(-30*x/10)

        # calculation of total electric field
        RNorm = 2
        R = np.zeros(N)

        # total electric field
        E = np.zeros([N,D])
    
        for i in range(N):
            for j in range(N):
                diff_and_square = list_difference_and_square(p_list[i], p_list[j])
                R = math.sqrt(sum(diff_and_square))
                R = R**Rpower

                # print(R)
                # print(np.finfo(float).eps)

                for k in range(D):
                    # total_electric_field = electric_field + (random_val * q[i] * q[j] * (p_list[i] - p_list[j])) / R[i][j] + np.finfo(float).eps
                    # print( q[j] )
                    E[i][k] = E[i][k] + (random.uniform(0,1) * q[j] * (p_list[j][k]-p_list[i][k]) / (R+np.finfo(float).eps) )

        print('Finished')

        print('Updation of acceleration, speed and positions:')
        # Calc of acceleration
        A = E * s

        # Calc of velocity
        V = np.zeros([N,D])
        V = random.uniform(0,1)*V+A

        p_list = p_list + V

        print('Finished.')


        print('Final probability calculation and binary string formation.')
        random_key = random.uniform(0,1)

        for i in range(len(p_list)):
            for j in range(len(p_list[i])):
                # try:
                # print(V[i][j])
                try:
                    probability_pi = 1 / (1 + (math.exp(0-V[i][j])))
                    #try:
                    #ans = math.exp(200000)
                except OverflowError:
                    #ans = float('inf')
                    probability_pi = 1 / (1 + (float('inf')))
                if probability_pi >  random_key:
                    p_list[i][j] = 1
                else:
                    p_list[i][j] = 0

        #Flag to check whether any feature subset has all zeroes, which means no features are selected.
        print('Finished.')


        flag = 0
        zeros = check_p_list_is_zero(p_list)
        for i in zeros:
            p_list[i] = p_list_copy[i]
            flag = 1
        
        # print(f"Iteration {x} Complete")
        # continue

        # new_fitness_vector = np.zeros(N)
        # new_performance_list = [0] * N

        # subset = generate_subset(p_list)
        # test_subset = generate_test_subset(p_list)
        # for i in range(N):
        #     new_fitness_vector, new_performance_list[i] = calculate_accuracy(i)

        if True:
            with open('accuracy.txt', 'a') as file:
                file.write(f'Iteration {x} Statistics:' + '\n\n')
                file.write('Fitness Values: \n')
                for item in fitness_vector:
                    file.write(str(item) + '\n')
                file.write('best ' + str(max(fitness_vector)) + '\n')
                file.write('worst ' + str(min(fitness_vector)) + '\n')
                file.write('Best fitness performance:' + str(max_performance) + '\n\n')
                # if x == (max_iter - 1):
                #     #file.write('score ' + str(max(score_list)) + '\n')
                #     file.write('performance ' + str(performance) + '\n')

        # for i in range(len(p_list)):
        #     if fitness_vector[i] > new_fitness_vector[i]:
        #         p_list[i] = p_list_copy[i]

        print(f"Iteration {x} Complete")




p_list = make_random_p_list(N = 25, D = 728)
print('Random population created')
while all(element == 0 for element in p_list):
    p_list = make_random_p_list(N=25, D=728)
    print('Since, random population all contains zeroes, again, making random population')

print('Beginning Optimisation')
optimize(N=25,D=728,p_list=p_list)
                





        
