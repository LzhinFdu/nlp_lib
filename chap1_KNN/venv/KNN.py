from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
'''
K近邻算法：将与样本数据特征最相近（用距离做评估）的K个标签中频次最高的作为样本数据的标签
'''
# def createDateSet():
#     group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
#     labels = ['A', 'A', 'B', 'B']
#     return group, labels

def classify0(inX, dataSet, labels, k):
    '''
    使用欧式距离公式，平方和开平方求距离
    :param inX: test_data
    :param dataSet: train_data
    :param labels: train_label
    :param k: distance
    :return: test_label
    '''
    dataSet_Size = dataSet.shape[0]
    diffMat = tile(inX, (dataSet_Size, 1)) - dataSet
    sq_diffMat = diffMat ** 2
    sq_distances = sq_diffMat.sum(axis=1)
    distances = sq_distances ** 0.5
    #argsort返回数组排序后每个位置上元素所对应的原始索引值
    sorted_distances_indince = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sorted_distances_indince[i]]
        #classCount作为dict类型，若不存在key = ’voteIlabel‘就返回0 + 1 = 1，作为该key的value
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #排序的key为classcount的value,次序为降序，对象是classCount的元组元素
    sorted_classCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_classCount[0][0]

def text2matrix(filename):
    file_data = open(filename)
    line_arr = file_data.readlines()
    data_num = len(line_arr)
    input_matrix = zeros((data_num, 3))
    label_vector = []
    index = 0
    for line in line_arr:
        line = line.strip()
        #strip函数用于出去字符串首尾的空格
        line = line.split('\t')
        input_matrix[index, :] = line[0:3]
        label_vector.append(int(line[-1]))
        index += 1
    # print(input_matrix)
    # print(label_vector)
    return input_matrix, label_vector

def plt_show(input_matrix, label_vetor):
    '''
    analyze the dataset before training
    :param input_matrix: input
    :param label_vetor: label
    :return: none
    '''
    fig = plt.figure()
    plt_sub1 = fig.add_subplot(111)
    #scatter(row, column, size, color)
    plt_sub1.scatter(input_matrix[:, 0], input_matrix[:, 1], 15.0*array(label_vector), 15.0*array(label_vector))
    plt.show()

def auto_norm(input_matrix):
    input_max = input_matrix.max(0)
    input_min = input_matrix.min(0)
    input_range = input_max - input_min
    new_input = zeros((input_matrix.shape[0], input_matrix.shape[1]))
    new_input = input_matrix - tile(input_min, (input_matrix.shape[0], 1))
    new_input = new_input / tile(input_range, (input_matrix.shape[0], 1))
    return new_input, input_range, input_min

def classify_test(filename, test_rate):
    '''
    calculate the accuracy of this classifier
    :param filename: the file name of the dataset
    :param test_rate: the num rate of test data
    :return: none
    '''
    input_matrix, label_vector = text2matrix(filename)
    input_matrix, input_range, input_min = auto_norm(input_matrix)
    length = len(input_matrix)
    test_lenth = int(test_rate * length)
    error_count = 0.0
    for i in range(test_lenth):
        result_label = classify0(input_matrix[i, :], input_matrix[test_lenth:length, :], label_vector[test_lenth:length], 3)
        print("data {}: the real label is {}, and the predict result is {}".format(i, label_vector[i], result_label))
        if(label_vector[i] != result_label):
            error_count += 1.0;
    print("the error_rate is {}".format(error_count/test_lenth))

def classify_person(filename):
    '''
    you can input the features manually, and get the type of this person
    :param filename: the file name of the dataset
    :return: none
    '''
    result_list = ["not at all", "a liitle", "very much"]
    FFmiles = float(input("how many filer miles per year:"))
    PercentTats = float(input("the persent of your time playing vedio games:"))
    Icecream = float(input("liters of ice_cream consumed per year:"))
    person_array = [FFmiles, PercentTats, Icecream]
    input_matrix, label_vector = text2matrix(filename)
    input_matrix, input_range, input_min = auto_norm(input_matrix)
    person_array = (person_array - input_min)/input_range
    result_index = classify0(person_array, input_matrix, label_vector, 3)
    print("the person is the type of {}".format(result_list[int(result_index)]))

if __name__ == '__main__':
    #plt_show(input_matrix, label_vector)
    #classify_test("datingTestSet2.txt", 0.1)
    classify_person("datingTestSet2.txt")