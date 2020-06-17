from numpy import *
import operator
'''
K近邻算法：将与样本数据特征最相近（用距离做评估）的K个标签中频次最高的作为样本数据的标签
'''
def createDateSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    #使用欧式距离公式，平方和开平方求距离
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
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #排序的key为classcount的value,次序为降序，对象是classCount的元组元素
    sorted_classCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_classCount[0][0]


if __name__ == '__main__':
    group, labels = createDateSet()
    label = classify0([0.5, 1.5], group, labels, 3)
    print(label)