import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import log
import types
import sklearn
from sklearn import tree
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals.six import StringIO
import pydot




"""
原始文件的12列分别为：

PassengerId
Survived            -1
Pclass              -2
fName
sName
Sex                 -5
Age                 -6
SibSp               -7
Parch               -8
Ticket
Fare                -10
Cabin
Embarked            -12
预处理之后保留的列为：
Survived                 0,1是否
Pclass                   1,2,3船舱等级
Sex                      0:male, 1:female
Age                      1（0-18）表示小孩 2（18-50）表示中年 3（>50）表示老年
slibSp                   0  1（1-2） 2（3-5） 3（>5）
Parch                    0  1（1-2） 2（3-5） 3（>5）
Fare                     1（0-20） 2（20-50） 3（50-100） 4（>100）
Embarked                 0:Q，1:S，2:C
"""




class decisionTree(object):
    """
    docstring for .showFeature
    """
    def __init__(self):
        super().__init__()

    def data_pre_handle(self, filepath1, filepath2):                           #注意训练集比测试集多一列，是否生存的列
        #df = pd.read_csv(self.filepath)
        with open(filepath1, 'r') as f:
            lines = f.readlines()
        with open(filepath2, 'w') as f:
            flag = 0
            number = 0
            for line in lines:
                if flag!=0:
                    cloumns = line.split(',')
                    print (cloumns)
                    #Survived = cloumns[1]
                    #Pclass = cloumns[2]
                    Pclass = cloumns[1]
                    if len(Pclass) == 0:        #Pclass缺失值为1
                        Pclass = "1"
                    #Sex = cloumns[5]
                    Sex = cloumns[4]
                    if len(Sex) == 0:
                        Sex = "male"             #Sex缺失值为Male
                    if Sex == "male":
                        Sex = "0"
                    else:
                        Sex = "1"
                    """
                    if len (cloumns[6]) == 0:
                        cloumns[6] = 10          #Age缺失值为10 ,即为小孩
                    Age = float(cloumns[6])
                    """
                    if len (cloumns[5]) == 0:
                        cloumns[5] = 10          #Age缺失值为10 ,即为小孩
                    Age = float(cloumns[5])
                    if Age<18:
                        Age = "1"
                    elif Age<50:
                        Age = "2"
                    else:
                        Age = "3"
                    """
                    if len(cloumns[7]) == 0:
                        cloumns = 1                #Slibsp 缺失值为1
                    Slibsp = int(cloumns[7])
                    """
                    if len(cloumns[6]) == 0:
                        cloumns = 1                #Slibsp 缺失值为1
                    Slibsp = int(cloumns[6])
                    if Slibsp <= 0:
                        Slibsp = "0"
                    elif Slibsp <= 2:
                        Slibsp = "1"
                    elif Slibsp <= 5:
                        Slibsp = "2"
                    else:
                        Slibsp = "3"
                    """
                    if len(cloumns[8]) == 0:
                        Parch = 1                   #Parch缺失值为1
                    Parch = int(cloumns[8])
                    """
                    if len(cloumns[7]) == 0:
                        Parch = 1                   #Parch缺失值为1
                    Parch = int(cloumns[7])
                    if Parch <= 0:
                        Parch = "0"
                    elif Parch<=2:
                        Parch = "1"
                    elif Parch <=5:
                        Parch = "2"
                    elif Parch > 5:
                        Parch = "3"
                    """
                    if len(cloumns[10]) == 0:
                        cloumns[10] = 10                  #Fare缺失值为10美元
                    Fare = float(cloumns[10])
                    """
                    if len(cloumns[9]) == 0:
                        cloumns[9] = 10                  #Fare缺失值为10美元
                    Fare = float(cloumns[9])
                    if Fare<20:
                        Fare = "1"
                    elif Fare<50:
                        Fare = "2"
                    elif Fare<100:
                        Fare = "3"
                    else:
                        Fare = "4"
                    #Embarked = cloumns[12]
                    Embarked = cloumns[11]
                    if len(Embarked) == 0:
                        Embarked = "Q"                            #缺失值为Q
                    if Embarked.strip() == "Q":
                        Embarked = "0"
                    elif Embarked.strip() == "S":
                        Embarked = "1"
                    else:
                        Embarked = "2"
                    #newline = Survived+"\t"+Pclass+"\t"+Sex+"\t"+Age+"\t"+Slibsp+"\t"+Parch+"\t"+Fare+"\t"+Embarked+"\n"
                    newline = Pclass+"\t"+Sex+"\t"+Age+"\t"+Slibsp+"\t"+Parch+"\t"+Fare+"\t"+Embarked+"\n"
                    f.write(newline)
                    number = number + 1
                flag = 1
        print (number)
    """
    """
    #建立决策树
    #case_set: 目前集合中的元素 行形式为Survived+"\t"+Pclass+"\t"+Sex+"\t"+Age+"\t"+Slibsp+"\t"+Parch+"\t"+Fare+"\t"+Embarked,多行
    #attribute_set：目前剩余可选的属性集合
    """
    def build_tree(self, case_set, attribute_set):
        classList = {}                                          #classList作为目前集合中包括的类别集合,key类别，value数量                                       #
        for line in case_set:
            if line.split("\t")[0] not in classList:
                classList[line.split("\t")[0]] = 1
            else:
                classList[line.split("\t")[0]] = classList[line.split("\t")[0]] + 1
        if len(classList) == 1:                                    #目前集合中只有一类，不用再进分裂了，作为叶节点
            for key in classList:
                return key
        if len(attribute_set) == 0:                                  #目前没有可选属性了，选择目前集合中数量最多的类别作为叶节点的类别
            value_key_pairs = [(num, classlabel) for num, classlabel in classList.items()]
            value_key_pairs.sort()
            #print (value_key_pairs)
            #print (value_key_pairs[0][0])
            return value_key_pairs[0][0]
        bestFeature = self.chooseBestFeature(case_set, attribute_set)      #选择最优的分裂特征
        mytree = { bestFeature: {} }
        print ("本次选择的最优特征是：")
        print (bestFeature)
        featValues = [line.split("\t")[bestFeature] for line in case_set]
        uniqueValues = set(featValues)
        subattrbute_set = attribute_set
        subattrbute_set.remove(bestFeature)
        for value in uniqueValues:
            sub_case_set = [line for line in case_set if line.split("\t")[bestFeature] == value]
            mytree[bestFeature][value] = self.build_tree(sub_case_set, subattrbute_set)
        print ("决策树已经建立完毕！！！")
        return mytree




    """
    #chooseBestFeature:选择最优的分裂属性或者特征
    #case_set:目前集合中的实例，行形式为Survived+"\t"+Pclass+"\t"+Sex+"\t"+Age+"\t"+Slibsp+"\t"+Parch+"\t"+Fare+"\t"+Embarked,多行
    #attribute_set：目前剩余可选的属性集合
    """


    def chooseBestFeature(self, case_set, attribute_set):
        Info_gains = {}                                        #Info_gains作为一个dict存储,key为可选属性集合中的属性，value为信息增益比率
        total_num = len(case_set)                              #当前集合的元素个数
        base_entropy = self.calcEntropy(case_set)                   #分裂前当前集合的信息熵
        #print (base_entropy)
        for attribute in attribute_set:                        #对每个属性计算信息增益比率
            attr_value_set = set()                                 #这个属性对应的所有可能取值的集合，set
            new_entropy = 0.0
            split_info_index = 0.0                               #信息增益比率的分母，熵是分子
            for line in case_set:
                if line.split("\t")[attribute] not in attr_value_set:
                    attr_value_set.add(line.split("\t")[attribute])
            for attr_value in attr_value_set:
                sub_case_set = [line for line in case_set if line.split("\t")[attribute] == attr_value]   #当前属性每个取值之后的子集合list
                prob = float(len(sub_case_set))/total_num                                                 #子集占总的比例
                new_entropy = new_entropy + prob*self.calcEntropy(sub_case_set)                                 #子集的信息熵
                split_info_index = split_info_index - prob*log(prob, 2)                                      #分裂信息指数
            if split_info_index <=0:
                Info_gains[attribute] = 0.0
            else:
                Info_gains[attribute] = (base_entropy  - new_entropy)/split_info_index                                           #计算属性的信息增益
        print (Info_gains)
        max = 0
        bestFeature = -1
        for attribute in Info_gains:
            if Info_gains[attribute] >= max:
                max = Info_gains[attribute]
                bestFeature = attribute
        return bestFeature                                                                     #返回最优的属性




    """
    #计算信息熵
    #case_set表示待计算信息熵的实例集合
    #224(0.406),   328(0.594)
    """
    def calcEntropy(self, case_set):
        class_value_pairs = {}                                               #dict，Key表示类别，value表示数量
        total_case = len(case_set)
        for line in case_set:
            if line.split("\t")[0] not in class_value_pairs:
                class_value_pairs[line.split("\t")[0]] = 1
            else:
                class_value_pairs[line.split("\t")[0]] = class_value_pairs[line.split("\t")[0]] + 1
        #print (class_value_pairs)

        entropy = 0.0
        for key in class_value_pairs:
            prob = float(class_value_pairs[key])/total_case
            entropy = entropy + prob*log(prob, 2)
        return -entropy

    """
    """
    决策树构建说明，sklearn构建决策树可以通过两种方式，一种是将非数值的离散属性转换为数值离散属性（如性别用0和1表示），第二种是onehot的方式重构特征向量，
    如共有两个特征，分别是年龄（老，中，少）和是否是学生（是，否），则对应onehot为(老？，中？，少？，学生？)四维向量，每维的取值只能是0或者1
    sklearn建立的决策树都是二叉的，多叉树可以转换
    """

    def sk_buile_tree(self, case_set):
        header = { "1":"Pclass", "2":"Sex ", "3": "Age", "4": "SlibSp", "5": "Parch", "6": "Fare", "7": "Embarked" }
        X = []
        Y = []
        for line in case_set:
            case = []
            cloumns = line.split("\t")
            for i in range(1,len(cloumns)):
                case.append(cloumns[i].strip())
            X.append(case)
            Y.append(cloumns[0])
        print (X[0])
        #print (Y)
        clf = tree.DecisionTreeClassifier(criterion='entropy')
        clf = clf.fit(X,Y)
        dot_data = StringIO()
        tree.export_graphviz(clf, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph[0].write_pdf("Taitannike.pdf")
        return clf













if __name__ == '__main__':
    ds = decisionTree()
    #ds.data_pre_handle("test.csv", "test_handled.txt")
    with open("train_part_handled.txt", 'r') as f:
        case_set = f.readlines()
    clf = ds.sk_buile_tree(case_set)
    """
    true_class = []
    predict_class = []
    with open("train_part_handled.txt", 'r') as f:
        for line in f.readlines():
            true_class.append(line.split("\t")[0])
            predict_case = []
            for i in range(1, len(line.split("\t"))):
                 predict_case.append(line.split("\t")[i].strip())
            predict_class.append(clf.predict(predict_case))
    right = 0
    total = len(true_class)
    for i in range(len(true_class)):
        if true_class[i] == predict_class[i]:
            right = right + 1
    print ("准确率是：" + str(float(right)/total))
    """
    predict_class = []
    with open("test_handled.txt", 'r') as f:
        for line in f.readlines():
            predict_case = []
            for i in range(len(line.split("\t"))):
                predict_case.append(line.split("\t")[i].strip())
            predict_class.append(clf.predict(predict_case))
    with open("result", 'w') as f:
        for predict in predict_class:
            line = str(predict[0]) + "\n"
            f.write(line)
    print ("预测完毕！！！")

    #print ("预测值是:" + str(predictedY))
