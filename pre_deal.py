import pandas as ps
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

















class preDeal(object):
    """docstring for ."""
    def __init__(self, rawdata_filepath, test_filepath):
        super().__init__()
        self.rawdata_filepath = rawdata_filepath
        self.test_filepath = test_filepath
    """
    参数：filepath：原始数据文件路径
    返回值：返回一个dict, key 为列名，value为这列为空的行的总行数
    数据CSV格式
    内容：
    Survived ： 取值0和1,0表示死亡，1表示获救
    Pclass ：乘客的船仓等级
    Name ：乘客姓名
    Sex ：乘客性别
    Age ：乘客年龄                    you null (训练集)   用随机森林来预测，把有年龄的条目用来预测没年龄的年龄
    SibSp ：船上配偶兄妹的人数
    Parch ：船上父母孩子的人数
    Ticket ：票号
    Fare ：票价                       you null (测试集)  同等（出发港口，船舱等级）均值来决定 可以手动，数量少
    Cabin ：乘客船舱号                 you null(训练集)  缺失值不用处理
    Embarked ：出发港口              you null(训练集)    同等（船舱等级，票价，年龄）均值来决定，可以手动，数量少
    """
    def get_dataframe(self):
        train_df = ps.read_csv(self.rawdata_filepath)
        test_df = ps.read_csv(self.test_filepath)
        test_df["Survived"] = np.nan
        TT_df = ps.concat([train_df, test_df])
        #print (len(TT_df)) 1309
        return TT_df

    def showcloumn_null(self, TT_data):
        result = TT_data.isnull().any()                 # 则会判断哪些”列”存在缺失值，返回的是一个类似Dict{列名：true/false}，实际类型为Series
        result = dict(result)                           #转为dict方便遍历
        for key in result:
            if result[key] == True:
                null_row_number = len(TT_data[TT_data[key].isin([np.nan])])
                result[key] = null_row_number
        #print (result)
        return result, TT_data
    """
    将非数值类型的转化为数值类型
    返回的数据为dataframe：
    Survived ： 取值0和1,0表示死亡，1表示获救         数值型
    Pclass ：乘客的船仓等级                           数值型
    Sex ：乘客性别                                    需要变为数值型      male:0  female:1
    Age ：乘客年龄                                   数值型
    SibSp ：船上配偶兄妹的人数                        数值型
    Parch ：船上父母孩子的人数                        数值型
    Fare ：票价                       数值型
    Embarked ：出发港口             需要变为数值型   S:0 C:1 Q:3
    """
    def strTofloat(self, TT_df):
        TT_df = TT_df[["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]

        TT_df["Embarked"] = ps.factorize(TT_df.Embarked)[0]                                                   #同时会破坏空值
        TT_df["Sex"] = ps.factorize(TT_df.Sex)[0]
        return TT_df

    """
    填充缺失的Embark
    参数：TT_df,dataframe 全部的训练集数据含有缺失值的
    返回值，一个填充好Embark的dataframe
    """
    def fill_Embarked(self, TT_df):
        embarked_df = TT_df[["Embarked","Pclass","Fare"]]
        embarked_not_null = embarked_df.loc[(TT_df["Embarked"].isin([0,1,2]))]
        embarked_null = embarked_df.loc[(~TT_df["Embarked"].isin([0,1,2]))]             #NULL会被弄成-1
        print (embarked_null)
        X = embarked_not_null.values[:,1:]                          #所有行的第二开始所有的列

        Y = embarked_not_null.values[:,0]                           #所有行的第一列
        rfr = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
        rfr.fit(X,Y)
        predict_em = rfr.predict(embarked_null.values[:,1:])               #返回np数组
        #print(type(predict_em))
        predict_em[predict_em<0.5] = 0
        predict_em[predict_em>1.5] = 2
        predict_em[abs(predict_em-1)<0.5] = 1
        print (predict_em)
        TT_df.loc[(~TT_df["Embarked"].isin([0,1,2])), "Embarked"] = predict_em
        return TT_df
    """
    填充缺失的Age
    参数：TT_df,dataframe全部的训练集数据含有缺失值的
    返回值，一个填充好AGE的dataframe
    """
    def fill_Age(self, df):
        #choose training data to predict age
        age_df = df[['Age','Sex','Fare', 'Parch', 'SibSp', 'Pclass','Embarked']]
        age_df_notnull = age_df.loc[(df.Age.notnull())]
        age_df_isnull = age_df.loc[(df.Age.isnull())]
        X = age_df_notnull.values[:,1:]
        Y = age_df_notnull.values[:,0]
        #use RandomForestRegressor to train data
        rfr = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
        rfr.fit(X,Y)
        predictAges = rfr.predict(age_df_isnull.values[:,1:])
        #可视化年龄分布
        """
        b_age = np.array(Y)
        a_age = predictAges
        a_age_dis = []
        b_age_dis = []
        for i in range(10):
            a_age_dis.append(len(a_age[abs(a_age-i*10-5)<=5])/len(a_age))
        for i in range(10):
            b_age_dis.append(len(b_age[abs(b_age-i*10-5)<=5])/len(b_age))
        """
        df.loc[(df.Age.isnull()),'Age'] = predictAges
        #plt.plot(a_age_dis, label ="after_fill", color="red")
        #plt.plot(b_age_dis, label= "before_fill", color="blue")
        #plt.legend()
        #plt.show()
        return df
    def get_result_RF(self, df):
        sur_df = df[['Survived','Age','Sex','Fare', 'Parch', 'SibSp', 'Pclass','Embarked']]
        enc = OneHotEncoder()
        enc.fit(sur_df.values[:,1:])
        sur_df_notnull = sur_df.loc[(df.Survived.notnull())]
        sur_df_isnull = sur_df.loc[(df.Survived.isnull())]
        X = sur_df_notnull.values[:,1:]
        Y = sur_df_notnull.values[:,0]
        one_hot_X= enc.transform(X)
        print (one_hot_X.shape)
        #use RandomForestRegressor to train data
        rfr = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
        rfr.fit(one_hot_X,Y)
        predict_X = enc.transform(sur_df_isnull.values[:,1:])
        print (predict_X.shape)
        predictsurs = rfr.predict(predict_X)
        return predictsurs

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    predeal = preDeal("../train.csv", "../test.csv")
    TT_data = predeal.get_dataframe()
    TT_data = predeal.strTofloat(TT_data)
    TT_data = predeal.fill_Embarked(TT_data)
    TT_data = predeal.fill_Age(TT_data)
    predicts = predeal.get_result_RF(TT_data)
    predicts[predicts-0.5<0] = 0
    predicts[predicts-0.5>=0] = 1
    passenger = ps.read_csv("./passenger_elv.csv")
    passenger["Survived"] = predicts
    print (passenger.describe())
    passenger.to_csv("./my_submission_V3.csv", index=False)

    #predeal.fill_Embarked(TT_data)
    #TT_data = predeal.fill_Embarked(TT_data)
