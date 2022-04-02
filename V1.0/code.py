import pandas as pd 
import random
from sklearn import ensemble
       
#n_folds分割
def spiltDataSet(dataSet, n_folds):
    #计算每个folds的尺寸
    fold_size = int(len(dataSet) / n_folds)
    #dataSet_copy = list(dataSet)
    #print(fold_size)
    dataSet_spilt = []
    #循环n次
    for i in range(n_folds):
        fold = []
        #每次循环使fold中达到目标个数
        while len(fold) < fold_size:  # 这里不能用if，if只是在第一次判断时起作用，while执行循环，直到条件不成立
            #随机选取一个数字作为索引    
            index = random.randrange(len(dataSet))
            fold.append(dataSet.iloc[index]) 
        dataSet_spilt.append(fold)
    return dataSet_spilt

#计算准确率
def calculate_precision(data,model):
    correct=0
    df=pd.DataFrame(data)
    y = df.Transported
    features = ['CryoSleep','HomePlanet',  'VIP', 'Destination','Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
    X = df[features]
    preds = model.predict(X)
    
    all_transported=[]
    for i in range(len(preds)):
        #print(preds[i])
        if preds[i]>0.5:
                all_transported.append(True)        
            #说明在休眠状态下
        else:
            all_transported.append(False)  
    Y = y.values.tolist()
    for i in range(len(all_transported)):
        if all_transported[i] == Y[i]:
            correct += 1
    return correct / float(len(all_transported))
    #for i in range(len(preds)):
        
#计算平均值
def Get_Average(precision):
    sum = 0
    for item in precision:
        sum += item
    return sum/len(precision)

def randomforestmethod():
    precision=[]
    for i in range(n_folds):
        precision.append(0)
    for i in range(len(datasplit)):
        forest_model = ensemble.RandomForestRegressor(random_state=1)
        for j in range(len(datasplit)):
            #如果i=j跳过此次循环作为验证集使用
            if i==j:
                continue
            #print(datasplit[j])
            df=pd.DataFrame(datasplit[j])
            y = df.Transported
            features = ['CryoSleep','HomePlanet',  'VIP', 'Destination','Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
            X = df[features]
            forest_model.fit(X, y)    
        precision[i] = calculate_precision(datasplit[i],forest_model)
    #print(precision)
    print("随机森林："+str(Get_Average(precision)))
    
def GradientBoostingmethod():
    precision=[]
    for i in range(n_folds):
        precision.append(0)
    for i in range(len(datasplit)):
        forest_model = ensemble.GradientBoostingRegressor()
        for j in range(len(datasplit)):
            #如果i=j跳过此次循环作为验证集使用
            if i==j:
                continue
            #print(datasplit[j])
            df=pd.DataFrame(datasplit[j])
            y = df.Transported
            features = ['CryoSleep','HomePlanet',  'VIP', 'Destination','Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
            X = df[features]
            forest_model.fit(X, y)    
        precision[i] = calculate_precision(datasplit[i],forest_model)
    #print(precision)
    print("GradientBoosting算法："+str(Get_Average(precision))) 
    
def AdaBoostRegressor():
    precision=[]
    for i in range(n_folds):
        precision.append(0)
    for i in range(len(datasplit)):
        forest_model = ensemble.AdaBoostRegressor()
        for j in range(len(datasplit)):
            #如果i=j跳过此次循环作为验证集使用
            if i==j:
                continue
            #print(datasplit[j])
            df=pd.DataFrame(datasplit[j])
            y = df.Transported
            features = ['CryoSleep','HomePlanet',  'VIP', 'Destination','Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
            X = df[features]
            forest_model.fit(X, y)    
        precision[i] = calculate_precision(datasplit[i],forest_model)
    #print(precision)
    print("AdaBoost算法："+str(Get_Average(precision)))

def BaggingRegressor():
    precision=[]
    for i in range(n_folds):
        precision.append(0)
    for i in range(len(datasplit)):
        forest_model = ensemble.BaggingRegressor()
        for j in range(len(datasplit)):
            #如果i=j跳过此次循环作为验证集使用
            if i==j:
                continue
            #print(datasplit[j])
            df=pd.DataFrame(datasplit[j])
            y = df.Transported
            features = ['CryoSleep','HomePlanet',  'VIP', 'Destination','Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
            X = df[features]
            forest_model.fit(X, y)    
        precision[i] = calculate_precision(datasplit[i],forest_model)
    #print(precision)
    print("BaggingRegressor算法："+str(Get_Average(precision)))

#初始化
fold_number = input("请输入fold数量:")
n_folds=int(fold_number)
path = input("请输入目标csv路径:")
#path_train = "C:\data\trainCHVD.csv"
path_train=path.replace('\\','/')
data = pd.read_csv(path_train)
data = data.dropna(axis=0)

datasplit = spiltDataSet(data,n_folds)

#定义模型
randomforestmethod()
GradientBoostingmethod()
AdaBoostRegressor()
BaggingRegressor()
   
