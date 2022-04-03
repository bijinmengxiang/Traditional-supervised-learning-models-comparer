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
def calculate_statistics(data,model):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
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
        #如果为bool型进行标准化统计
        if isinstance(all_transported[i], bool) and isinstance(Y[i], bool):
            if Y[i]==True:
                if all_transported[i]==True:
                    tp+=1
                elif all_transported[i]==False:
                    fn+=1
            elif Y[i]==False:
                if all_transported[i]==True:
                    fp+=1
                elif all_transported[i]==False:
                    tn+=1
        #如果不是简单的二分类问题则采用粗略的计算方法    
        else :
            print("暂不支持多分类")
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = (2*precision*recall)/(precision+recall)
    accuracy = (tp+tn)/len(all_transported)
    result = [precision,recall,f1_score,accuracy]
    return result
    #for i in range(len(preds)):
        
#计算平均值
def Get_Average(result):
    sum_precision = 0
    sum_recall = 0
    sum_f1_score = 0
    sum_accuracy = 0
    for item in result:
        sum_precision += item[0]
        sum_recall += item[1]
        sum_f1_score += item[2]
        sum_accuracy += item[3]
    precision = sum_precision/len(result)
    recall = sum_recall/len(result)
    f1_score = sum_f1_score/len(result)
    accuracy = sum_accuracy/len(result)
    result = [precision,recall,f1_score,accuracy]
    
    return result

def randomforestmethod():
    result=[]
    for i in range(n_folds):
        result.append(0)
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
        result[i] = calculate_statistics(datasplit[i],forest_model)
    #print(precision)
    print("随机森林precision："+str(Get_Average(result)[0]))
    print("随机森林recall："+str(Get_Average(result)[1]))
    print("随机森林f1_score："+str(Get_Average(result)[2]))
    print("随机森林accuracy："+str(Get_Average(result)[3]))
    print()
    
def GradientBoostingmethod():
    result=[]
    for i in range(n_folds):
        result.append(0)
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
        result[i] = calculate_statistics(datasplit[i],forest_model)
    #print(precision)
    #print("GradientBoosting算法："+str(Get_Average(result))) 
    print("GradientBoosting算法precision："+str(Get_Average(result)[0]))
    print("GradientBoosting算法recall："+str(Get_Average(result)[1]))
    print("GradientBoosting算法f1_score："+str(Get_Average(result)[2]))
    print("GradientBoosting算法accuracy："+str(Get_Average(result)[3]))
    print()
    
def AdaBoostRegressor():
    result=[]
    for i in range(n_folds):
        result.append(0)
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
        result[i] = calculate_statistics(datasplit[i],forest_model)
    #print(result)
    #print("AdaBoost算法："+str(Get_Average(result)))
    print("AdaBoost算法precision："+str(Get_Average(result)[0]))
    print("AdaBoost算法recall："+str(Get_Average(result)[1]))
    print("AdaBoost算法f1_score："+str(Get_Average(result)[2]))
    print("AdaBoost算法accuracy："+str(Get_Average(result)[3]))
    print()

def BaggingRegressor():
    result=[]
    for i in range(n_folds):
        result.append(0)
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
        result[i] = calculate_statistics(datasplit[i],forest_model)
    #print(precision)
    #print("BaggingRegressor算法："+str(Get_Average(result)))
    print("BaggingRegressor算法precision："+str(Get_Average(result)[0]))
    print("BaggingRegressor算法recall："+str(Get_Average(result)[1]))
    print("BaggingRegressor算法f1_score："+str(Get_Average(result)[2]))
    print("BaggingRegressor算法accuracy："+str(Get_Average(result)[3]))
    print()

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
   
