from tkinter import *
from tkinter.filedialog import askopenfilename
import pandas as pd 
import random
from sklearn import ensemble
global filepath

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

def randomforestmethod(datasplit,n_folds):
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
    return Get_Average(result)
    
def GradientBoostingmethod(datasplit,n_folds):
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
    return Get_Average(result)
    
def AdaBoostRegressor(datasplit,n_folds):
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
    return Get_Average(result)

def BaggingRegressor(datasplit,n_folds):
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
    return Get_Average(result)
    
filepath = ""

class ui():
    def set_statistics(self,result,flag):
        
        if flag==1:
            self.ran_precision.set(result[0])
            self.ran_recall.set(result[1])
            self.ran_f1_scores.set(result[2])
            self.ran_accuracy.set(result[3])
        elif flag==2:
            self.gra_precision.set(result[0])
            self.gra_recall.set(result[1])
            self.gra_f1_scores.set(result[2])
            self.gra_accuracy.set(result[3])
        elif flag==3:
            self.ada_precision.set(result[0])
            self.ada_recall.set(result[1])
            self.ada_f1_scores.set(result[2])
            self.ada_accuracy.set(result[3])
        elif flag==4:
            self.bag_precision.set(result[0])
            self.bag_recall.set(result[1])
            self.bag_f1_scores.set(result[2])
            self.bag_accuracy.set(result[3])
        
        
    
    def selectPath(self):
        path_ = askopenfilename()
        self.path.set(path_)
        
    def startreading(self):
        
        filepath = self.path.get()
        n_folds = self.n.get()
        print(filepath,n_folds)
        path_train=filepath.replace('\\','/')
        data = pd.read_csv(path_train)
        data = data.dropna(axis=0)
        datasplit = spiltDataSet(data,int(n_folds))
        
        randomforest_result = randomforestmethod(datasplit,int(n_folds))       
        GradientBoosting_result = GradientBoostingmethod(datasplit,int(n_folds))       
        AdaBoost_result = AdaBoostRegressor(datasplit,int(n_folds))       
        Bagging_result = BaggingRegressor(datasplit,int(n_folds))
        self.set_statistics(randomforest_result,1)
        self.set_statistics(GradientBoosting_result,2)
        self.set_statistics(AdaBoost_result,3)
        self.set_statistics(Bagging_result,4)
   
    def __init__(self,root):
        self.path = StringVar()
        self.n = StringVar()
        #self.conditional = StringVar()
        self.ran_precision = StringVar()
        self.ran_recall = StringVar()
        self.ran_f1_scores = StringVar()
        self.ran_accuracy = StringVar()
        self.gra_precision = StringVar()
        self.gra_recall = StringVar()
        self.gra_f1_scores = StringVar()
        self.gra_accuracy = StringVar()
        self.ada_precision = StringVar()
        self.ada_recall = StringVar()
        self.ada_f1_scores = StringVar()
        self.ada_accuracy = StringVar()
        self.bag_precision = StringVar()
        self.bag_recall = StringVar()
        self.bag_f1_scores = StringVar()
        self.bag_accuracy = StringVar()

        self.label_title = Label(root,text = "传统机器学习方法比较器:",justify=CENTER,font=('黑体',18,'bold','italic')).grid(row = 0)
        self.label_choose = Label(root,text = "数据集选择:",font=('黑体',12,'bold','italic'),anchor="w").grid(row = 2, column = 2)
        self.Button =Button(root, text = "选择", command = self.selectPath,width=20).grid(row = 2, column = 3)
        #self.condition = Label(root,textvariable = self.conditional,font=('黑体',18,'bold','italic')).grid(row = 2, column = 0)
        
        self.label_path = Label(root,text = "N-folds数:",font=('黑体',12,'bold','italic'),anchor="w").grid(row = 3, column = 0)
        self.Entry_n = Entry(root, textvariable = self.n).grid(row = 3, column = 1)
        self.label_path = Label(root,text = "目标路径:",font=('黑体',12,'bold','italic'),anchor="w").grid(row = 3, column = 2)
        self.Entry = Entry(root, textvariable = self.path).grid(row = 3, column = 3)
        self.Button = Button(root, text = "确定", command = self.startreading).grid(row = 3, column = 4)
        
        
        self.label_path = Label(root,text = "",font=('黑体',18,'bold','italic')).grid(row = 4, column = 0)
        self.label_path = Label(root,text = "各算法计算结果:",font=('黑体',18,'bold','italic'),anchor="w").grid(row = 5, column = 0)
        self.label_path = Label(root,text = "precisio",font=('黑体',18,'bold','italic'),anchor="w").grid(row = 5, column = 1)
        self.label_path = Label(root,text = "recall",font=('黑体',18,'bold','italic'),anchor="w").grid(row = 5, column = 2)
        self.label_path = Label(root,text = "f1_score",font=('黑体',18,'bold','italic'),anchor="w").grid(row = 5, column = 3)
        self.label_path = Label(root,text = "accuracy",font=('黑体',18,'bold','italic'),anchor="w").grid(row = 5, column = 4)
        self.label_path = Label(root,text = "").grid(row = 6, column = 0)
        
        self.label_path = Label(root,text = "随机森林算法:",font=('黑体',13,'bold','italic'),anchor="w").grid(row = 7, column = 0)
        self.randomforest_precision = Label(root,textvariable =self.ran_precision ).grid(row = 7, column = 1)
        self.randomforest_recall = Label(root,textvariable =self.ran_recall ).grid(row = 7, column = 2)
        self.randomforest_f1_scores = Label(root,textvariable =self.ran_f1_scores ).grid(row = 7, column = 3)
        self.randomforest_accuracy = Label(root,textvariable =self.ran_accuracy ).grid(row = 7, column = 4)
        self.label_path = Label(root,textvariable = "").grid(row = 8, column = 0)
        
        self.label_path = Label(root,text = "GradientBoosting算法:",font=('黑体',13,'bold','italic'),anchor="w").grid(row = 9, column = 0)
        self.GradientBoosting_precision = Label(root,textvariable =self.gra_precision ).grid(row = 9, column = 1)
        self.GradientBoosting_recall = Label(root,textvariable =self.gra_recall ).grid(row = 9, column = 2)
        self.GradientBoosting_f1_scores = Label(root,textvariable =self.gra_f1_scores ).grid(row = 9, column = 3)
        self.GradientBoosting_accuracy = Label(root,textvariable =self.gra_accuracy ).grid(row = 9, column = 4)
        self.label_path = Label(root,text = "").grid(row = 10, column = 0)
        
        self.label_path = Label(root,text = "AdaBoost算法:",font=('黑体',13,'bold','italic'),anchor="w").grid(row = 11, column = 0)
        self.AdaBoost_precision = Label(root,textvariable =self.ada_precision ).grid(row = 11, column = 1)
        self.AdaBoost_recall = Label(root,textvariable =self.ada_recall ).grid(row = 11, column = 2)
        self.AdaBoost_f1_scores = Label(root,textvariable =self.ada_f1_scores ).grid(row = 11, column = 3)
        self.AdaBoost_accuracy = Label(root,textvariable =self.ada_accuracy ).grid(row = 11, column = 4)
        self.label_path = Label(root,text = "").grid(row = 12, column = 0)
        
        self.label_path = Label(root,text = "BaggingRegressor算法:",font=('黑体',13,'bold','italic'),anchor="w").grid(row = 13, column = 0)
        self.Bagging_precision = Label(root,textvariable =self.bag_precision ).grid(row = 13, column = 1)
        self.Bagging_recall = Label(root,textvariable =self.bag_recall ).grid(row = 13, column = 2)
        self.Bagging_f1_scores = Label(root,textvariable =self.bag_f1_scores ).grid(row = 13, column = 3)
        self.Bagging_accuracy = Label(root,textvariable =self.bag_accuracy ).grid(row = 13, column = 4)
        self.label_path = Label(root,text = "").grid(row = 14, column = 0)
        
    

root = Tk()
display = ui(root)

root.mainloop()
