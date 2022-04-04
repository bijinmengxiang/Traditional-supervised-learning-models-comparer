# Traditional-supervised-learning-models-comparer
Using some pubilc dataset to compare the efficiency of traditional supervised learning models

V1.0:

只含四种集成算法的precision比较（准确来说应该是accuracy）。能够完成读取csv数据集，并进行自动训练。
对各类机器学习算法采用K-folds交叉验证方法求取平均正确率。（可调节fold数）

仅包括随机森林，GradientBoosting，AdaBoost，BaggingRegressor四种集成学习算法。

缺点与未来版本目标：
1. 目前此系统并未含有前端系统界面，未来考虑添加系统界面
2. 在机器学习领域，调参一直是一个永恒不变的热点话题，后期版本会增加调参功能。
3. 目前所包含的算法种类较少，且无法进行选择。未来考虑对算法数量进行增加，并与前端系统配合完成选择操作。
4. 目前的结果计算方法并不是标准的分类计算方法，后期会对分类方法进行标准化。
5. 目前仅支持读取csv格式，未来考虑可对其他格式文件进行处理训练。


V1.1：

1. 完成了二分类的标准化数据输出，包括precision,recall,f1_scores,accuracy的计算



V1.2：

1. 基于Tkinter完成前端系统的编写，可通过选择文件方式（目前仅支持csv格式）进行读取文件，并可输入folds数
2. 完成对前端系统的标准化数据显示。



V1.3:（算是第一个可用的正式版本。。。）

1. 完成对标准csv数据集格式的读取，默认采用最后一列作为目标预测列，采用除最后一列的所有列进行训练。
2. 优化了训练函数，将多个训练函数进行合并
3. 对K-folds数进行判空处理，如未填写，使用默认值3进行计算。
