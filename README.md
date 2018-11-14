# ResNet-TensorFlow
开发环境：
--------
    python 3.5
    TensorFlow 1.3.0

Unit.cfg    配置文件说明
-----------------------

[Image data]        图像数据

    image_size          图像大小
    train_ratio         训练集占比
    num_classes         分类数目

[Train data]        训练数据

    num_train           训练集图像数量
    learning_rate       学习率
    step                训练迭代步数
    batch_size          训练批度

[Test data]         测试数据

    num_test            测试集图像数量
    batch_size          测试批度

--------------------------------------

    FlowIO.py   数据流控制器

    主要用于数据封装、读取等操作
    
    train.py    训练
    
    Evaluation.py   精度测试
    
    useModel.py     使用模型预测
    
    ResNet_lib.py   ResNet系列模型前趋关系

    
自行创建目录 

    data/dataset/       源数据集位置
    data/TFRecode/      TFRecode位置
    model/              模型保存位置
    log_file/           TensorBoard位置

训练你的模型
-----------
    1、将数据集放入data/dataset/中，子文件夹名称为标签
        sample：data/dataset/0/1.jpeg
                data/dataset/8/46.jpeg
                ......
    2、在Unit.cfg中调整分类数目、图像大小以及训练分割占比
    3、run python3 FlowIO.py 分割数据集并且打包成TFRecode格式
    4、run python3 train.py 训练数据集
    训练好的模型将会保存到 model/ 路径下
    如果你想在某一时刻模型继续训练，可输入 --model=模型路径与模型名称 来指定模型继续训练
测试你的模型
-----------
    1、确保你的TFRecode路径正确，检查Unit.cfg中的图像参数也正确
    2、run python3 --model=模型路径与模型名
使用你的模型
-----------
    1、确保useModel中的图像大小、分类数目以及分类标签的正确
    2、run python3 --model=模型路径与模型名
    
文章地址：公众号：python疯子

