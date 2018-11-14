import tensorflow as tf
import tensorflow.contrib.slim as slim


ResNet_demo = {
    "layer_50":[{"depth": 256,"num_class": 3},
                {"depth": 512,"num_class": 4},
                {"depth": 1024,"num_class": 6},
                {"depth": 2048,"num_class": 3}],

    "layer_101": [{"depth": 256, "num_class": 3},
                  {"depth": 512, "num_class": 4},
                  {"depth": 1024, "num_class": 23},
                  {"depth": 2048, "num_class": 3}],

    "layer_152": [{"depth": 256, "num_class": 3},
                  {"depth": 512, "num_class": 8},
                  {"depth": 1024, "num_class": 36},
                  {"depth": 2048, "num_class": 3}]
               }




ResNet_mini_demo = {
    "layer_50":[{"depth": 128,"num_class": 3},
                {"depth": 256,"num_class": 4},
                {"depth": 512,"num_class": 6},
                {"depth": 1024,"num_class": 3}],

    "layer_101": [{"depth": 128, "num_class": 3},
                  {"depth": 256, "num_class": 4},
                  {"depth": 512, "num_class": 23},
                  {"depth": 1024, "num_class": 3}],

    "layer_152": [{"depth": 128, "num_class": 3},
                  {"depth": 256, "num_class": 8},
                  {"depth": 512, "num_class": 36},
                  {"depth": 1024, "num_class": 3}]
               }


WEIGHT_DECAY = 0.001
#通道改变
def highway(input_tensor,  #输入Tensor
                 depth,
                 is_train):        #输出深度
    data = input_tensor
    data = slim.conv2d(data,depth,1,activation_fn=None)
    data = tf.layers.batch_normalization(data, training=is_train)
    print("shortcut  ",depth)
    return data


#降采样
def sampling(input_tensor,      #Tensor入口
            ksize = 1,          #采样块大小
            stride = 2):        #采样步长
    data = input_tensor
    if stride > 1:
        data = slim.max_pool2d(data,ksize,stride = stride)
        print("sampling  ", 2)
    return data



def conv2d_same(input_tensor,num_outputs,kernel_size,stride,is_train = True,activation_fn=tf.nn.relu,normalizer_fn = True,scope = None):
    data = input_tensor
    if stride is 1:
        data = slim.conv2d(inputs = data,num_outputs = num_outputs,kernel_size = kernel_size,stride = stride,
                           weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),activation_fn=None,padding='SAME',scope=scope)
    else:

        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        data = tf.pad(data,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        data = slim.conv2d(inputs = data,num_outputs = num_outputs,kernel_size = kernel_size,stride = stride,
                           weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),activation_fn=None,padding='VALID',scope=scope)

    print("Conv ",kernel_size, "depth = ", num_outputs, "  stride = ", stride)
    if normalizer_fn:
        data = tf.layers.batch_normalization(data, training=is_train)
        print("batch_norm")
    if activation_fn is not None:
        data = activation_fn(data)
        print("Relu")

    return data




    #瓶颈模块内部
def bottleneck(input_tensor,output_depth,stride,is_train):

    data = input_tensor
    #取出通道
    redepth = input_tensor.get_shape().as_list()[3]
    # 当通道不相符时，进行全零填充并降采样
    if output_depth == redepth:
        shortcut_tensor = sampling(input_tensor,stride = stride)
    else:
        #通道改变
        shortcut_tensor = highway(input_tensor,output_depth,is_train)
    #降通道处理
    data = conv2d_same(data,output_depth//4,1,1,is_train,scope='conv1_1x1')
    #提取特征
    data = conv2d_same(data,output_depth//4,3,stride,is_train,scope='conv2_3x3')
    #通道还原
    data = conv2d_same(data,output_depth,1,1,is_train,activation_fn = None,normalizer_fn = False,scope='conv3_1x1')
    #生成残差
    data = data + shortcut_tensor
    data = tf.nn.relu(data)
    print("output : ", data)
    print("***************res*****************")
    return data
#堆叠ResNet模块
def inference(input_tensor,  #数据入口
               demos,               #模型资料（list）
               num_output,          #出口数量
               is_train):
    data = input_tensor
    #第一层卷积7*7,stride = 2,深度为64
    data = conv2d_same(data,64,7,2,is_train,None,normalizer_fn = False)
    data = slim.max_pool2d(data,3,2,scope="pool_1")

    with tf.variable_scope("resnet"):
    # with tf.variable_scope("resnet"):
        #堆叠总类瓶颈模块
        demo_num = 0
        for demo in demos:
            demo_num += 1
            print("--------------------------------------------")
            #堆叠子类瓶颈模块
            with tf.variable_scope("num_" + str(demo_num)):
                for i in range(demo["num_class"]):
                    print(demo_num)
                    if demo_num is not 4:
                        if i == demo["num_class"] - 1:
                            stride = 2
                        else:
                            stride = 1
                    else:
                        stride = 1
                    with tf.variable_scope("bottleneck_" + str(i + 1)):
                        data = bottleneck(data,demo["depth"],stride,is_train)

            print("--------------------------------------------")
    data = tf.layers.batch_normalization(data,training=is_train)
    data = tf.nn.relu(data)
    #平均池化，也可用Avg_pool函数
    data = tf.reduce_mean(data, [1, 2], keep_dims=True,name='nal_pool')
    #data = slim.avg_pool2d(data,2)
    print("output : ", data)

    #最后全连接层
    data = slim.conv2d(data,num_output,1,activation_fn=None,scope='final_conv')

    data_shape = data.get_shape().as_list()
    nodes = data_shape[1] * data_shape[2] * data_shape[3]
    print(data,nodes)
    data = tf.reshape(tensor=data,shape = [-1, nodes])
    return data

