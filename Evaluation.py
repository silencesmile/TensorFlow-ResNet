import tensorflow as tf
import numpy as np
from FlowIO import DataSetLib as OF
import argparse as aps
from configobj import ConfigObj
from tqdm import tqdm

cfg_struct = ConfigObj("Unit.cfg")
cfg_test = cfg_struct['Test data']
cfg_image = cfg_struct['Image data']
parser = aps.ArgumentParser(description="manual to this script")
parser.add_argument("--model",type=str,default=None)
parser.add_argument("--num_test",type=int,default=cfg_test["num_test"])
parser.add_argument("--classes",type=int,default=cfg_image["num_classes"])
parser.add_argument("--batch_size",type=int,default=cfg_test["batch_size"])
args = parser.parse_args()

BATCH_SIZE =  args.batch_size
IMAGE_SIZE = cfg_image['image_size']
N_CLASSES = args.classes

#模型地址
MODEL_PATH = "model/mod.ckpt-3000"

#测试集总量
NUM_TEST = args.num_test

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

def test():
    model = tf.train.import_meta_graph(MODEL_PATH + ".meta")
    graph = tf.get_default_graph()
    inputs = graph.get_operation_by_name('x-input').outputs[0]
    labels = graph.get_operation_by_name('y-input').outputs[0]
    is_train = graph.get_operation_by_name('is_train').outputs[0]
    # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
    pred = tf.get_collection('pred_network')[0]

    with tf.Session(graph=graph) as sess:
        model.restore(sess, MODEL_PATH)
        # x, y = ds.get_batch_data(TEST_DIR, sess, False, BATCH_SIZE)
        get_flow = OF(sess,"Test",[IMAGE_SIZE,IMAGE_SIZE,3],BATCH_SIZE)
        next_batch = get_flow.get_batch_data()
        # 取出测试集合
        test_pred_acc = []
        test_label_acc = []
        for i in tqdm(range(NUM_TEST // BATCH_SIZE),"测试中"):
            test_x,test_y = sess.run(next_batch)
            test_label_acc.append(np.reshape(test_y,[-1]))
            # 使用y进行预测
            pred_y = sess.run(tf.nn.softmax(pred,1), feed_dict={inputs:test_x,labels:test_y,is_train:[False]})
            test_pred_acc.append(pred_y)
            print("pred : ",np.argmax(pred_y,1))
            print("real : ",np.reshape(test_y,[-1]))
            
        test_label_acc = np.reshape(test_label_acc,[(NUM_TEST // BATCH_SIZE) * BATCH_SIZE])
        test_pred_acc = np.reshape(test_pred_acc,[(NUM_TEST // BATCH_SIZE) * BATCH_SIZE,N_CLASSES])
        test_pred_acc = tf.cast(test_pred_acc,dtype=tf.float32)
        test_label_acc = tf.cast(test_label_acc,dtype=tf.int32)
        pred_acc = evaluation(test_pred_acc, test_label_acc)
        acc = sess.run(pred_acc)
        print("accuracy : ",acc)

# 传入模型
if __name__ == '__main__':
    with tf.device("/cpu:0"):
        test()
