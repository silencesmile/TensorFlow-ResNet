import tensorflow as tf
import ResNet_lib as resnet
from FlowIO import DataSetLib as OF
import os
import argparse as aps
from configobj import ConfigObj

cfg_struct = ConfigObj("Unit.cfg")
cfg_train = cfg_struct['Train data']
cfg_image = cfg_struct['Image data']

parser = aps.ArgumentParser(description="manual to this script")
parser.add_argument("--lr",type=float,default=cfg_train["learning_rate"])
parser.add_argument("--classes",type=int,default=cfg_image["num_classes"])
parser.add_argument("--batch_size",type=int,default=cfg_train["batch_size"])
parser.add_argument("--step",type=int,default=cfg_train["step"])
parser.add_argument("--output",type=str,default="model")
parser.add_argument("--save_num",type=int,default=1000)
parser.add_argument("--log_num",type=int,default=50)
parser.add_argument("--model",type=str,default=None)

args = parser.parse_args()

LEARNING_RATE_BASE = args.lr  # 基础学习率
N_CLASSES = args.classes      # 分类数目
BATCH_SIZE = args.batch_size  # 批大小
IMAGE_SIZE = cfg_image["image_size"] # 图像大小
NUM_CHANNELS = 3              # 图像深度
STEP = args.step              # 迭代步数
SAVE_NUM = args.save_num      # 保存步长
MODEL_PATH = args.output      # 模型保存地址
LOG_NUM = args.log_num        # 输出步长
MODEL = args.model            # 加载模型继续训练



def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        labels = tf.reshape(labels, [-1])
        labels = tf.one_hot(labels, depth=logits.get_shape().as_list()[1])
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss') + tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def trainning(loss, learning_rate,batch_size,num_data):
    with tf.name_scope('optimizer'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(
        learning_rate,
        global_step,
        num_data / batch_size, 0.99,
        staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        labels = tf.reshape(labels, [-1])
        labels = tf.one_hot(labels, depth=logits.get_shape().as_list()[1])
        labels = tf.argmax(labels,1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1), labels), tf.float32))
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
    
def train():
    x = tf.placeholder(tf.float32, [
        None,IMAGE_SIZE,IMAGE_SIZE,
        NUM_CHANNELS],name='x-input')
    # 标签
    y_ = tf.placeholder(tf.int64, [None,1], name='y-input')
    # 是否处于训练状态
    is_train = tf.placeholder(tf.bool, [1], name="is_train")
    # 获取结果
    y = resnet.inference(x, resnet.ResNet_mini_demo['layer_50'], N_CLASSES, is_train[0])

    loss = losses(y, y_)
    acc = evaluation(y, y_)
    train_step = trainning(loss, LEARNING_RATE_BASE, BATCH_SIZE, 1000)
    with tf.control_dependencies([train_step]):
        train_op = tf.no_op(name='train')
    # TensorFlow持久化类。
    tf.add_to_collection('pred_network', y)
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)
    with tf.Session() as sess:
        # 初始化神经网络
        tf.global_variables_initializer().run()
        with tf.device("/cpu:0"):
            get_flow = OF(sess, "Train", [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], BATCH_SIZE)
            # 获取训练TenSor
            next_batch = get_flow.get_batch_data()
        if MODEL is not None:
            # 加载模型
            saver.restore(sess, MODEL)
        tf.summary.image("R.G.B", tf.expand_dims(next_batch[0][0], 0))
        merged = tf.summary.merge_all()
        log_summary = tf.summary.FileWriter("log_files", sess.graph)
        # 迭代训练
        for i in range(STEP):
            iamges,labels = sess.run(next_batch)
            _, loss_value,acc_value,merged_value = sess.run([train_op, loss,acc,merged],
                                           feed_dict={x: iamges, y_: labels,is_train:[True]})
            log_summary.add_summary(merged_value,i)
            if i % LOG_NUM == 0:
                print("After %d training step(s), loss on training batch is %g." % (i, loss_value),"acc : ",acc_value)
                # 模型存储
                if loss_value is not None and i % SAVE_NUM == 0:
                    save_path = os.path.join(MODEL_PATH, 'mod.ckpt')
                    saver.save(sess, save_path, global_step=i)
        log_summary.close()
def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
