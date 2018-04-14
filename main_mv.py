# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
'''
4/12
22. cnn 으로 실행, embedding 넣고 10 으로 수정 하고 

31.minmax + embedding 10 = 8.252
32.minmax + embedding 10 + epoch20 = 7.675
35.embedding 10 + epoch20 = 6.976
38.35+keep_prob=0.9+y_1625 = 6.488
39.38동일 -12.xx
42.y_ 625 = 6.23
43.cnn W3 추가 = 6.333
44.keep_prob = 0.7 + y_300 = (train)7.298
52.epoch60,batch1000,embedd8,max_pool(stride 1,1,1,1) = 6.43
60.softmax,추가 = train 0.9 근데 리더보드엔 안올라감 왜? 
'''
import argparse
import os

import numpy as np
import tensorflow as tf

import nsml
from dataset import MovieReviewDataset, preprocess
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        re_preprocessed_data = np.reshape(preprocessed_data,(-1,225))
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred1 = []
        for i in range(len(preprocessed_data)):
            #re_preprocessed_data = np.reshape(preprocessed_data[i,:,],(-1,225))
            pred = sess.run(logits, feed_dict={x: re_preprocessed_data[i:i+1], keep_prob: 1})
            pred1.extend(pred)
        
        point = np.array(pred1)
        point = np.squeeze(point)
        point_arg = np.argmax(point, 1)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point_arg+1))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)

def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]
        
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=20)
    args.add_argument('--batch', type=int, default=1000)
    args.add_argument('--strmaxlen', type=int, default=225) #한문장의 길이
    args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--img_input', type=int, default=15)  #x_img 의 input변수(한문장의길이의 루트)

    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    #모델의 specification
    
    # parameter ========
    input_size = config.embedding*config.strmaxlen
    output_size = 1
    output_layer = 625
    learning_rate = 0.001
    character_size = 251
    nb_classes = 10 #1~10
    # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
    keep_prob = tf.placeholder(tf.float32)
    
    x = tf.placeholder(tf.int32, [None, config.strmaxlen])    #107 * 225
    x_img = tf.reshape(x, [-1,config.img_input,config.img_input])
    x_img = tf.cast(x_img, tf.int32)
    
    y_ = tf.placeholder(tf.int32, [None, output_size])
    Y_one_hot = tf.one_hot(y_, nb_classes)  # one hot
    #print("one_hot", Y_one_hot)
    Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
    #print("reshape", Y_one_hot)
    Y_one_hot = tf.cast(Y_one_hot, tf.float32)
    # 임베딩
    
    char_embedding = tf.get_variable('char_embedding', [character_size, config.embedding])
    embedded = tf.nn.embedding_lookup(char_embedding, x_img)
    embedded_img=tf.reshape(embedded,[-1,config.img_input,config.img_input,config.embedding])

    print("============================")

   # 첫 번째 레이어
    #32: 임의
    W1 = tf.Variable(tf.random_normal([2, 2, config.embedding, 32], stddev=0.01))
    #    Conv     -> (?, 28, 28, 32) 
    #    Pool     -> (?, 14, 14, 32)
    L1 = tf.nn.conv2d(embedded_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
    # 두 번째 레이어

    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    #    Conv      ->(?, 14, 14, 64)
    #    Pool      ->(?, 7, 7, 64)
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    
    W3=tf.Variable(tf.random_normal([3,3,64,64],stddev=0.01))
    L3=tf.nn.conv2d(L2,W3,strides=[1,1,1,1],padding='SAME')
    L3=tf.nn.relu(L3)
    L3=tf.nn.max_pool(L3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    L3=tf.nn.dropout(L3,keep_prob=keep_prob)
    L3_flat=tf.reshape(L3,[-1,64*2*2])
    
    # L4 FC 64 * 4 * 4 inputs -> 625 outputs
    W4 = tf.get_variable("W4", shape=[64 * 2 * 2, output_layer],
                         initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([output_layer]))
    L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
    '''
    Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
    Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
    '''
    # L5 Final FC 625 inputs -> 10 outputs
    W5 = tf.get_variable("W5", shape=[output_layer, nb_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([nb_classes]))
    logits = tf.matmul(L4, W5) + b5
    logits = tf.nn.softmax(logits)
    #logits = tf.nn.softmax(tf.matmul(L4, W5)+b5)

    
    # loss와 optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y_one_hot))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)


    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)
    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())
    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        
        dataset_len = len(dataset)
        one_batch_size = dataset_len//config.batch 
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):
                #data = MinMaxScaler(data) * 100
                labels = np.reshape(labels,(-1,1))
                feed_dict={x: data, y_: labels, keep_prob:0.7}
                _, loss = sess.run([train_step, cost], feed_dict=feed_dict)
                print('Batch : ', i + 1, '/', one_batch_size,
                      ', BCE in this minibatch: ', loss.astype(float))
                avg_loss += loss.astype(float)
            print('epoch:', epoch, ' train_loss:', (avg_loss/one_batch_size).astype(float))
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=(avg_loss/one_batch_size).astype(float), step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)
            
        '''
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)
        tf.reset_default_graph()
        '''
    
       
    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)
        

        