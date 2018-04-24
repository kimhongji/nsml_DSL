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
4/14
las: 0.00729(마지막 리더보드)
**train_loss 는 데이터 차이 정도 실제 정확도 구하는 것과
는 별개임**
Minmax : 적당하지 않음 
4. epoch:250,batch:100 = 0.4(nan error)
5. epoch:200,batch:1000 = 0.5(nan error)
6. epch:200, batch:2000, l1,l2:600, data/10 = 0.001
8. 6+minmax*10 = 0.012 (nan error )
10.epoc:200,batch:2000,1000-800-500, lr = 0.01 = (nan)
12.epoc:200,batch:2000,1000-800-500, lr = 0.001 = 0.002(리더보드)

4/25 
질문별로 나눠서 MLP로 학습시킨후 합쳤ㅇ늠 
'''

import argparse
import os

import numpy as np
import tensorflow as tf

import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

def feature_normalization(data):
    # parameter 
    feature_num = data.shape[1]
    data_point = data.shape[0]
    # you should get this parameter correctly
    nomal_feature = np.zeros([data_point,feature_num])
    ## your code here
    mu=np.mean(data,0)
    std=np.std(data,0)
    nomal_feature=(data-mu)/std
    ## end
    return nomal_feature
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
        #preprocessed_data = feature_normalization(preprocessed_data)
        #preprocessed_data = MinMaxScaler(preprocessed_data)* 10
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(fc_output_sigmoid, feed_dict={x: preprocessed_data})
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(pred.flatten(), clipped.flatten()))

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
    args.add_argument('--epochs', type=int, default=200)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=200)
    args.add_argument('--embedding', type=int, default=10)
    args.add_argument('--threshold', type=float, default=0.5)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/kin/'

    # 모델의 specification
    input_size = config.embedding*config.strmaxlen
    m1_output_size = 100
    m2_output_size = 100
    fc_output_size = 1
    hidden_layer_size0 = 400
    hidden_layer_size1 = 200
    fc_hidden_layer_size0 = 100
    learning_rate = 0.001
    character_size = 251

    x1 = tf.placeholder(tf.int32, [None, config.strmaxlen])
    x2 = tf.placeholder(tf.int32, [None, config.strmaxlen])
    y_ = tf.placeholder(tf.float32, [None, fc_output_size])
    
    
    
    # 임베딩
    char_embedding = tf.get_variable('char_embedding', [character_size, config.embedding])
    embedded1 = tf.nn.embedding_lookup(char_embedding, x1)
    embedded2 = tf.nn.embedding_lookup(char_embedding, x2)
    print("============================")
    print(embedded1)
    print(embedded2)

    # 첫 번째 레이어(첫번째 질문 )============================================
    m1_first_layer_weight0 = weight_variable([input_size, hidden_layer_size0])
    m1_first_layer_bias0= bias_variable([hidden_layer_size0])
    m1_hidden_layer0 = tf.matmul(tf.reshape(embedded1, (-1, input_size)), m1_first_layer_weight0) + m1_first_layer_bias0
    #hidden_layer0 = tf.nn.dropout(hidden_layer0,)
    # 두 번째 레이어
    m1_first_layer_weight1 = weight_variable([hidden_layer_size0, hidden_layer_size1])
    m1_first_layer_bias1 = bias_variable([hidden_layer_size1])
    m1_hidden_layer1 = tf.matmul(m1_hidden_layer0, m1_first_layer_weight1) + m1_first_layer_bias1
    m1_hidden_layer1 = tf.sigmoid(m1_hidden_layer1)
    #hidden_layer1 = tf.nn.dropout(hidden_layer1,1)
    # 아웃 레이어
    m1_second_layer_weight = weight_variable([hidden_layer_size1, m1_output_size])
    m1_second_layer_bias = bias_variable([m1_output_size])
    m1_output = tf.matmul(m1_hidden_layer1, m1_second_layer_weight) + m1_second_layer_bias
    m1_output_sigmoid = tf.sigmoid(m1_output)

    #========================================================================
    
    # 첫 번째 레이어(두번째 질문 )============================================
    m2_first_layer_weight0 = weight_variable([input_size, hidden_layer_size0])
    m2_first_layer_bias0= bias_variable([hidden_layer_size0])
    m2_hidden_layer0 = tf.matmul(tf.reshape(embedded2, (-1, input_size)), m2_first_layer_weight0) + m2_first_layer_bias0
    #hidden_layer0 = tf.nn.dropout(hidden_layer0,)
    # 두 번째 레이어
    m2_first_layer_weight1 = weight_variable([hidden_layer_size0, hidden_layer_size1])
    m2_first_layer_bias1 = bias_variable([hidden_layer_size1])
    m2_hidden_layer1 = tf.matmul(m2_hidden_layer0, m2_first_layer_weight1) + m2_first_layer_bias1
    m2_hidden_layer1 = tf.sigmoid(m2_hidden_layer1)
    #hidden_layer1 = tf.nn.dropout(hidden_layer1,1)
    # 아웃 레이어
    m2_second_layer_weight = weight_variable([hidden_layer_size1, m2_output_size])
    m2_second_layer_bias = bias_variable([m2_output_size])
    m2_output = tf.matmul(m2_hidden_layer1, m2_second_layer_weight) + m2_second_layer_bias
    m2_output_sigmoid = tf.sigmoid(m2_output)
    
    #========================================================================
    
     # 첫 번째 레이어(FC 질문1 + 질문2 )=======================================
    fc_first_layer_weight0 = weight_variable([m1_output_size+m2_output_size, fc_hidden_layer_size0])
    fc_first_layer_bias0 = bias_variable([fc_hidden_layer_size0])
    #fc_output = tf.concat([m1_output,m2_output],0)
    fc_hidden_layer0 = tf.matmul(tf.concat([m1_output_sigmoid,m2_output_sigmoid],1), fc_first_layer_weight0) + fc_first_layer_bias0
    #hidden_layer0 = tf.nn.dropout(hidden_layer0,)
    
    # 아웃 레이어
    fc_second_layer_weight = weight_variable([fc_hidden_layer_size0, fc_output_size])
    fc_second_layer_bias = bias_variable([fc_output_size])
    fc_output = tf.matmul(fc_hidden_layer0, fc_second_layer_weight) + fc_second_layer_bias
    fc_output_sigmoid = tf.sigmoid(fc_output)
    
    #========================================================================
    # loss와 optimizer
    binary_cross_entropy = tf.reduce_mean(-(y_ * tf.log(fc_output_sigmoid)) - (1-y_) * tf.log(1-fc_output_sigmoid))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(binary_cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):
                #data = feature_normalization(data)
                #data = MinMaxScaler(data) * 10
                #ml_lis :질문 첫번째 list, m2_lis : 질문 두번째 list
                m1_lis =[]
                m2_lis =[]
                pre_j = 0
                for i in range(len(data)):
                    for j in range(config.strmaxlen):
                        if data[i][j] == 76 :
                            m1_lis.append(data[i][0:j].copy())
                        elif data[i][j] == 77 :
                            pre_j = j
                            break
                        
                    m2_lis.append(data[i][0:j].copy())
                #각 데이터를 나눠서 각각의 list 에 저장하는 과정 완료 
                
                m1_data_array= np.zeros([len(data),int(config.strmaxlen)])
                m2_data_array= np.zeros([len(data),int(config.strmaxlen)])
                
                for i in range(len(data)):
                    for j in range(int(m1_lis[i].size)-1):
                           m1_data_array[i][j]=m1_lis[i][j]
                           
                for i in range(len(data)):
                    for j in range(int(m2_lis[i].size)-1):
                           m2_data_array[i][j]=m2_lis[i][j]
                #각각의 데이터 모두 같은 크기 ( 200 * len(data)) 를 갖기 위해 조     
                m1_data_array = np.reshape(m1_data_array,(-1,200))
                m2_data_array = np.reshape(m2_data_array,(-1,200)) 
                
                _, loss = sess.run([train_step, binary_cross_entropy],
                                   feed_dict={x1: m1_data_array, x2:m2_data_array, y_: labels})
                
                print('Batch : ', i + 1, '/', one_batch_size,
                      ', BCE in this minibatch: ', float(loss))
                avg_loss += float(loss)
            print('epoch:', epoch, ' train_loss:', float(avg_loss/one_batch_size))
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/one_batch_size), step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)
            tf.reset_default_graph()


    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)