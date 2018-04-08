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

import argparse
import os

import numpy as np
import tensorflow as tf
#import torch

#from torch.autograd import Variable
#from torch import nn, optim
#from torch.utils.data import DataLoader

#import nsml
from dataset import MovieReviewDataset, preprocess
#from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard


def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver=tf.train.Saver()
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
        #model.eval()
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        #output_prediction = model(preprocessed_data)
        #point = output_prediction.data.squeeze(dim=1).tolist()
        pred = sess.run(output_sigmoid, feed_dict={x: preprocessed_data})
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        #return list(zip(np.zeros(len(point)), point))
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    #nsml.bind(save=save, load=load, infer=infer)

'''
def collate_fn(data: list):
    """
    PyTorch DataLoader에서 사용하는 collate_fn 입니다.
    기본 collate_fn가 리스트를 flatten하기 때문에 벡터 입력에 대해서 사용이 불가능해, 직접 작성합니다.

    :param data: 데이터 리스트
    :return:
    """
    review = []
    label = []
    for datum in data:
        review.append(datum[0])
        label.append(datum[1])
    # 각각 데이터, 레이블을 리턴
    return review, np.array(label)
'''

def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다.
    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0,length,n):
        yield iterable[n_idx:min(n_idx+n,length)]
        
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    #
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')
    
    #User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=400)
    args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--threshold', type=float, default=0.5)
    config = args.parse_args()
    
    DATASET_PATH='../sample_data/movie_review/'
    
    #모델의 specification
    input_size= config.embedding*config.strmaxlen
    output_size=1
    hidden_layer_size=200
    learning_rate=0.001
    character_size=251
    
    x = tf.placeholder(tf.int32, [None, config.strmaxlen])
    y_ = tf.placeholder(tf.float32, [None, output_size])
    

    # 임베딩
    char_embedding = tf.get_variable('char_embedding', [character_size, config.embedding])
    embedded = tf.nn.embedding_lookup(char_embedding, x)
    
    #첫번째 레이어
    first_layer_weight = weight_variable([input_size, hidden_layer_size])
    first_layer_bias = bias_variable([hidden_layer_size])
    hidden_layer=tf.matmul(tf.reshape(embedded, (-1, input_size)), first_layer_weight)+first_layer_bias
    
    #두번째 레이어
    second_layer_weight=weight_variable([hidden_layer_size, output_size])
    second_layer_bias=bias_variable([output_size])
    output=tf.matmul(hidden_layer, second_layer_weight)+second_layer_bias
    output_sigmoid=tf.sigmoid(output)
    
    
    # loss와 optimizer
    binary_cross_entropy = tf.reduce_mean(tf.square(output_sigmoid - y_))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(binary_cross_entropy)
    
    sess=tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    bind_model(sess=sess, config=config)
    
    if config.mode == 'train':
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        one_batch_size=dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        #epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss =0.0
            for i,(data,labels) in enumerate(_batch_loader(dataset, config.batch)):
                _, loss = sess.run([train_step, binary_cross_entropy], feed_dict={x: data, y_: labels})
                print('Batch : ', i+1, '/', one_batch_size, ', BCE in this minibatch: ', float(loss))
                avg_loss += float(loss)
            print('epoch:', epoch, ' train_loss:', float(avg_loss/one_batch_size))
            
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        #for batch in _batch_loader(queries, config.batch):
            #temp_res = nsml.infer(batch)
            #res += temp_res
        print(res)
        
    
    
                

