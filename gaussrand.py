# -*- coding: utf-8 -*-
#caffenetの形にplacesを合わせる
import caffe
import sys
import numpy as np
import random
argvs = sys.argv  # コマンドライン引数を格納したリストの取得
argc = len(argvs) #
num=100 #全体の何％ガウシアンノイズをのせるか

if (argc != 4):   # 引数が足りない場合は、その旨を表示
    print 'Usage: # python %s deploy.prototxt model seednum' % argvs[0]
    quit()         # プログラムの終了

proto,model,seed=argvs[1:]
print proto,model,seed
random.seed(seed)
r=random.randint
# Load the original network and extract the fully connected layers' parameters.
cnet = caffe.Net(proto,model,caffe.TEST)
#pnet = caffe.Net('hybridCNN_deploy_upgraded.prototxt','hybridCNN_iter_700000_upgraded.caffemodel',caffe.TEST)

params = ['conv1','conv2','conv3']#,'conv4','conv5']#,'fc6', 'fc7']
params = ['conv3','conv4','conv5']#,'fc6', 'fc7']
params = ['conv1','conv2','conv3','conv4','conv5']#,'fc6', 'fc7']

# fc_params = {name: (weights, biases)}
nc_params = {pr: (cnet.params[pr][0].data,cnet.params[pr][1].data) for pr in params}
ncp=[cnet.params[pr][0].data.shape for pr in params]#レイヤーのユニット数など

burst_layer_list=[(x,int(x[0]*num*0.01)) for x in ncp]
burst_filter_info=[(i[0],random.sample(range(i[0][0]),i[1])) for i in burst_layer_list]

print burst_layer_list
print burst_filter_info
for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, nc_params[fc][0].shape, nc_params[fc][1].shape)

for layer_name,filter_list in zip(params,burst_filter_info):
    for unit in filter_list[1]:
        nc_params[layer_name][0][unit]+=0.01*np.random.randn(filter_list[0][1],filter_list[0][2],filter_list[0][3])
a=""
for i in params:
    a+=i
savename="gauss"+str(num)+"_"+a+"_"+"seed"+seed+"_"+model

cnet.save(savename)
print "save "+savename+" ok!!"
