# -*- coding: utf-8 -*-
#caffenetの形にplacesを合わせる
import caffe

# Load the original network and extract the fully connected layers' parameters.
cnet = caffe.Net('deploy.prototxt','caffe_reference_imagenet_model',caffe.TEST)
pnet = caffe.Net('deploy.prototxt','places_caff',caffe.TEST)
#pnet = caffe.Net('hybridCNN_deploy_upgraded.prototxt','hybridCNN_iter_700000_upgraded.caffemodel',caffe.TEST)

params = ['conv1','conv2','conv3','conv4','conv5','fc6', 'fc7']
# fc_params = {name: (weights, biases)}
nc_params = {pr: (cnet.params[pr][0].data,cnet.params[pr][1].data) for pr in params}
pc_params = {pr: (pnet.params[pr][0].data,pnet.params[pr][1].data) for pr in params}
np=[cnet.params[pr][0].data.shape for pr in params]
pp=[pnet.params[pr][0].data.shape for pr in params]

for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, nc_params[fc][0].shape, nc_params[fc][1].shape)
for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, pc_params[fc][0].shape, pc_params[fc][1].shape)

def aa():
    for nc in params:
        nc_params[nc][0].flat=pc_params[nc][0].flat
        if len(nc_params)==len(pc_params):
            print "ok ninnini"
        for i in range(len(nc_params[nc][1])):
            nc_params[nc][1][i]=pc_params[nc][1][i]
        print "%s copyed"%nc
