# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
# ref: http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/filter_visualization.ipynb
def vis_square( data, padsize=1, padval=0 ):
  data -= data.min()
  data /= data.max()
   
  # force the number of filters to be square
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
  data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
   
  # tile the filters into an image
  data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
  data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

  plt.imshow( data )




def plot_filters( caffemodel_path, modelfile_path, save_path ):
  import caffe

  cnet = caffe.Classifier(
      modelfile_path,
      caffemodel_path,
      channel_swap=( 2,1,0 ),
      raw_scale=255,
      image_dims=( 227,227 ),

    )
  filters = cnet.params[ 'conv1' ][ 0 ].data
  vis_square( filters.transpose( 0, 2, 3, 1 ) )

  file_name = caffemodel_path.split( '/' )[ -1 ].split( '.' )[ 0 ]
  save_file_name = save_path+'/'+file_name+'.png'
  print 'Save to %s' % save_file_name
  plt.savefig( save_file_name )


if __name__ == '__main__':
  import sys
  if len( sys.argv ) != 4:
    print 'Usage: python vis_filters.py model.caffemodel model.prototxt save_folder_path'
    sys.exit()

  plt.rcParams[ 'xtick.direction' ] = 'out'
  plt.rcParams[ 'ytick.direction' ] = 'out'
  plt.rcParams[ 'figure.figsize' ] = ( 18, 18 )
  plt.rcParams[ 'image.interpolation' ] = 'nearest'
  plt.rcParams[ 'image.cmap' ] = 'gray'

  plot_filters( sys.argv[ 1 ], sys.argv[ 2 ], sys.argv[ 3 ] )
