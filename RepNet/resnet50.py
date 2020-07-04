from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
import resnet50_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size',4,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'BSDS500//',"""Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,"""Train the model using fp16.""")
tf.app.flags.DEFINE_string('weights', 'fp_weights.npy',"""Path to the Pre-trained weights.""")
tf.app.flags.DEFINE_float('wd', 0.00025,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('dropout', 0.5,"Define the dropout ratio")
tf.app.flags.DEFINE_boolean('projection', True,"""Projection layer or zero padding.""")
tf.app.flags.DEFINE_boolean('use_pretrained', True,"""Whether to use the pre-trained weights or not.""")
tf.app.flags.DEFINE_float('learning_rate', 0.0001,"""Initial learning rate.""")
tf.app.flags.DEFINE_float('lr_decay_epochs', 350,"""Number of epochs per learning rate decay.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = resnet50_input.IMAGE_SIZE
LABEL_SIZE = resnet50_input.LABEL_SIZE
NUM_CLASSES = resnet50_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = resnet50_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def conv_relu(scope, layer_name, prev_layer, conv_shape, stride, train, atrous):
  kernel = _variable_with_weight_decay('weights', shape=conv_shape, wd=FLAGS.wd, layer_name=layer_name)
  if atrous>0:
      conv = tf.nn.atrous_conv2d(prev_layer, kernel, atrous, padding='SAME')
  else:
      conv = tf.nn.conv2d(prev_layer, kernel, [1, stride, stride, 1], padding='SAME')
  conv_normed = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=train, scope=scope)
  output = tf.nn.relu(conv_normed)
  return output


def res_block(scope, prev_layer, conv_shapes, stride, train, atrous=0):
  # branch2a
  with tf.variable_scope('branch2a') as scope_inner:
      branch2a = conv_relu(scope_inner, scope.name+'_branch2a', prev_layer, conv_shapes[0], stride, train, atrous)
  
  # branch2b
  with tf.variable_scope('branch2b') as scope_inner:
      branch2b = conv_relu(scope_inner, scope.name+'_branch2b', branch2a, conv_shapes[1], 1, train, atrous)
  
  # branch2c
  with tf.variable_scope('branch2c') as scope_inner:
      kernel = _variable_with_weight_decay('weights', shape=conv_shapes[2], wd=FLAGS.wd, layer_name=scope.name+'_branch2c')
      if atrous>0:
          conv = tf.nn.atrous_conv2d(branch2b, kernel, atrous, padding='SAME')
      else:
          conv = tf.nn.conv2d(branch2b, kernel, strides=[1,1,1,1], padding='SAME')
      branch2c = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=train, scope=scope_inner)

  # Input projection and output
  input_depth = prev_layer.get_shape().as_list()[3]
  output_depth = conv_shapes[2][3]
  if (input_depth != output_depth) & (FLAGS.projection==True):
      with tf.variable_scope('branch1') as scope_inner:
          kernel = _variable_with_weight_decay('weights', shape=[1, 1, input_depth, output_depth], wd=FLAGS.wd, layer_name=scope.name+'_branch1')
          conv = tf.nn.conv2d(prev_layer, kernel, strides=[1, stride, stride, 1], padding='SAME')
          branch1 = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=train, scope=scope_inner)
  elif (input_depth != output_depth) & (FLAGS.projection==False):
      with tf.variable_scope('branch1') as scope_inner:
          prev_layer = tf.nn.max_pool(prev_layer, ksize=[1, 1, 1, 1], strides=[1, stride, stride, 1], padding='SAME')
          branch1 = tf.pad(prev_layer, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
  else:
      branch1 = prev_layer
  output = tf.nn.relu(tf.add(branch2c, branch1))
  return output


def _activation_summary(x):
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer, layer_name):
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, wd, layer_name):
  var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer(uniform=False), layer_name)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  images, labels, edges, ignore, poss_lbls = resnet50_input.distorted_inputs(FLAGS.data_dir, FLAGS.batch_size)
  return images, labels, edges, ignore, poss_lbls


def inputs():
  images, labels, edges, ignore, poss_lbls = resnet50_input.inputs(FLAGS.data_dir, FLAGS.batch_size)
  return images, labels, edges, ignore, poss_lbls


def test_inputs():
  image, filenames = resnet50_input.test_inputs(FLAGS.data_dir)
  return image, filenames
  

def inference(images, ignore, possibleLabels, train=False, rep = False):
  tf.summary.image('images', images, max_outputs=1)
  shape = tf.shape(images)

  ## conv1
  with tf.variable_scope('conv1') as scope:
    conv1 = conv_relu(scope, scope.name, images, [7, 7, 3, 64], 1, train, 0)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

  ## conv2
  # conv2_1
  with tf.variable_scope('conv2_1') as scope:
    conv2_1 = res_block(scope, pool1, [[1,1,64,64],[3,3,64,64],[1,1,64,256]], 1, train)
    _activation_summary(conv2_1)

  # conv2_2
  with tf.variable_scope('conv2_2') as scope:
    conv2_2 = res_block(scope, conv2_1, [[1,1,256,64],[3,3,64,64],[1,1,64,256]], 1, train)
    _activation_summary(conv2_2)

  # conv2_3
  with tf.variable_scope('conv2_3') as scope:
    conv2_3 = res_block(scope, conv2_2, [[1,1,256,64],[3,3,64,64],[1,1,64,256]], 1, train)
    _activation_summary(conv2_3)

  ## conv3
  # conv3_1
  with tf.variable_scope('conv3_1') as scope:
    conv3_1 = res_block(scope, conv2_3, [[1,1,256,128],[3,3,128,128],[1,1,128,512]], 2, train)
    _activation_summary(conv3_1)

  # conv3_2
  with tf.variable_scope('conv3_2') as scope:
    conv3_2 = res_block(scope, conv3_1, [[1,1,512,128],[3,3,128,128],[1,1,128,512]], 1, train)
    _activation_summary(conv3_2)

  # conv3_3
  with tf.variable_scope('conv3_3') as scope:
    conv3_3 = res_block(scope, conv3_2, [[1,1,512,128],[3,3,128,128],[1,1,128,512]], 1, train)
    _activation_summary(conv3_3)

  # conv3_4
  with tf.variable_scope('conv3_4') as scope:
    conv3_4 = res_block(scope, conv3_3, [[1,1,512,128],[3,3,128,128],[1,1,128,512]], 1, train)
    _activation_summary(conv3_4)

  shape = tf.shape(conv3_4)
  ## conv4  
  # conv4_1
  with tf.variable_scope('conv4_1') as scope:
    conv4_1 = res_block(scope, conv3_4, [[1,1,512,256],[3,3,256,256],[1,1,256,1024]], 2, train)
    _activation_summary(conv4_1)

  # conv4_2
  with tf.variable_scope('conv4_2') as scope:
    conv4_2 = res_block(scope, conv4_1, [[1,1,1024,256],[3,3,256,256],[1,1,256,1024]], 1, train)
    _activation_summary(conv4_2)

  # conv4_3
  with tf.variable_scope('conv4_3') as scope:
    conv4_3 = res_block(scope, conv4_2, [[1,1,1024,256],[3,3,256,256],[1,1,256,1024]], 1, train)
    _activation_summary(conv4_3)

  # conv4_4
  with tf.variable_scope('conv4_4') as scope:
    conv4_4 = res_block(scope, conv4_3, [[1,1,1024,256],[3,3,256,256],[1,1,256,1024]], 1, train)
    _activation_summary(conv4_4)

  # conv4_5
  with tf.variable_scope('conv4_5') as scope:
    conv4_5 = res_block(scope, conv4_4, [[1,1,1024,256],[3,3,256,256],[1,1,256,1024]], 1, train)
    _activation_summary(conv4_5)

  # conv4_6
  with tf.variable_scope('conv4_6') as scope:
    conv4_6 = res_block(scope, conv4_5, [[1,1,1024,256],[3,3,256,256],[1,1,256,1024]], 1, train)
    _activation_summary(conv4_6)
    
  ##conv5    
  # conv5_1
  with tf.variable_scope('conv5_1') as scope:
    conv5_1 = res_block(scope, conv4_6, [[1,1,1024,512],[3,3,512,512],[1,1,512,2048]], 1, train, 2)
    _activation_summary(conv5_1)
    
  # conv5_2
  with tf.variable_scope('conv5_2') as scope:
    conv5_2 = res_block(scope, conv5_1, [[1,1,2048,512],[3,3,512,512],[1,1,512,2048]], 1, train, 2)
    _activation_summary(conv5_2)
    
  # conv5_3
  with tf.variable_scope('conv5_3') as scope:
    conv5_3 = res_block(scope, conv5_2, [[1,1,2048,512],[3,3,512,512],[1,1,512,1024]], 1, train, 2)
#    if train:
#        conv5_3 = tf.nn.dropout(conv5_3, FLAGS.dropout, noise_shape=[FLAGS.batch_size, 1, 1, 512])
    _activation_summary(conv5_3)

  ## Fuse multiple layers
  with tf.variable_scope('concat') as scope:
#    conv1_ = tf.nn.avg_pool(conv1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
#    conv2_3_ = tf.image.resize_images(conv2_3, [shape[1], shape[2]], method=tf.image.ResizeMethod.BILINEAR) 
#    conv3_4_ = tf.image.resize_images(conv3_4, [shape[1], shape[2]], method=tf.image.ResizeMethod.BILINEAR)
    conv4_6_ = tf.image.resize_images(conv4_6, [shape[1], shape[2]], method=tf.image.ResizeMethod.BILINEAR)
    conv5_3_ = tf.image.resize_images(conv5_3, [shape[1], shape[2]], method=tf.image.ResizeMethod.BILINEAR)
    reps_concat = tf.concat([conv3_4, conv4_6_, conv5_3_], axis=3)
    
  # fuse
  with tf.variable_scope('fuse') as scope:
    fuse = res_block(scope, reps_concat, [[1,1,2560,512],[1,1,512,512],[1,1,512,512]], 1, train)
    if train:
        fuse = tf.nn.dropout(fuse, FLAGS.dropout, noise_shape=[FLAGS.batch_size, 1, 1, 512])
    _activation_summary(fuse)
  
  # linear layer(WX + b),
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [1, 1, 512, NUM_CLASSES], wd=FLAGS.wd, layer_name=scope.name+'_w')
    conv = tf.nn.conv2d(fuse, weights, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0), layer_name=scope.name+'_b')
    softmax_linear = tf.add(conv, biases)
    _activation_summary(softmax_linear)

#  ignore = tf.cast(ignore, tf.float32)
#  ignore_ = tf.scalar_mul(1e10, ignore)
#  softmax_linear = softmax_linear + ignore_
#
  if not rep:
    possibleLabels = tf.cast(possibleLabels, tf.float32)
    possibleLabels = tf.scalar_mul(1e10, possibleLabels)
    softmax_linear = softmax_linear - possibleLabels

  return softmax_linear, ignore, fuse


def loss(logits, labels, edges):
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
  edges = tf.cast(tf.multiply(edges,5), tf.float32) # was 15
  cross_entropy_scaled = cross_entropy + tf.multiply(cross_entropy, edges)
  cross_entropy_mean = tf.reduce_mean(cross_entropy_scaled, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op
