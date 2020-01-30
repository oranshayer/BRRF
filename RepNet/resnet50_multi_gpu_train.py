from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import resnet50
#import resnet50_eval

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'chk',"""Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('epochs', 600, "Number of epochs to run.")
tf.app.flags.DEFINE_integer('num_gpus', 1,"""How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,"""Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('new_run', True,"""Whether this is a new run or not.""")
tf.app.flags.DEFINE_boolean('start_new_phase', True,"""Whether this is the first phase of the run or not.""")


def save_training_params():
    f = open(FLAGS.train_dir + '/summary.txt', 'w')
    f.write('Epochs = ' + str(FLAGS.epochs) + '\n')
    f.write('Batch size = ' + str(FLAGS.batch_size * FLAGS.num_gpus) + '\n')
    f.write('Learning rate = ' + str(FLAGS.learning_rate) + '\n')
    f.write('Weight decay = ' + str(FLAGS.wd) + '\n')
    f.write('Epochs per LR decay = ' + str(FLAGS.lr_decay_epochs) + '\n\n')
    f.close()


def tower_loss(scope, images, labels, edges, ignore, poss_lbls):
  # Build inference Graph.
  logits, _, _ = resnet50.inference(images, ignore, poss_lbls, train=True)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  resnet50.loss(logits, labels, edges)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % resnet50.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)

  return total_loss


def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_steps_per_epoch = int(resnet50.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             (FLAGS.batch_size * FLAGS.num_gpus))
    decay_steps = int(num_steps_per_epoch * FLAGS.lr_decay_epochs)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                    global_step,
                                    decay_steps,
                                    resnet50.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.AdamOptimizer(lr)

    # Create the dataset and placeholders
    images, labels, edges, ignore, poss_lbls = resnet50.distorted_inputs()
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
          [images, labels, edges, ignore, poss_lbls], capacity=2 * FLAGS.num_gpus)
    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (resnet50.TOWER_NAME, i)) as scope:
            # Dequeues one batch for the GPU
            image_batch, label_batch, edges_batch, ignore_batch, poss_lbls_batch = batch_queue.dequeue()
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss = tower_loss(scope, image_batch, label_batch, edges_batch, ignore_batch, poss_lbls_batch)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            #Added for BN - 31.7.17 Oran
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
              grads = opt.compute_gradients(loss)
              tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
#    variable_averages = tf.train.ExponentialMovingAverage(
#        resnet50.MOVING_AVERAGE_DECAY, global_step)
#    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op)#, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())
    
    trainablevars = tf.global_variables()
    trainablevars_new=[]
    for v in trainablevars:
        if v.op.name.find("softmax")==-1:
            trainablevars_new.append(v)
    
    saver_for_new_phase = tf.train.Saver(trainablevars_new)
        
    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)
    
    # load pre-trained parameters
    if FLAGS.use_pretrained & FLAGS.new_run:
        model_dict = np.load(FLAGS.weights).item()
        all_vars = tf.trainable_variables()
        for v in all_vars:
            if (v.op.name.find("weights")>-1) and (v.op.name.find("softmax_linear")==-1) and (v.op.name.find("fuse")==-1) and (v.op.name.find("conv5_3")==-1):
                assign_op = v.assign(model_dict[v.op.name])
                sess.run(assign_op)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    step_reached = -1
    max_steps = int(FLAGS.epochs * resnet50.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / (FLAGS.batch_size * FLAGS.num_gpus))
    # Load model if not a new run
    if (FLAGS.new_run==False) & (FLAGS.start_new_phase==False):
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)
          step_reached = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

    if (FLAGS.new_run==False) & (FLAGS.start_new_phase==True):
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
          saver_for_new_phase.restore(sess, ckpt.model_checkpoint_path)
          step_reached = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
          
    for step in xrange(int(step_reached)+1,max_steps):
      start_time = time.time()
      _, loss_value, image_batch_, label_batch_, edges_batch_, ignore_batch_, poss_lbls_batch_ = sess.run([train_op, loss, image_batch, label_batch, edges_batch, ignore_batch, poss_lbls_batch])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 100 == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
        print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

      if step % 300 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if (step % (num_steps_per_epoch*20) == 0) and (step != 0):
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
          

def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.new_run:
      if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
      tf.gfile.MakeDirs(FLAGS.train_dir)
      tf.gfile.MakeDirs(FLAGS.train_dir+'/weights')
      save_training_params()
  train()


if __name__ == '__main__':
  tf.app.run()
