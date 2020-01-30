from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.decomposition import PCA
from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import resnet50_input
import resnet50

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'eval',"""Directory where to write event logs.""")
tf.app.flags.DEFINE_string('train_dir', 'chk',"""Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_boolean('evaluation', False, "Whether this is evaluation or representation calculation")
tf.app.flags.DEFINE_integer('num_examples', 500,"""Number of examples to run.""")


def eval_once(saver, summary_writer, top_k_op, summary_op, ignore, preds):
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print('No checkpoint file found')
      return
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))


      num_iter = int(math.floor(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      zeros_count = 0
      step = 0
      while step < num_iter:
        predictions, ignore_, preds_ = sess.run([top_k_op, ignore, preds])
        true_count += np.sum(predictions)
        zeros_count += np.sum(ignore_)
        step += 1

      # Compute precision @ 1.
      precision = (true_count - zeros_count) / (num_iter*FLAGS.batch_size*resnet50.LABEL_SIZE*resnet50.LABEL_SIZE - zeros_count)
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

  return precision

def evaluate():
  with tf.Graph().as_default() as g:
    images, labels, _, ignore, poss_lbls = resnet50.inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, ignore_, _, _, _ = resnet50.inference(images, ignore, poss_lbls)
    _, preds = tf.nn.top_k(logits)
    logits = tf.reshape(logits, [-1, resnet50.NUM_CLASSES])

    # Calculate predictions.
    labels = tf.cast(labels, tf.int64)
    labels = tf.reshape(labels, [-1])
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    gv = tf.global_variables()
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    precision = eval_once(saver, summary_writer, top_k_op, summary_op, ignore_, preds)
  return precision


def get_representations():
  with tf.Graph().as_default():
    image, filenames = resnet50.test_inputs()
    filenames = [element.split('.')[0] for element in filenames]

    # Build a Graph that computes the logits predictions from the
    # inference model.
    _, _, fuse = resnet50.inference(image, tf.zeros([1,112,112,resnet50_input.NUM_CLASSES]), tf.zeros([1,112,112,resnet50_input.NUM_CLASSES]))
#    _, predictions = tf.nn.top_k(logits)

    # Restore the moving average version of the learned variables for eval.
#    variable_averages = tf.train.ExponentialMovingAverage(
#        resnet50.MOVING_AVERAGE_DECAY)
#    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        print('No checkpoint file found')
        return
      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                           start=True))

        for i in range(FLAGS.num_examples):
          representation = sess.run(fuse)
          res_im = np.squeeze(representation)
          res_vec = res_im.reshape((res_im.shape[0] * res_im.shape[1], res_im.shape[2]))
          res_reduced = PCA(n_components=9,svd_solver='full').fit_transform(res_vec)
          representation = np.reshape(res_reduced, (res_im.shape[0], res_im.shape[1], 9))

      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):  # pylint: disable=unused-argument-+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  if FLAGS.evaluation:
    evaluate()
  else:
    get_representations()


if __name__ == '__main__':
  tf.app.run()
