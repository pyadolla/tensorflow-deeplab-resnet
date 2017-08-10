"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import socket
import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, prepare_label
hostname = socket.gethostname()
IMG_MEAN = np.array((106.13218097,116.46633804,124.99149765), dtype=np.float32)

DATA_DIRECTORY = '/root/alearn/data/ADEalearning_class_remap'
DATA_LIST_PATH = '/root/alearn/data/ADEalearning_class_remap/val.txt';
IGNORE_LABEL = 255
NUM_CLASSES = 13
NUM_STEPS = 1189 # Number of images in the validation set.
RESTORE_FROM = '/root/alearn/snapshots/'+ hostname +'/model.ckpt-20000'
SNAPSHOT_DIR = '/root/alearn/snapshots/'+ hostname +'/eval'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def _get_streaming_metrics(prediction,label,num_classes):

    with tf.name_scope("test"):
        # the streaming accuracy (lookup and update tensors)
        accuracy,accuracy_update = tf.metrics.accuracy(label, prediction,
                                               name='accuracy')
        # Compute a per-batch confusion
        batch_confusion = tf.confusion_matrix(label, prediction,
                                             num_classes=num_classes,
                                             name='batch_confusion')
        # Create an accumulator variable to hold the counts
        total_cm = tf.Variable( tf.zeros([num_classes,num_classes],
                                          dtype=tf.int32 ),
                                 name='confusion' )
        # Create the update op for doing a "+=" accumulation on the batch
        confusion_update = total_cm.assign( total_cm + batch_confusion )

        # Cast counts to float so tf.summary.image renormalizes to [0,255]
        sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1,keep_dims=True))
        sum_over_col = tf.tile(sum_over_col,[1,num_classes]);

        sum_over_col= tf.where(
          tf.greater(sum_over_col, 0),
          sum_over_col,
          tf.ones_like(sum_over_col))
        confusion_image = tf.reshape( tf.div(tf.cast(total_cm, tf.float32),sum_over_col),
                                  [1, num_classes, num_classes, 1])
        # Combine streaming accuracy and confusion matrix updates in one op
        test_op = tf.group(accuracy_update, confusion_update)

        tf.summary.image('confusion',confusion_image)
        tf.summary.scalar('accuracy',accuracy)

    return test_op,accuracy,total_cm

def iou(labels,predictions,num_classes):
    update_op,accuracy,total_cm= _get_streaming_metrics(tf.reshape(predictions,[-1]),tf.reshape(labels,[-1]),num_classes)

    def compute_iou(name):
      """Compute the mean intersection-over-union via the confusion matrix."""
      sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
      sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
      cm_diag = tf.to_float(tf.diag_part(total_cm))
      denominator = sum_over_row + sum_over_col - cm_diag

      # If the value of the denominator is 0, set it to 1 to avoid
      # zero division.
      denominator = tf.where(
          tf.greater(denominator, 0),
          denominator,
          tf.ones_like(denominator))
      iou = tf.div(cm_diag, denominator,name=name)
      return iou, sum_over_col

    iou_v, colsum_cm = compute_iou('iou')
    return iou_v, accuracy, colsum_cm, update_op


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    val_writer= tf.summary.FileWriter(SNAPSHOT_DIR)

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            None, # No defined input size.
            False, # No random scale.
            False, # No random mirror.
            args.ignore_label,
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label

    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.
    h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
    #image_batch075 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
    image_batch05 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.5)), tf.to_int32(tf.multiply(w_orig, 0.5))]))

    # Create network.
    #with tf.variable_scope('', reuse=False):
    #    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)
    #with tf.variable_scope('', reuse=True):
    #    net075 = DeepLabResNetModel({'data': image_batch075}, is_training=False, num_classes=args.num_classes)
    #with tf.variable_scope('', reuse=True):
    net05 = DeepLabResNetModel({'data': image_batch05}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    #raw_output100 = net.layers['fc1_voc12']
    #raw_output075 = tf.image.resize_images(net075.layers['fc1_voc12'], tf.shape(raw_output100)[1:3,])
    #raw_output05 = tf.image.resize_images(net05.layers['fc1_voc12'], tf.shape(raw_output100)[1:3,])
    raw_output = net05.layers['fc1_voc12']

    #raw_output = tf.reduce_max(tf.stack([raw_output100, raw_output075, raw_output05]), axis=0)
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.

    # mIoU
    pred = tf.reshape(pred, [-1,])
    gt = tf.reshape(label_batch, [-1,])
    weights = tf.cast(tf.less_equal(gt, args.num_classes - 1), tf.int32) # Ignoring all labels greater than or equal to n_classes.
    mIoU, update_miou_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=args.num_classes, weights=weights)
    iou_v,accuracy,colsum_cm,update_iou_op = iou(gt,pred,args.num_classes)
    colsum_cm_norm = tf.div(tf.reduce_sum(colsum_cm),colsum_cm) / 100


    summary_op = tf.summary.merge_all()


    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(args.num_steps):
        preds, _, _ = sess.run([pred, update_iou_op, update_miou_op])
        if step % 100 == 0:
            print('step {:d}'.format(step))
    print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
    summaryT=sess.run(summary_op)
    val_writer.add_summary(summaryT)

    print(accuracy.eval(session=sess))
    print(iou_v.eval(session=sess))
    print(colsum_cm_norm.eval(session=sess))
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
