"""Training script with multi-scale inputs for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
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

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, inv_preprocess, prepare_label
hostname = socket.gethostname()
IMG_MEAN = np.array((106.13218097,116.46633804,124.99149765), dtype=np.float32)
BATCH_SIZE = 1
DATA_DIRECTORY = '/root/alearn/data/ADEalearning_class_remap'
DATA_LIST_TRAIN_PATH = '/root/alearn/data/ADEalearning_class_remap/train.txt';
GRAD_UPDATE_EVERY = 10
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 13
NUM_STEPS = 20001
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = '/root/alearn/checkpoints/'+ hostname +'/deeplab_resnet_init.ckpt'
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = '/root/alearn/snapshots3/'+ hostname +'/'
WEIGHT_DECAY = 0.0005
# WEIGHTS = np.array((1.07259629e-02, 6.22674847e+00, 3.13767719e+00, 6.24749005e-01,
                    # 7.03957653e+00, 4.73124504e+00, 8.77013922e-01, 4.30406141e+00,
                    # 7.04289913e+00, 5.07218408e+01, 3.59946132e+00, 6.52679491e+00,
                    # 4.18563664e-01), dtype=np.float32)*0.1
WEIGHTS = np.array((.1, 1, 1, 1,
                     1, 1, 1, 1,
                     1, 1, 1, 1,
                     1), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_TRAIN_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--grad-update-every", type=int, default=GRAD_UPDATE_EVERY,
                        help="Number of steps after which gradient update is applied.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to update the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
   '''Save weights.

   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def _get_streaming_metrics(prediction,label,num_classes):
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
    confusion_update = total_cm.assign_add( batch_confusion,use_locking=True )

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
    eval_op= tf.group(confusion_update,accuracy_update)

    cm_summary=tf.summary.image('confusion',confusion_image)
    acc_summary=tf.summary.scalar('accuracy',accuracy)

    return eval_op,accuracy,total_cm,cm_summary,acc_summary

def iou(labels,predictions,num_classes):
    update_op,accuracy,total_cm,cm_summary,acc_summary= _get_streaming_metrics(tf.reshape(predictions,[-1]),tf.reshape(labels,[-1]),num_classes)

    def compute_iou(name):
      """Compute the mean intersection-over-union via the confusion matrix."""
      sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
      sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
      cm_diag = tf.to_float(tf.diag_part(total_cm))
      denominator = sum_over_row + sum_over_col - cm_diag

      tot_sum = tf.to_float(tf.reduce_sum(tf.reduce_sum(total_cm,1),0))
      # If the value of the denominator is 0, set it to 1 to avoid
      # zero division.
      denominator = tf.where(
          tf.greater(denominator, 0),
          denominator,
          tf.ones_like(denominator))
      iou = tf.div(cm_diag, denominator,name=name)
      miou = tf.reduce_mean(iou,name='mean_' + name)
      return iou,miou,sum_over_col,tot_sum

    iou_v, miou, colsum_cm, tot_sum = compute_iou('iou')

    miou_summary =tf.summary.scalar('mean-iou',miou)
    iou_summary = tf.summary.histogram('ious',iou_v)

    return iou_v, miou, accuracy, colsum_cm, update_op,cm_summary,acc_summary,miou_summary,iou_summary,tot_sum



def main():
    """Create the model and start the training."""
    args = get_arguments()

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    tf.set_random_seed(args.random_seed)

    # Create queue coordinator.
    coord = tf.train.Coordinator()

    train_summary_writer = tf.summary.FileWriter(SNAPSHOT_DIR +'/train', graph=tf.get_default_graph())

    # Load train reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            args.random_scale,
            args.random_mirror,
            args.ignore_label,
            IMG_MEAN,
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)
        image_batch075 = tf.image.resize_images(image_batch, [int(h * 0.75), int(w * 0.75)])
        image_batch05 = tf.image.resize_images(image_batch, [int(h * 0.5), int(w * 0.5)])

    # Create network.
    with tf.variable_scope('', reuse=False):
        net = DeepLabResNetModel({'data': image_batch}, is_training=args.is_training, num_classes=args.num_classes)
    with tf.variable_scope('', reuse=True):
        net075 = DeepLabResNetModel({'data': image_batch075}, is_training=args.is_training, num_classes=args.num_classes)
    with tf.variable_scope('', reuse=True):
        net05 = DeepLabResNetModel({'data': image_batch05}, is_training=args.is_training, num_classes=args.num_classes)

    # For a small batch size, it is better to keep
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model.
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    # Predictions.
    raw_output100 = net.layers['fc1_voc12']
    raw_output075 = net075.layers['fc1_voc12']
    raw_output05 = net05.layers['fc1_voc12']
    raw_output = tf.reduce_max(tf.stack([raw_output100,
                                         tf.image.resize_images(raw_output075, tf.shape(raw_output100)[1:3,]),
                                         tf.image.resize_images(raw_output05, tf.shape(raw_output100)[1:3,])]), axis=0)

    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0
    assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
    assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))


    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    raw_prediction = tf.reshape(raw_output, [-1, args.num_classes])
    raw_prediction100 = tf.reshape(raw_output100, [-1, args.num_classes])
    raw_prediction075 = tf.reshape(raw_output075, [-1, args.num_classes])
    raw_prediction05 = tf.reshape(raw_output05, [-1, args.num_classes])

    label_proc = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False) # [batch_size, h, w]
    label_proc075 = prepare_label(label_batch, tf.stack(raw_output075.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False)
    label_proc05 = prepare_label(label_batch, tf.stack(raw_output05.get_shape()[1:3]), num_classes=args.num_classes, one_hot=False)

    raw_gt = tf.reshape(label_proc, [-1,])
    raw_gt075 = tf.reshape(label_proc075, [-1,])
    raw_gt05 = tf.reshape(label_proc05, [-1,])

    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.num_classes - 1)), 1)
    indices075 = tf.squeeze(tf.where(tf.less_equal(raw_gt075, args.num_classes - 1)), 1)
    indices05 = tf.squeeze(tf.where(tf.less_equal(raw_gt05, args.num_classes - 1)), 1)

    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    gt075 = tf.cast(tf.gather(raw_gt075, indices075), tf.int32)
    gt05 = tf.cast(tf.gather(raw_gt05, indices05), tf.int32)

    prediction = tf.gather(raw_prediction, indices)
    prediction100 = tf.gather(raw_prediction100, indices)
    prediction075 = tf.gather(raw_prediction075, indices075)
    prediction05 = tf.gather(raw_prediction05, indices05)


    # Pixel-wise softmax loss.
    weights=tf.reshape(tf.gather(WEIGHTS,tf.reshape(gt,[-1,])),tf.shape(gt))
    weights075=tf.reshape(tf.gather(WEIGHTS,tf.reshape(gt075,[-1,])),tf.shape(gt075))
    weights05=tf.reshape(tf.gather(WEIGHTS,tf.reshape(gt05,[-1,])),tf.shape(gt05))
    loss = tf.losses.sparse_softmax_cross_entropy(logits=prediction, labels=gt, weights=weights)
    loss100 = tf.losses.sparse_softmax_cross_entropy(logits=prediction100, labels=gt, weights=weights)
    loss075 = tf.losses.sparse_softmax_cross_entropy(logits=prediction075, labels=gt075, weights=weights075)
    loss05 = tf.losses.sparse_softmax_cross_entropy(logits=prediction05, labels=gt05, weights=weights05)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.reduce_mean(loss100) + tf.reduce_mean(loss075) + tf.reduce_mean(loss05) + tf.add_n(l2_losses)

    # Processed predictions: for visualisation.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Image summary.
    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images, args.num_classes], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images, args.num_classes], tf.uint8)

    total_summary = tf.summary.image('images',
                                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
                                     max_outputs=args.save_num_images) # Concatenate row-wise.


    weights = tf.cast(tf.less_equal(gt, args.num_classes - 1), tf.int32) # Ignoring all labels greater than or equal to n_classes.
    #mIoU, update_miou_op = tf.contrib.metrics.streaming_mean_iou(pred, label_batch, num_classes=args.num_classes, weights=weights)

    with tf.name_scope("metrics"):
        iou_v,miou,accuracy,colsum_cm,update_iou_op,cm_summary,acc_summary,miou_summary,iou_summary,tot_sum = iou(label_batch,pred,args.num_classes)
        #colsum_cm_norm = tf.div(tf.reduce_sum(colsum_cm),colsum_cm) / 100

    stream_vars = [i for i in tf.global_variables() if i.name.split('/')[0] == 'metrics']
    print('Num metrics variables: %d\n' % (len(stream_vars)))
    print(stream_vars[0].name)
    reset_op = [tf.variables_initializer(stream_vars)]

    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / (40000+args.num_steps)), args.power))

    opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
    opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, args.momentum)
    opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, args.momentum)

    # Define a variable to accumulate gradients.
    #accum_grads = [tf.Variable(tf.zeros_like(v.initialized_value()),
    #                           trainable=False) for v in conv_trainable + fc_w_trainable + fc_b_trainable]
    accum_grads = [tf.Variable(tf.zeros_like(v.initialized_value()),
                               trainable=False) for v in fc_w_trainable + fc_b_trainable]

    # Define an operation to clear the accumulated gradients for next batch.
    zero_op = [v.assign(tf.zeros_like(v)) for v in accum_grads]

    # Compute gradients.
    #grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
    grads = tf.gradients(reduced_loss, fc_w_trainable + fc_b_trainable)

    # Accumulate and normalise the gradients.
    accum_grads_op = [accum_grads[i].assign_add(grad / args.grad_update_every) for i, grad in
                       enumerate(grads)]

    #grads_conv = accum_grads[:len(conv_trainable)]
    #grads_fc_w = accum_grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
    #grads_fc_b = accum_grads[(len(conv_trainable) + len(fc_w_trainable)):]
    grads_fc_w = accum_grads[:len(fc_w_trainable)]
    grads_fc_b = accum_grads[len(fc_w_trainable):]

    # Apply the gradients.
    # train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    # train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    # train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

    # train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)
    train_op = tf.group(train_op_fc_w, train_op_fc_b)


    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

    # Load variables if the checkpoint is provided.
    if args.restore_from is not None:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.restore_from)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(40001,40000+args.num_steps):
        start_time = time.time()
        feed_dict = { step_ph : step }
        loss_value = 0

        # Clear the accumulated gradients.
        sess.run(zero_op, feed_dict=feed_dict)

        # Accumulate gradients.
        for i in range(args.grad_update_every):
            _,_,l_val, = sess.run([accum_grads_op, update_iou_op, reduced_loss], feed_dict=feed_dict)
            loss_value += l_val

        # Normalise the loss.
        loss_value /= args.grad_update_every

        # Apply gradients.
        if step % args.save_pred_every == 0:
            images, labels, summaryT, _ = sess.run([image_batch, label_batch, total_summary, train_op], feed_dict=feed_dict)
            train_summary_writer.add_summary(summaryT,step)
            save(saver, sess, args.snapshot_dir, step)
        else:
            sess.run(train_op, feed_dict=feed_dict)

        if step % args.save_pred_every == 0:
            cm_summaryT,acc_summaryT,miou_summaryT,iou_summaryT,tot_sumT=sess.run([cm_summary,acc_summary,miou_summary,iou_summary,tot_sum])
            #print('Total cm sum: %f\n' %(tot_sumT))
            train_summary_writer.add_summary(cm_summaryT, step)
            train_summary_writer.add_summary(acc_summaryT, step)
            train_summary_writer.add_summary(miou_summaryT, step)
            train_summary_writer.add_summary(iou_summaryT, step)
            sess.run(reset_op)

        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    main()
