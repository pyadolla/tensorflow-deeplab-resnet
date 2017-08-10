import tensorflow as tf

# DATA
train_items = ["train_file_{}".format(i) for i in range(6)]
valid_items = ["valid_file_{}".format(i) for i in range(3)]

# SETTINGS
batch_size = 3
batches_per_epoch = 1
epochs = 10

# ------------------------------------------------
#                                            GRAPH
# ------------------------------------------------
graph = tf.Graph()
with graph.as_default():
    # TRAIN QUEUE
    train_q = tf.train.string_input_producer(train_items, shuffle=False)

    # VALID/TEST QUEUE
    test_q = tf.train.string_input_producer(valid_items, shuffle=False)

    # SELECT QUEUE
    is_training = tf.placeholder(tf.bool, shape=None, name="is_training")
    q_selector = tf.cond(is_training,
                         lambda: tf.constant(0),
                         lambda: tf.constant(1))

    # select_q = tf.placeholder(tf.int32, [])
    q = tf.QueueBase.from_list(q_selector, [train_q, test_q])

    # # Create batch of items.
    data = q.dequeue_many(batch_size)


# ------------------------------------------------
#                                          SESSION
# ------------------------------------------------
with tf.Session(graph=graph) as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Start populating the queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    try:
        for epoch in range(epochs):
            print("-" * 60)
            # TRAIN
            for step in range(batches_per_epoch):
                if coord.should_stop():
                    break
                print("TRAIN.dequeue = " + str(sess.run(data, {is_training: True})))

            # VALIDATION
            print("\nVALID.dequeue = " + str(sess.run(data, {is_training: False})))
    except Exception as e:
        coord.request_stop(e)

    finally:
        coord.request_stop()
        coord.join(threads)
