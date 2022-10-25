import tensorflow as tf


def conv2d(x, w, stride=[1, 1, 1, 1], pad="SAME"):
    return tf.nn.conv2d(x, w, strides=stride, padding=pad)


def max_pooling(x, name, k_size=[1, 3, 3, 1], stride=[1, 2, 2, 1]):
    return tf.nn.max_pool(x, ksize=k_size, strides=stride, padding="VALID", name=name)


def weight_variables(name, shape):
    return tf.compat.v1.get_variable(
        name=name,
        shape=shape,
        dtype=tf.float32,
        initializer=tf.keras.initializers.GlorotUniform(),
    )


def bias_variables(name, shape, ini_val=0):
    return tf.compat.v1.get_variable(
        name=name,
        shape=shape,
        dtype=tf.float32,
        initializer=tf.constant_initializer(ini_val),
    )


def batch_norm(input, phase):
    return tf.compat.v1.layers.batch_normalization(
        input, axis=-1, center=True, scale=True, training=phase
    )


def read_and_decode_pairs_key(
    tfrecords_file, batch_size, name, n_classes, patch_size=256, num_threads=8
):

    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    img_features = tf.parse_single_example(
        serialized_example,
        features={
            "label": tf.FixedLenFeature([], tf.int64),
            "key": tf.FixedLenFeature([], tf.int64),
            "patch1": tf.FixedLenFeature([], tf.string),
            "patch2": tf.FixedLenFeature([], tf.string),
        },
    )

    patch1 = tf.decode_raw(img_features["patch1"], tf.uint8)
    patch1 = tf.cast(patch1, tf.float32)
    patch1 = tf.reshape(patch1, [patch_size, patch_size, 3])

    patch2 = tf.decode_raw(img_features["patch2"], tf.uint8)
    patch2 = tf.cast(patch2, tf.float32)
    patch2 = tf.reshape(patch2, [patch_size, patch_size, 3])

    label = tf.cast(img_features["label"], tf.int32)
    key = tf.cast(img_features["key"], tf.int32)

    patch1_batch, patch2_batch, label_batch, key_batch = tf.train.batch(
        [patch1, patch2, label, key],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=2000,
        name=name,
    )

    label_batch = tf.one_hot(label_batch, depth=n_classes)

    return (
        patch1_batch,
        patch2_batch,
        tf.reshape(label_batch, [batch_size, n_classes]),
        key_batch,
    )


def read_and_decode_pairs(
    tfrecords_file, batch_size, name, n_classes, patch_size=256, num_threads=8
):

    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    img_features = tf.parse_single_example(
        serialized_example,
        features={
            "label": tf.FixedLenFeature([], tf.int64),
            "patch1": tf.FixedLenFeature([], tf.string),
            "patch2": tf.FixedLenFeature([], tf.string),
        },
    )

    patch1 = tf.decode_raw(img_features["patch1"], tf.uint8)
    patch1 = tf.cast(patch1, tf.float32)
    patch1 = tf.reshape(patch1, [patch_size, patch_size, 3])

    patch2 = tf.decode_raw(img_features["patch2"], tf.uint8)
    patch2 = tf.cast(patch2, tf.float32)
    patch2 = tf.reshape(patch2, [patch_size, patch_size, 3])

    label = tf.cast(img_features["label"], tf.int32)

    patch1_batch, patch2_batch, label_batch = tf.train.batch(
        [patch1, patch2, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=2000,
        name=name,
    )

    label_batch = tf.one_hot(label_batch, depth=n_classes)

    return patch1_batch, patch2_batch, tf.reshape(label_batch, [batch_size, n_classes])


def MISLNet(x, phase, nprefilt=3, reuse=False):
    with tf.compat.v1.variable_scope("MISLNet", reuse=reuse) as scope:
        w_constr = weight_variables(name="weights_cstr", shape=[5, 5, 3, nprefilt])
        constr_output = conv2d(x, w_constr, pad="VALID")

        w_1 = weight_variables(name="weights_conv1", shape=[7, 7, nprefilt, 96])
        b_1 = bias_variables(name="bias_conv1", shape=[96], ini_val=0.0)
        conv1 = conv2d(constr_output, w_1, stride=[1, 2, 2, 1])
        conv1 = tf.nn.bias_add(conv1, b_1)
        bn1 = batch_norm(conv1, phase)
        bn1 = tf.nn.tanh(bn1, "tanh_conv1")
        conv1_output = max_pooling(bn1, name="maxpool_conv1")

        w_2 = weight_variables(name="weights_conv2", shape=[5, 5, 96, 64])
        b_2 = bias_variables(name="bias_conv2", shape=[64], ini_val=0.0)
        conv2 = conv2d(conv1_output, w_2)
        conv2 = tf.nn.bias_add(conv2, b_2)
        bn2 = batch_norm(conv2, phase)
        bn2 = tf.nn.tanh(bn2, "tanh_conv2")
        conv2_output = max_pooling(bn2, name="maxpool_conv2")

        w_3 = weight_variables(name="weights_conv3", shape=[5, 5, 64, 64])
        b_3 = bias_variables(name="bias_conv3", shape=[64], ini_val=0.0)
        conv3 = conv2d(conv2_output, w_3)
        conv3 = tf.nn.bias_add(conv3, b_3)
        bn3 = batch_norm(conv3, phase)
        bn3 = tf.nn.tanh(bn3, "tanh_conv3")
        conv3_output = max_pooling(bn3, name="maxpool_conv3")

        w_4 = weight_variables(name="weights_conv4", shape=[1, 1, 64, 128])
        b_4 = bias_variables(name="bias_conv4", shape=[128], ini_val=0.0)
        conv4 = conv2d(conv3_output, w_4)
        conv4 = tf.nn.bias_add(conv4, b_4)
        bn4 = batch_norm(conv4, phase)
        bn4 = tf.nn.tanh(bn4, "tanh_conv4")
        conv4_output = max_pooling(bn4, name="maxpool_conv4")

        reshaped_output = tf.reshape(
            conv4_output, [-1, 6 * 6 * 128]
        )  # should figure out how to calculate this dimension

        w_fc1 = weight_variables(name="weights_d1", shape=[6 * 6 * 128, 200])
        b_fc1 = tf.compat.v1.get_variable(
            name="bias_d1",
            shape=[200],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fc1 = tf.nn.tanh(
            tf.add(tf.matmul(reshaped_output, w_fc1), b_fc1), name="dense1_out"
        )

        w_fc2 = weight_variables(name="weights_d2", shape=[200, 200])
        b_fc2 = tf.compat.v1.get_variable(
            name="bias_d2",
            shape=[200],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fc2 = tf.nn.tanh(tf.add(tf.matmul(fc1, w_fc2), b_fc2), name="dense2_out")

    return fc2


def MISLNet128(x, phase, nprefilt=3, reuse=False):
    with tf.compat.v1.variable_scope("MISLNet", reuse=reuse) as scope:
        w_constr = weight_variables(name="weights_cstr", shape=[5, 5, 3, nprefilt])
        constr_output = conv2d(x, w_constr, pad="VALID")

        w_1 = weight_variables(name="weights_conv1", shape=[7, 7, nprefilt, 96])
        b_1 = bias_variables(name="bias_conv1", shape=[96], ini_val=0.0)
        conv1 = conv2d(constr_output, w_1, stride=[1, 2, 2, 1])
        conv1 = tf.nn.bias_add(conv1, b_1)
        bn1 = batch_norm(conv1, phase)
        tanh1 = tf.nn.tanh(bn1, "tanh_conv1")
        conv1_output = max_pooling(tanh1, name="maxpool_conv1")

        w_2 = weight_variables(name="weights_conv2", shape=[5, 5, 96, 64])
        b_2 = bias_variables(name="bias_conv2", shape=[64], ini_val=0.0)
        conv2 = conv2d(conv1_output, w_2)
        conv2 = tf.nn.bias_add(conv2, b_2)
        bn2 = batch_norm(conv2, phase)
        bn2 = tf.nn.tanh(bn2, "tanh_conv2")
        conv2_output = max_pooling(bn2, name="maxpool_conv2")

        w_3 = weight_variables(name="weights_conv3", shape=[5, 5, 64, 64])
        b_3 = bias_variables(name="bias_conv3", shape=[64], ini_val=0.0)
        conv3 = conv2d(conv2_output, w_3)
        conv3 = tf.nn.bias_add(conv3, b_3)
        bn3 = batch_norm(conv3, phase)
        bn3 = tf.nn.tanh(bn3, "tanh_conv3")
        conv3_output = max_pooling(bn3, name="maxpool_conv3")

        w_4 = weight_variables(name="weights_conv4", shape=[1, 1, 64, 128])
        b_4 = bias_variables(name="bias_conv4", shape=[128], ini_val=0.0)
        conv4 = conv2d(conv3_output, w_4)
        conv4 = tf.nn.bias_add(conv4, b_4)
        bn4 = batch_norm(conv4, phase)
        bn4 = tf.nn.tanh(bn4, "tanh_conv4")
        conv4_output = max_pooling(bn4, name="maxpool_conv4")

        reshaped_output = tf.reshape(
            conv4_output, [-1, 2 * 2 * 128]
        )  # should figure out how to calculate this dimension

        w_fc1 = weight_variables(name="weights_d1", shape=[2 * 2 * 128, 200])
        b_fc1 = tf.compat.v1.get_variable(
            name="bias_d1",
            shape=[200],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fc1 = tf.nn.tanh(
            tf.add(tf.matmul(reshaped_output, w_fc1), b_fc1), name="dense1_out"
        )

        w_fc2 = weight_variables(name="weights_d2", shape=[200, 200])
        b_fc2 = tf.compat.v1.get_variable(
            name="bias_d2",
            shape=[200],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fc2 = tf.nn.tanh(tf.add(tf.matmul(fc1, w_fc2), b_fc2), name="dense2_out")

    return fc2


def MISLNet64(x, phase, nprefilt=3, reuse=False):
    with tf.compat.v1.variable_scope("MISLNet", reuse=reuse) as scope:
        w_constr = weight_variables(name="weights_cstr", shape=[5, 5, 3, nprefilt])
        constr_output = conv2d(x, w_constr, pad="VALID")

        w_1 = weight_variables(name="weights_conv1", shape=[7, 7, nprefilt, 96])
        b_1 = bias_variables(name="bias_conv1", shape=[96], ini_val=0.0)
        conv1 = conv2d(constr_output, w_1, stride=[1, 2, 2, 1])
        conv1 = tf.nn.bias_add(conv1, b_1)
        bn1 = batch_norm(conv1, phase)
        bn1 = tf.nn.tanh(bn1, "tanh_conv1")
        conv1_output = max_pooling(bn1, name="maxpool_conv1")

        w_2 = weight_variables(name="weights_conv2", shape=[5, 5, 96, 64])
        b_2 = bias_variables(name="bias_conv2", shape=[64], ini_val=0.0)
        conv2 = conv2d(conv1_output, w_2)
        conv2 = tf.nn.bias_add(conv2, b_2)
        bn2 = batch_norm(conv2, phase)
        bn2 = tf.nn.tanh(bn2, "tanh_conv2")
        conv2_output = max_pooling(bn2, name="maxpool_conv2")

        w_3 = weight_variables(name="weights_conv3", shape=[5, 5, 64, 64])
        b_3 = bias_variables(name="bias_conv3", shape=[64], ini_val=0.0)
        conv3 = conv2d(conv2_output, w_3)
        conv3 = tf.nn.bias_add(conv3, b_3)
        bn3 = batch_norm(conv3, phase)
        bn3 = tf.nn.tanh(bn3, "tanh_conv3")
        conv3_output = max_pooling(bn3, name="maxpool_conv3")

        w_4 = weight_variables(name="weights_conv4", shape=[1, 1, 64, 128])
        b_4 = bias_variables(name="bias_conv4", shape=[128], ini_val=0.0)
        conv4 = conv2d(conv3_output, w_4)
        conv4 = tf.nn.bias_add(conv4, b_4)
        bn4 = batch_norm(conv4, phase)
        bn4 = tf.nn.tanh(bn4, "tanh_conv4")
        # conv4_output = max_pooling(bn4, name='maxpool_conv4')
        conv4_output = bn4  # remove pooling to get dimensions right

        # reshaped_output = tf.reshape(conv4_output, [-1, 6 * 6 * 128])
        reshaped_output = tf.reshape(conv4_output, [-1, 2 * 4 * 64])

        # w_fc1 = weight_variables(name="weights_d1", shape=[6 * 6 * 128, 200])
        w_fc1 = weight_variables(name="weights_d1", shape=[2 * 4 * 64, 200])
        b_fc1 = tf.compat.v1.get_variable(
            name="bias_d1",
            shape=[200],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fc1 = tf.nn.tanh(
            tf.add(tf.matmul(reshaped_output, w_fc1), b_fc1), name="dense1_out"
        )

        w_fc2 = weight_variables(name="weights_d2", shape=[200, 200])
        b_fc2 = tf.compat.v1.get_variable(
            name="bias_d2",
            shape=[200],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fc2 = tf.nn.tanh(tf.add(tf.matmul(fc1, w_fc2), b_fc2), name="dense2_out")

    return fc2


def MISLNet96(x, phase, nprefilt=3, reuse=False):
    with tf.compat.v1.variable_scope("MISLNet", reuse=reuse) as scope:
        w_constr = weight_variables(name="weights_cstr", shape=[5, 5, 3, nprefilt])
        constr_output = conv2d(x, w_constr, pad="VALID")

        w_1 = weight_variables(name="weights_conv1", shape=[7, 7, nprefilt, 96])
        b_1 = bias_variables(name="bias_conv1", shape=[96], ini_val=0.0)
        conv1 = conv2d(constr_output, w_1, stride=[1, 2, 2, 1])
        conv1 = tf.nn.bias_add(conv1, b_1)
        bn1 = batch_norm(conv1, phase)
        bn1 = tf.nn.tanh(bn1, "tanh_conv1")
        conv1_output = max_pooling(bn1, name="maxpool_conv1")

        w_2 = weight_variables(name="weights_conv2", shape=[5, 5, 96, 64])
        b_2 = bias_variables(name="bias_conv2", shape=[64], ini_val=0.0)
        conv2 = conv2d(conv1_output, w_2)
        conv2 = tf.nn.bias_add(conv2, b_2)
        bn2 = batch_norm(conv2, phase)
        bn2 = tf.nn.tanh(bn2, "tanh_conv2")
        conv2_output = max_pooling(bn2, name="maxpool_conv2")

        w_3 = weight_variables(name="weights_conv3", shape=[5, 5, 64, 64])
        b_3 = bias_variables(name="bias_conv3", shape=[64], ini_val=0.0)
        conv3 = conv2d(conv2_output, w_3)
        conv3 = tf.nn.bias_add(conv3, b_3)
        bn3 = batch_norm(conv3, phase)
        bn3 = tf.nn.tanh(bn3, "tanh_conv3")
        conv3_output = max_pooling(bn3, name="maxpool_conv3")

        w_4 = weight_variables(name="weights_conv4", shape=[1, 1, 64, 128])
        b_4 = bias_variables(name="bias_conv4", shape=[128], ini_val=0.0)
        conv4 = conv2d(conv3_output, w_4)
        conv4 = tf.nn.bias_add(conv4, b_4)
        bn4 = batch_norm(conv4, phase)
        bn4 = tf.nn.tanh(bn4, "tanh_conv4")
        # conv4_output = max_pooling(bn4, name='maxpool_conv4')
        conv4_output = bn4  # remove pooling to get dimensions right

        # reshaped_output = tf.reshape(conv4_output, [-1, 6 * 6 * 128])
        reshaped_output = tf.reshape(conv4_output, [-1, 8 * 4 * 64])

        # w_fc1 = weight_variables(name="weights_d1", shape=[6 * 6 * 128, 200])
        w_fc1 = weight_variables(name="weights_d1", shape=[8 * 4 * 64, 200])
        b_fc1 = tf.compat.v1.get_variable(
            name="bias_d1",
            shape=[200],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fc1 = tf.nn.tanh(
            tf.add(tf.matmul(reshaped_output, w_fc1), b_fc1), name="dense1_out"
        )

        w_fc2 = weight_variables(name="weights_d2", shape=[200, 200])
        b_fc2 = tf.compat.v1.get_variable(
            name="bias_d2",
            shape=[200],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fc2 = tf.nn.tanh(tf.add(tf.matmul(fc1, w_fc2), b_fc2), name="dense2_out")

    return fc2


def CompareNet_v1_256(x1, x2, phase, nprefilt=6, nb12=2048, nb3=64):

    MISL_output1 = MISLNet(
        x1, phase, nprefilt=nprefilt, reuse=False
    )  # feature extractor for patch 1
    MISL_output2 = MISLNet(
        x2, phase, nprefilt=nprefilt, reuse=True
    )  # feature extractor for patch 2, with same weights as 1

    with tf.compat.v1.variable_scope("CompareNet") as scope:
        w_fcb1 = weight_variables(name="weights_fcb1", shape=[200, nb12])
        b_fcb1 = tf.compat.v1.get_variable(
            name="bias_fcb1",
            shape=[nb12],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fcb1 = tf.nn.relu(
            tf.add(tf.matmul(MISL_output1, w_fcb1), b_fcb1), name="fcb1_out"
        )
        fcb1_drop = tf.nn.dropout(fcb1, 0.5)

        fcb2 = tf.nn.relu(
            tf.add(tf.matmul(MISL_output2, w_fcb1), b_fcb1), name="fcb2_out"
        )
        fcb2_drop = tf.nn.dropout(fcb2, 0.5)

        fcb1b2_mult = tf.multiply(fcb1, fcb2, name="fcb1b2_mult")
        fcb1b2_concat = tf.concat([fcb1, fcb1b2_mult, fcb2], 1)

        w_fcb3 = weight_variables(name="weights_fcb3", shape=[nb12 * 3, nb3])
        b_fcb3 = tf.compat.v1.get_variable(
            name="bias_fcb3",
            shape=[nb3],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fcb3 = tf.nn.relu(
            tf.add(tf.matmul(fcb1b2_concat, w_fcb3), b_fcb3), name="fcb3_out"
        )

        w_out = weight_variables(name="weights_out", shape=[nb3, 2])
        b_out = tf.compat.v1.get_variable(
            name="bias_out",
            shape=[2],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        output = tf.matmul(fcb3, w_out) + b_out

    return output


def CompareNet_v1_128(x1, x2, phase, nprefilt=6, nb12=2048, nb3=64):

    MISL_output1 = MISLNet128(
        x1, phase, nprefilt=nprefilt, reuse=False
    )  # feature extractor for patch 1
    MISL_output2 = MISLNet128(
        x2, phase, nprefilt=nprefilt, reuse=True
    )  # feature extractor for patch 2, with same weights as 1

    with tf.compat.v1.variable_scope("CompareNet") as scope:
        w_fcb1 = weight_variables(name="weights_fcb1", shape=[200, nb12])
        b_fcb1 = tf.compat.v1.get_variable(
            name="bias_fcb1",
            shape=[nb12],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fcb1 = tf.nn.relu(
            tf.add(tf.matmul(MISL_output1, w_fcb1), b_fcb1), name="fcb1_out"
        )
        fcb1_drop = tf.nn.dropout(fcb1, 0.5)

        fcb2 = tf.nn.relu(
            tf.add(tf.matmul(MISL_output2, w_fcb1), b_fcb1), name="fcb2_out"
        )
        fcb2_drop = tf.nn.dropout(fcb2, 0.5)

        fcb1b2_mult = tf.multiply(fcb1, fcb2, name="fcb1b2_mult")
        fcb1b2_concat = tf.concat([fcb1, fcb1b2_mult, fcb2], 1)

        w_fcb3 = weight_variables(name="weights_fcb3", shape=[nb12 * 3, nb3])
        b_fcb3 = tf.compat.v1.get_variable(
            name="bias_fcb3",
            shape=[nb3],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fcb3 = tf.nn.relu(
            tf.add(tf.matmul(fcb1b2_concat, w_fcb3), b_fcb3), name="fcb3_out"
        )

        w_out = weight_variables(name="weights_out", shape=[nb3, 2])
        b_out = tf.compat.v1.get_variable(
            name="bias_out",
            shape=[2],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        output = tf.matmul(fcb3, w_out) + b_out

    return output


def CompareNet_v1_96(x1, x2, phase, nprefilt=6, nb12=2048, nb3=64):

    MISL_output1 = MISLNet96(
        x1, phase, nprefilt=nprefilt, reuse=False
    )  # feature extractor for patch 1
    MISL_output2 = MISLNet96(
        x2, phase, nprefilt=nprefilt, reuse=True
    )  # feature extractor for patch 2, with same weights as 1

    with tf.compat.v1.variable_scope("CompareNet") as scope:
        w_fcb1 = weight_variables(name="weights_fcb1", shape=[200, nb12])
        b_fcb1 = tf.compat.v1.get_variable(
            name="bias_fcb1",
            shape=[nb12],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fcb1 = tf.nn.relu(
            tf.add(tf.matmul(MISL_output1, w_fcb1), b_fcb1), name="fcb1_out"
        )
        fcb1_drop = tf.nn.dropout(fcb1, 0.5)

        fcb2 = tf.nn.relu(
            tf.add(tf.matmul(MISL_output2, w_fcb1), b_fcb1), name="fcb2_out"
        )
        fcb2_drop = tf.nn.dropout(fcb2, 0.5)

        fcb1b2_mult = tf.multiply(fcb1, fcb2, name="fcb1b2_mult")
        fcb1b2_concat = tf.concat([fcb1, fcb1b2_mult, fcb2], 1)

        w_fcb3 = weight_variables(name="weights_fcb3", shape=[nb12 * 3, nb3])
        b_fcb3 = tf.compat.v1.get_variable(
            name="bias_fcb3",
            shape=[nb3],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fcb3 = tf.nn.relu(
            tf.add(tf.matmul(fcb1b2_concat, w_fcb3), b_fcb3), name="fcb3_out"
        )

        w_out = weight_variables(name="weights_out", shape=[nb3, 2])
        b_out = tf.compat.v1.get_variable(
            name="bias_out",
            shape=[2],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        output = tf.matmul(fcb3, w_out) + b_out

    return output


def CompareNet_v1_64(x1, x2, phase, nprefilt=6, nb12=2048, nb3=64):

    MISL_output1 = MISLNet64(
        x1, phase, nprefilt=nprefilt, reuse=False
    )  # feature extractor for patch 1
    MISL_output2 = MISLNet64(
        x2, phase, nprefilt=nprefilt, reuse=True
    )  # feature extractor for patch 2, with same weights as 1

    with tf.compat.v1.variable_scope("CompareNet") as scope:
        w_fcb1 = weight_variables(name="weights_fcb1", shape=[200, nb12])
        b_fcb1 = tf.compat.v1.get_variable(
            name="bias_fcb1",
            shape=[nb12],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fcb1 = tf.nn.relu(
            tf.add(tf.matmul(MISL_output1, w_fcb1), b_fcb1), name="fcb1_out"
        )
        fcb1_drop = tf.nn.dropout(fcb1, 0.5)

        fcb2 = tf.nn.relu(
            tf.add(tf.matmul(MISL_output2, w_fcb1), b_fcb1), name="fcb2_out"
        )
        fcb2_drop = tf.nn.dropout(fcb2, 0.5)

        fcb1b2_mult = tf.multiply(fcb1, fcb2, name="fcb1b2_mult")
        fcb1b2_concat = tf.concat([fcb1, fcb1b2_mult, fcb2], 1)

        w_fcb3 = weight_variables(name="weights_fcb3", shape=[nb12 * 3, nb3])
        b_fcb3 = tf.compat.v1.get_variable(
            name="bias_fcb3",
            shape=[nb3],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fcb3 = tf.nn.relu(
            tf.add(tf.matmul(fcb1b2_concat, w_fcb3), b_fcb3), name="fcb3_out"
        )

        w_out = weight_variables(name="weights_out", shape=[nb3, 2])
        b_out = tf.compat.v1.get_variable(
            name="bias_out",
            shape=[2],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        output = tf.matmul(fcb3, w_out) + b_out

    return output


def prefeat_CompareNet_v1(f1, f2, nb12=2048, nb3=64):

    with tf.compat.v1.variable_scope("CompareNet") as scope:
        w_fcb1 = weight_variables(name="weights_fcb1", shape=[200, nb12])
        b_fcb1 = tf.compat.v1.get_variable(
            name="bias_fcb1",
            shape=[nb12],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )

        fcb1 = tf.nn.relu(tf.add(tf.matmul(f1, w_fcb1), b_fcb1), name="fcb1_out")
        fcb2 = tf.nn.relu(tf.add(tf.matmul(f2, w_fcb1), b_fcb1), name="fcb2_out")

        fcb1b2_mult = tf.multiply(fcb1, fcb2, name="fcb1b2_mult")
        fcb1b2_concat = tf.concat([fcb1, fcb1b2_mult, fcb2], 1)

        w_fcb3 = weight_variables(name="weights_fcb3", shape=[nb12 * 3, nb3])
        b_fcb3 = tf.compat.v1.get_variable(
            name="bias_fcb3",
            shape=[nb3],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        fcb3 = tf.nn.relu(
            tf.add(tf.matmul(fcb1b2_concat, w_fcb3), b_fcb3), name="fcb3_out"
        )

        w_out = weight_variables(name="weights_out", shape=[nb3, 2])
        b_out = tf.compat.v1.get_variable(
            name="bias_out",
            shape=[2],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
        )
        output = tf.matmul(fcb3, w_out) + b_out

    return output
