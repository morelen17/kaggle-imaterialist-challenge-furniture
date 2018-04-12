import random
from multiprocessing import cpu_count

import tensorflow as tf
from tqdm import tqdm

from params import *


class WRN(object):
    def __init__(self, depth: int = 16, k: int = 8, is_training: bool = True):
        assert ((depth - 4) % 6 == 0)

        self._is_training = is_training
        self._n = (depth - 4) // 6  # is 2 for depth=16; is 4 for depth=28
        self._n_stages = [16, 16 * k, 32 * k, 64 * k]

    def _wide_basic(self, inputs, n_input_plane, n_output_plane, stride):
        n_bottleneck_plane = n_output_plane

        if n_input_plane != n_output_plane:
            inputs = tf.layers.batch_normalization(inputs, training=self._is_training)
            inputs = tf.nn.relu(inputs)
            convs = inputs
        else:
            convs = tf.layers.batch_normalization(inputs, training=self._is_training)
            convs = tf.nn.relu(convs)

        convs = tf.layers.conv2d(convs,
                                 filters=n_bottleneck_plane,
                                 kernel_size=(3, 3),
                                 strides=stride,
                                 padding='same')

        convs = tf.layers.batch_normalization(convs, training=self._is_training)
        convs = tf.nn.relu(convs)
        convs = tf.layers.conv2d(convs,
                                 filters=n_bottleneck_plane,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding='same')

        if n_input_plane != n_output_plane:
            shortcut = tf.layers.conv2d(inputs,
                                        filters=n_output_plane,
                                        kernel_size=(1, 1),
                                        strides=stride,
                                        padding='same')
        else:
            shortcut = inputs

        return tf.add(convs, shortcut)

    def _block(self, inputs, n_input_plane, n_output_plane, count, stride):
        inputs = self._wide_basic(inputs, n_input_plane, n_output_plane, stride)
        for i in range(2, count + 1):
            inputs = self._wide_basic(inputs, n_input_plane, n_output_plane, (1, 1))
        return inputs

    def build(self, inputs):
        conv1 = tf.layers.conv2d(inputs,
                                 filters=self._n_stages[0],
                                 kernel_size=(3, 3),
                                 padding='same')

        conv2 = self._block(conv1,
                            n_input_plane=self._n_stages[0],
                            n_output_plane=self._n_stages[1],
                            count=self._n,
                            stride=(1, 1))

        conv3 = self._block(conv2,
                            n_input_plane=self._n_stages[1],
                            n_output_plane=self._n_stages[2],
                            count=self._n,
                            stride=(2, 2))

        conv4 = self._block(conv3,
                            n_input_plane=self._n_stages[2],
                            n_output_plane=self._n_stages[3],
                            count=self._n,
                            stride=(2, 2))

        normalized = tf.layers.batch_normalization(conv4, training=self._is_training)
        relued = tf.nn.relu(normalized)

        pooled = tf.layers.average_pooling2d(relued,
                                             pool_size=(8, 8),
                                             strides=(1, 1),
                                             padding='same')
        flattened = tf.layers.flatten(pooled)
        self.logits = tf.layers.dense(flattened, units=NUM_CLASS)
        return self.logits

    def add_loss(self, ground_truth, global_step):
        loss_ops = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=ground_truth))
        learning_rate = tf.train.exponential_decay(INIT_LEARNING_RATE, global_step, 100000, 0.5)
        train_ops = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_ops)
        return loss_ops, train_ops


def _image_transform(image):
    image = tf.image.rot90(image, k=random.randint(0, 3))
    image = tf.image.random_brightness(image, max_delta=32 / 255)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    return tf.image.per_image_standardization(image)


def _map_func(line, data_folder: str):
    def cond_true():
        offset = tf.div(tf.subtract(image_h, image_w), 2)
        return tf.image.pad_to_bounding_box(image, 0, offset, image_h, image_h)

    def cond_false():
        offset = tf.div(tf.subtract(image_w, image_h), 2)
        return tf.image.pad_to_bounding_box(image, offset, 0, image_w, image_w)

    image_str, _, label = tf.decode_csv(line, record_defaults=[[""], [0], [0]])
    image_str = tf.string_join([data_folder, image_str])
    image = tf.image.decode_image(tf.read_file(image_str), channels=3)

    # .gif files have extra dimension that describes "frames", we take only first frame
    image = tf.cond(tf.equal(tf.rank(image), 4), lambda: image[0, :, :, :], lambda: image)

    # image must have only 3 channels
    image = image[:, :, :3]

    image_h = tf.shape(image)[0]
    image_w = tf.shape(image)[1]
    # pad image to square
    image = tf.cond(tf.greater(image_h, image_w), cond_true, cond_false)

    # resizing
    # image.set_shape([None, None, None])
    image = tf.image.resize_images(image, size=[IMAGE_SIZE, IMAGE_SIZE])
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

    # applying data augmentation
    image = _image_transform(image)

    # one-hot label encoding
    label_onehot = tf.one_hot(label, NUM_CLASS)
    return image, label_onehot


def train_dataset():
    train_data_folder = './data/images/train/'
    ds = tf.data.TextLineDataset(['./data/train_actual_extended.csv']).skip(1)
    ds = ds.shuffle(buffer_size=TRAIN_SET_ACTUAL_SIZE)  # or 10k is OKNOTOK ???
    ds = ds.map(lambda x: _map_func(x, train_data_folder), num_parallel_calls=cpu_count())
    # ds = ds.repeat(NUM_EPOCH)
    ds = ds.batch(BATCH_SIZE)
    return ds


def validation_dataset():
    train_data_folder = './data/images/validation/'
    ds = tf.data.TextLineDataset(['./data/validation_actual_extended.csv']).skip(1)
    ds = ds.shuffle(buffer_size=TRAIN_SET_ACTUAL_SIZE)  # or 10k is OKNOTOK ???
    ds = ds.map(lambda x: _map_func(x, train_data_folder), num_parallel_calls=cpu_count())
    ds = ds.batch(BATCH_SIZE)
    return ds


def add_stats(loss, accuracy):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    return tf.summary.merge_all()


def train():
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_ds = train_dataset()
    validation_ds = validation_dataset()
    iterator = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
    img_batch, label_batch = iterator.get_next()

    model = WRN()
    logits = model.build(img_batch)
    loss_op, train_op = model.add_loss(label_batch, global_step)
    accuracy_op = tf.metrics.accuracy(tf.argmax(logits, 1), tf.argmax(label_batch, 1))
    statistics = add_stats(loss_op, accuracy_op[1])

    training_init_op = iterator.make_initializer(train_ds)
    validation_init_op = iterator.make_initializer(validation_ds)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # to use tf.metrics.accuracy

        statistics_writer = tf.summary.FileWriter('logs-wrn', sess.graph)

        for epoch in range(1, NUM_EPOCH + 1):
            # train
            with tqdm() as counter:
                sess.run(training_init_op)
                step = 0
                try:
                    while True:
                        step, loss, accuracy, _ = sess.run([global_step, loss_op, accuracy_op, train_op])
                        counter.set_postfix({
                            "loss": "{:.6}".format(loss),
                            "accuracy": "{:.6}".format(accuracy[1])
                        })
                        counter.update(1)
                        if step % 500 == 0:
                            statistics_writer.add_summary(sess.run(statistics), step)

                except tf.errors.OutOfRangeError:
                    print("End of epoch # {}".format(epoch))
                    saver.save(sess, 'logs-wrn/model.ckpt', global_step=step)

            # validate
            with tqdm() as counter:
                sess.run(validation_init_op)
                try:
                    while True:
                        accuracy, _ = sess.run([accuracy_op, loss_op])
                        counter.set_postfix({
                            "accuracy": "{:.6}".format(accuracy[1])
                        })
                        counter.update(1)
                except tf.errors.OutOfRangeError:
                    print("End of validation after epoch # {}".format(epoch))


if __name__ == "__main__":
    train()
