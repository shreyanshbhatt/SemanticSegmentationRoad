import os.path
import numpy as np
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import timeit

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_data = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return input_data, keep, l3out, l4out, l7out


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    std_dev = 0.01
    # TODO: Implement function
    # add a 1x1 convolution
    layer81x1 = tf.layers.conv2d(
        vgg_layer7_out,
        num_classes,
        1,
        padding='SAME',
        kernel_initializer=tf.random_normal_initializer(stddev=std_dev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
    # apply a conv2d transpose with 4 kernel size and (2, 2) strides
    print(layer81x1)
    layer9 = tf.layers.conv2d_transpose(
        layer81x1,
        num_classes,
        4,
        strides=(2, 2),
        padding='SAME',
        kernel_initializer=tf.random_normal_initializer(stddev=std_dev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
    print(layer9)
    vgg_layer4_scaled = tf.multiply(
        vgg_layer4_out, 0.001, name='pool4_out_scaled')
    # Add a skip laye
    vgg_layer4_1x1 = tf.layers.conv2d(
        vgg_layer4_scaled,
        num_classes,
        1,
        padding='SAME',
        kernel_initializer=tf.random_normal_initializer(stddev=std_dev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
    print(vgg_layer4_1x1)
    layer9_added = tf.add(layer9, vgg_layer4_1x1)
    layer10 = tf.layers.conv2d_transpose(
        layer9_added,
        num_classes,
        4,
        strides=(2, 2),
        padding='SAME',
        kernel_initializer=tf.random_normal_initializer(stddev=std_dev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
    vgg_layer3_scaled = tf.multiply(
        vgg_layer3_out, 0.001, name='pool3_out_scaled')
    vgg_layer3_1x1 = tf.layers.conv2d(
        vgg_layer3_scaled,
        num_classes,
        1,
        padding='SAME',
        kernel_initializer=tf.random_normal_initializer(stddev=std_dev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
    layer10_added = tf.add(layer10, vgg_layer3_1x1)
    output_layer = tf.layers.conv2d_transpose(
        layer10_added,
        num_classes,
        16,
        strides=(8, 8),
        padding='SAME',
        kernel_initializer=tf.random_normal_initializer(stddev=std_dev),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
        name='Softmax')
    return output_layer


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(cross_entropy_loss +
                                            1.0 * sum(reg_losses))
    return logits, training_operation, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image, correct_label, keep_prob,
             learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())
    losses = []
    for epoch in range(epochs):
        for img, label in get_batches_fn(batch_size):
            _, loss = sess.run(
                [train_op, cross_entropy_loss],
                feed_dict={
                    input_image: img,
                    correct_label: label,
                    keep_prob: 0.5,
                    learning_rate: 0.0001
                })
            losses.append(loss)
        print(np.mean(losses))
    pass


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road/training'), image_shape)
        epochs = 50
        batch_size = 8
        labels = tf.placeholder(
            tf.int32, [None, None, None, num_classes], name='labels')
        learning_rate = tf.placeholder(tf.float32, name='l_rate')
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(
            sess, vgg_path)
        last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        # TODO: Train NN using the train_nn function
        logits, train_op, cross_entropy_loss = optimize(
            last_layer, labels, learning_rate, num_classes)
        pred_softmax = tf.nn.softmax(logits, name="prediction_softmax")
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
                 cross_entropy_loss, image_input, labels, keep_prob, learning_rate)
        saver = tf.train.Saver(tf.all_variables())
        saver.save(sess, './model/ckpt', write_meta_graph=True)
        # TODO: Save inference data using helper.save_inference_samples
        # image inference code in a diffrent file
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

if __name__ == '__main__':
    run()
