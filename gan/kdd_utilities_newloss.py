  
import tensorflow as tf
import tensorflow_probability as tfp
import keras
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

"""Class for KDD10 percent GAN architecture.
Generator and discriminator.
"""

learning_rate_gen = 0.00001
learning_rate_gen_mine = 0.00001
learning_rate = 0.00001
batch_size = 50
layer = 1
latent_dim = 16
dis_inter_layer_dim = 128
init_kernel = tf.contrib.layers.xavier_initializer()

def generator(z_inp, is_training=False, getter=None, reuse=False):
    """ Generator architecture in tensorflow
    Generates data from the latent space
    Args:
        z_inp (tensor): variable in the latent space
        reuse (bool): sharing variables or not
    Returns:
        (tensor): last activation layer of the generator
    """
    #decoderx = tfk.Sequential([
    #            tfkl.InputLayer(input_shape= z_inp.shape),
    #            tfkl.Dense(32, activation ='relu',kernel_initializer = init_kernel),
    #            # tfkl.Dense(64, activation  = 'relu',kernel_initializer = init_kernel),
    #            # tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(29), activation='relu',kernel_initializer = init_kernel),
    #            # tfpl.MultivariateNormalTriL(29)
    #            tfkl.Dense(tfpl.IndependentNormal.params_size(39)),
    #            tfpl.IndependentNormal(39),
    #            # tfkl.Dense(39, activation  = 'relu',kernel_initializer = init_kernel)
    #            ])
#
#
    #return decoderx(z_inp)

    with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):

        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(z_inp,
                                  units=32,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net, name='relu')

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=64,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net, name='relu')

        name_net = 'layer_4'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=39,
                                  kernel_initializer=init_kernel,
                                  name='fc')

        return net





    #with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):
#
    #    name_net = 'layer_1'
    #    with tf.variable_scope(name_net):
    #        net = tf.layers.dense(z_inp,
    #                              units=32,
    #                              kernel_initializer=init_kernel,
    #                              name='fc')
    #        net = tf.nn.relu(net, name='relu')
#
    #    name_net = 'layer_2'
    #    with tf.variable_scope(name_net):
    #        net = tf.layers.dense(net,
    #                              units=64,
    #                              kernel_initializer=init_kernel,
    #                              name='fc')
    #        net = tf.nn.relu(net, name='relu')
#
    #    name_net = 'layer_4'
    #    with tf.variable_scope(name_net):
    #        net = tf.layers.dense(net,
    #                              units=39,
    #                              kernel_initializer=init_kernel,
    #                              name='fc')
#
    #    return net

def discriminator(x_inp, is_training=False, getter=None, reuse=False):
    """ Discriminator architecture in tensorflow
    Discriminates between real data and generated data
    Args:
        x_inp (tensor): input data for the encoder.
        reuse (bool): sharing variables or not
    Returns:
        logits (tensor): last activation layer of the discriminator (shape 1)
        intermediate_layer (tensor): intermediate layer for feature matching
    """
    with tf.variable_scope('discriminator', reuse=reuse, custom_getter=getter):

        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(x_inp,
                                  units=256,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = leakyReLu(net)
            net = tf.layers.dropout(net, rate=0.2, name='dropout',
                                  training=is_training)
        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=dis_inter_layer_dim,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = leakyReLu(net)
            net = tf.layers.dropout(net,
                                    rate=0.2,
                                    name='dropout',
                                    training=is_training)

        intermediate_layer = net

        name_net = 'layer_4'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=1,
                                  kernel_initializer=init_kernel,
                                  name='fc')

        net = tf.squeeze(net)

        return net, intermediate_layer

def leakyReLu(x, alpha=0.1, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))