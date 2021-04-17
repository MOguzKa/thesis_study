import tensorflow as tf
import tensorflow_probability as tfp
import keras
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

"""KDD BiGAN architecture.

decodera probability layer eklemeyi basardim
simdi sirada generator lossunu degistirmek var
onu da yaptim sanki daha iyi ogreniyor gibi. Gercekten ogrenip ogrenmedigini weightleri inceleyerek bakacagim 
gercekten degisim varsa o ise yariyor demektir. Mesela ilk gen egitiminden once ve sonra ve en son egitimden sonra weightleri yazdirip bakabilirim.
Veya 2.egitimdeki losslari silip sanki orda sadece encoderi egitiyor gibi davranip ilk lossta degisim olacak mi diye bakabilirim.
Haftasonu bunlari bitir...  

Generator (decoder), encoder and discriminator.

yaptim galiba calisyiyor

"""
# learning_rate_gen = 0.001
# learning_rate_gen_mine = 0.001
# learning_rate = 0.001

learning_rate_gen = 0.00001
learning_rate_gen_mine = 0.00001
learning_rate = 0.00001
batch_size = 50
layer = 1
latent_dim = 8
dis_inter_layer_dim = 128
init_kernel = tf.contrib.layers.xavier_initializer()

def encoder(x_inp, is_training=False, getter=None, reuse=False):
    """ Encoder architecture in tensorflow

    Maps the data into the latent space

    Args:
        x_inp (tensor): input data for the encoder.
        reuse (bool): sharing variables or not

    Returns:
        (tensor): last activation layer of the encoder

    """

    with tf.variable_scope('encoder', reuse=reuse, custom_getter=getter):

        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(x_inp,
                              units=32,
                              kernel_initializer=init_kernel,
                              name='fc')
            net = leakyReLu(net)

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(x_inp,
                              units=16,
                              kernel_initializer=init_kernel,
                              name='fc')
            net = leakyReLu(net)

        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                              units=latent_dim,
                              kernel_initializer=init_kernel,
                              name='fc')

    return net

def decoder(z_inp, is_training=False, getter=None, reuse=False):
    """ Decoder architecture in tensorflow

    Generates data from the latent space

    Args:
        z_inp (tensor): variable in the latent space
        reuse (bool): sharing variables or not

    Returns:
        (tensor): last activation layer of the generator

    """

    ############################

    #probability icin ilk returne kadar ac 2.returne kadar commentle
    # ayrica run kdd ` de loss+generator 1 ile ilgili her yeri commetten cikar
    # digeri icin de tam tersi silemleri yap

    ############################
    
    # decoderx = tfk.Sequential([
    # tfkl.InputLayer(input_shape= z_inp.shape),
    # tfkl.Dense(64, activation ='relu',kernel_initializer = init_kernel),
    # tfkl.Dense(32, activation  = 'relu',kernel_initializer = init_kernel),
    # # tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(29), activation='relu',kernel_initializer = init_kernel),
    # # tfpl.MultivariateNormalTriL(29)
    # tfkl.Dense(tfpl.IndependentNormal.params_size(29)),
    # tfpl.IndependentNormal(29)
    # ])




    # return decoderx(z_inp)

    with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):
        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(z_inp,
                                  units=16,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net)

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=32,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net)

        name_net = 'layer_3'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units = 29,
                                  kernel_initializer=init_kernel,
                                  name='fc')

    return net

    

def discriminator(z_inp, x_inp, is_training=False, getter=None, reuse=False):
    """ Discriminator architecture in tensorflow

    Discriminates between pairs (E(x), x) and (z, G(z))

    Args:
        z_inp (tensor): variable in the latent space
        x_inp (tensor): input data for the encoder.
        reuse (bool): sharing variables or not

    Returns:
        logits (tensor): last activation layer of the discriminator (shape 1)
        intermediate_layer (tensor): intermediate layer for feature matching

    """
    with tf.variable_scope('discriminator', reuse=reuse, custom_getter=getter):
        # D(x)
        name_x = 'x_layer_1'
        with tf.variable_scope(name_x):
            x = tf.layers.dense(x_inp,
                          units=32,
                          kernel_initializer=init_kernel,
                          name='fc')
            x = leakyReLu(x)
            x = tf.layers.dropout(x, rate=0.2, name='dropout', training=is_training)

        # D(z)
        name_z = 'z_fc_1'
        with tf.variable_scope(name_z):
            z = tf.layers.dense(z_inp, 32, kernel_initializer=init_kernel)
            z = leakyReLu(z)
            z = tf.layers.dropout(z, rate=0.2, name='dropout', training=is_training)

        # D(x,z)
        y = tf.concat([x, z], axis=1)

        name_y = 'y_fc_1'
        with tf.variable_scope(name_y):
            y = tf.layers.dense(y,
                                32,
                                kernel_initializer=init_kernel)
            y = leakyReLu(y)
            y = tf.layers.dropout(y, rate=0.2, name='dropout', training=is_training)

        intermediate_layer = y

        name_y = 'y_fc_logits'
        with tf.variable_scope(name_y):
            logits = tf.layers.dense(y,
                                     1,
                                     kernel_initializer=init_kernel)

    return logits, intermediate_layer


def leakyReLu(x, alpha=0.1, name='leaky_relu'):
    """ Leaky relu """
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))
