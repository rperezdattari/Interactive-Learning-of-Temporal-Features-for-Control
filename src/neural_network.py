import tensorflow as tf
import tensorflow.contrib.layers as lays
import os


class NeuralNetwork:
    def __init__(self, policy_learning_rate, transition_model_learning_rate, lstm_hidden_state_size,
                 load_transition_model, load_policy, dim_a, network_loc, image_size):

        self.lstm_hidden_state_size = lstm_hidden_state_size
        self.policy_learning_rate = policy_learning_rate
        self.image_width = image_size  # we assume that the image is a square
        self.dim_a = dim_a
        self.network_loc = network_loc
        self.transition_model_learning_rate = transition_model_learning_rate

        # Build and load network if requested
        self._build_network()

        if load_policy:
            self._load_policy()

        if load_transition_model:
            self._load_transition_model()

    def _build_network(self):  # check this
        with tf.variable_scope('transition_model'):
            # Create placeholders
            transition_model_input = tf.placeholder(tf.float32, (None, self.image_width, self.image_width, 1), name='transition_model_input')
            transition_model_label = tf.placeholder(tf.float32, (None, self.image_width, self.image_width, 1), name='transition_model_label')
            action_in = tf.placeholder(tf.float32, (None, self.dim_a), name='action_in')
            batch_size = tf.placeholder(tf.int32, [], name='batch_size')
            sequence_length = tf.placeholder(tf.int32, name='sequence_length')
            autoencoder_mode = tf.placeholder(tf.bool, shape=(), name='autoencoder_mode')  # True -> recurrency, False -> no recurrency

            # Convolutional encoder
            conv1 = tf.contrib.layers.layer_norm(lays.conv2d(transition_model_input, 16, [3, 3], stride=2, padding='SAME'))
            conv2 = tf.contrib.layers.layer_norm(lays.conv2d(conv1, 8, [3, 3], stride=2, padding='SAME'))
            conv3 = lays.conv2d(conv2, 4, [3, 3], stride=2, padding='SAME', activation_fn=tf.nn.sigmoid)
            conv3_shape = conv3.get_shape()

            # Autoencoder latent space (without recurrency)
            latent_space = tf.contrib.layers.flatten(conv3)
            latent_space_shape = latent_space.get_shape()

            # Combine latent space information with actions from the policy
            fc_1 = tf.layers.dense(action_in, latent_space_shape[1], activation=tf.nn.tanh)
            fc_2 = tf.layers.dense(latent_space, latent_space_shape[1], activation=tf.nn.tanh)
            concat_1 = tf.concat([fc_1, fc_2], axis=1)  # concatenate latent space and action transformations
            concat_1_shape = concat_1.get_shape()

            # Transform data into 3-D sequential structures: [batch size, sequence length, data size]
            sequential_concat_1 = tf.reshape(concat_1, shape=[batch_size, sequence_length, concat_1_shape[-1]])
            sequential_latent_space = tf.reshape(latent_space, shape=[batch_size, sequence_length, latent_space_shape[-1]])

            # LSTM
            cell = tf.nn.rnn_cell.LSTMCell(self.lstm_hidden_state_size, state_is_tuple=False)
            lstm_hidden_state_in = cell.zero_state(batch_size, tf.float32)
            lstm_hidden_state_in = tf.identity(lstm_hidden_state_in, name='lstm_hidden_state_in')

            _, lstm_hidden_state_out = tf.nn.dynamic_rnn(cell, sequential_concat_1, initial_state=lstm_hidden_state_in, dtype=tf.float32)

            self.lstm_hidden_state = tf.identity(lstm_hidden_state_out, name='lstm_hidden_state_out')

            concat_2 = tf.concat([self.lstm_hidden_state[:, -self.lstm_hidden_state_size:], sequential_latent_space[:, -1, :]], axis=1)

            # State representation
            self.state_representation = tf.layers.dense(concat_2, 1000, activation=tf.nn.tanh)  # fc_3 TODO: cells num in config file

            fc_4 = tf.reshape(tf.layers.dense(self.state_representation, latent_space_shape[1], activation=tf.nn.tanh), [-1, latent_space_shape[1]])
            fc_4 = tf.reshape(fc_4, [-1, conv3_shape[1], conv3_shape[2], conv3_shape[3]])  # go to shape of the latent space

            # Convolutional decoder
            dec_input = tf.cond(autoencoder_mode, lambda: fc_4, lambda: conv3)

            deconv1 = tf.contrib.layers.layer_norm(lays.conv2d_transpose(dec_input, 8, [3, 3], stride=2, padding='SAME'))
            deconv2 = tf.contrib.layers.layer_norm(lays.conv2d_transpose(deconv1, 16, [3, 3], stride=2, padding='SAME'))
            deconv3 = lays.conv2d_transpose(deconv2, 1, [3, 3], stride=2, padding='SAME', activation_fn=tf.nn.sigmoid)

            self.transition_model_output = tf.identity(deconv3, name='transition_model_output')

            # Prediction reconstruction loss
            reconstruction_loss = tf.reduce_mean(tf.square(self.transition_model_output - transition_model_label))

        with tf.variable_scope('policy'):
            # Placeholders
            policy_output_label = tf.placeholder(tf.float32, [None, self.dim_a], name='policy_label')

            # Inputs
            state_representation = tf.identity(self.state_representation, name='state_representation')
            self.policy_input = tf.contrib.layers.layer_norm(state_representation)

            # Fully connected layers
            fc_5 = tf.layers.dense(self.policy_input, 1000, activation=tf.nn.relu)
            fc_6 = tf.layers.dense(fc_5, 1000, activation=tf.nn.relu)
            self.policy_output = tf.layers.dense(fc_6, self.dim_a, activation=tf.nn.tanh, name='action')  # fc_7

            # Policy loss
            policy_loss = 0.5 * tf.reduce_mean(tf.square(self.policy_output - policy_output_label))

        # Policy optimizer
        variables_policy = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy')
        self.train_policy = tf.train.GradientDescentOptimizer(
            learning_rate=self.policy_learning_rate).minimize(policy_loss, var_list=variables_policy)

        # Autoencoder/Transition Model optimizer
        variables_transition_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'transition_model')
        self.train_transition_model = tf.train.AdamOptimizer(learning_rate=self.transition_model_learning_rate).minimize(reconstruction_loss, var_list=variables_transition_model)

        # Initialize tensorflow
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=100)
        self.saver_transition_model = tf.train.Saver(var_list=variables_transition_model)
        self.saver_policy = tf.train.Saver(var_list=variables_policy)

    def save_transition_model(self):
        if not os.path.exists(self.network_loc):
            os.makedirs(self.network_loc)

        self.saver_transition_model.save(self.sess, self.network_loc + '_transition_model')

    def _load_transition_model(self):
        self.saver_transition_model.restore(self.sess, self.network_loc + '_transition_model')

    def save_policy(self):
        if not os.path.exists(self.network_loc):
            os.makedirs(self.network_loc)

        self.saver_transition_model.save(self.sess, self.network_loc + '_policy')

    def _load_policy(self):
        self.saver_transition_model.restore(self.sess, self.network_loc + '_transition_policy')
