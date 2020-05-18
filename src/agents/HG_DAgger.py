import numpy as np
from tools.functions import str_2_array
from buffer import Buffer

"""
Implementation of HG-DAgger without uncertainty estimation
"""


class HG_DAGGER:
    def __init__(self, dim_a, action_upper_limits, action_lower_limits, buffer_min_size, buffer_max_size,
                 buffer_sampling_rate, buffer_sampling_size, number_training_iterations, train_end_episode):
        # Initialize variables
        self.dim_a = dim_a
        self.action_upper_limits = str_2_array(action_upper_limits, type_n='float')
        self.action_lower_limits = str_2_array(action_lower_limits, type_n='float')
        self.count = 0
        self.buffer_sampling_rate = buffer_sampling_rate
        self.buffer_sampling_size = buffer_sampling_size
        self.number_training_iterations = number_training_iterations
        self.train_end_episode = train_end_episode

        # Initialize HG_DAgger buffer
        self.buffer = Buffer(min_size=buffer_min_size, max_size=buffer_max_size)

    def feed_h(self, h):
        self.h = np.reshape(h, [1, self.dim_a])

    def action(self, neural_network, state_representation):
        self.count += 1

        if np.any(self.h):  # if feedback, human teleoperates
            action = self.h
            print("feedback:", self.h[0])
        else:
            action = neural_network.sess.run(neural_network.policy_output,
                                             feed_dict={'policy/state_representation:0': state_representation})

        out_action = []

        for i in range(self.dim_a):
            action[0, i] = np.clip(action[0, i], -1, 1) * self.action_upper_limits[i]
            out_action.append(action[0, i])

        return np.array(out_action)

    def train(self, neural_network, transition_model, action, t, done):
        # Add last step to memory buffer
        if transition_model.last_step(action) is not None and np.any(self.h):  # if human teleoperates, add action to database
            self.buffer.add(transition_model.last_step(action))

        # Train policy every k time steps from buffer
        if self.buffer.initialized() and (t % self.buffer_sampling_rate == 0 or (self.train_end_episode and done)):
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 20) == 0:
                    print('Progress Policy training: %i %%' % (i / self.number_training_iterations * 100))

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                observation_sequence_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
                action_sequence_batch = [np.array(pair[1]) for pair in batch]
                current_observation_batch = [np.array(pair[2]) for pair in batch]  # last
                action_label_batch = [np.array(pair[3]) for pair in batch]

                state_representation_batch = transition_model.get_state_representation_batch(neural_network,
                                                                                             observation_sequence_batch,
                                                                                             action_sequence_batch,
                                                                                             current_observation_batch)

                neural_network.sess.run(neural_network.train_policy,
                                        feed_dict={'policy/state_representation:0': state_representation_batch,
                                                   'policy/policy_label:0': action_label_batch})
