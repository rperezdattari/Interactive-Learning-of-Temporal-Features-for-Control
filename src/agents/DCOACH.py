import numpy as np
from tools.functions import str_2_array
from buffer import Buffer


class DCOACH:
    def __init__(self, dim_a, action_upper_limits, action_lower_limits, e, buffer_min_size, buffer_max_size,
                 buffer_sampling_rate, buffer_sampling_size):
        # Initialize variables
        self.policy_action_label = None
        self.e = np.array(str_2_array(e, type_n='float'))
        self.dim_a = dim_a
        self.action_upper_limits = str_2_array(action_upper_limits, type_n='float')
        self.action_lower_limits = str_2_array(action_lower_limits, type_n='float')
        self.count = 0
        self.buffer_sampling_rate = buffer_sampling_rate
        self.buffer_sampling_size = buffer_sampling_size

        # Initialize DCOACH buffer
        self.buffer = Buffer(min_size=buffer_min_size, max_size=buffer_max_size)

    def _generate_policy_label(self, h, action):
        if np.any(h):
            error = np.array(h * self.e).reshape(1, self.dim_a)
            self.policy_action_label = []

            for i in range(self.dim_a):
                self.policy_action_label.append(np.clip(action[i] / self.action_upper_limits[i] + error[0, i], -1, 1))

            self.policy_action_label = np.array(self.policy_action_label).reshape(1, self.dim_a)
        else:
            self.policy_action_label = np.reshape(action, [1, self.dim_a])

    def _single_update(self, neural_network, state_representation):
        neural_network.sess.run(neural_network.train_policy, feed_dict={'policy/state_representation:0': state_representation,
                                                                        'policy/policy_label:0': self.policy_action_label})

    def _batch_update(self, neural_network, transition_model, batch):
        observation_sequence_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        action_sequence_batch = [np.array(pair[1]) for pair in batch]
        current_observation_batch = [np.array(pair[2]) for pair in batch]  # last
        action_label_batch = [np.array(pair[3]) for pair in batch]

        state_representation_batch = transition_model.get_state_representation_batch(neural_network, observation_sequence_batch, action_sequence_batch, current_observation_batch)

        neural_network.sess.run(neural_network.train_policy,
                                feed_dict={'policy/state_representation:0': state_representation_batch,
                                           'policy/policy_label:0': action_label_batch})

    def action(self, neural_network, state_representation):
        self.count += 1

        action = neural_network.sess.run(neural_network.policy_output,
                                         feed_dict={'policy/state_representation:0': state_representation})
        out_action = []

        for i in range(self.dim_a):
            action[0, i] = np.clip(action[0, i], -1, 1) * self.action_upper_limits[i]
            out_action.append(action[0, i])

        return np.array(out_action)

    def train(self, neural_network, transition_model, state_representation, action, h, t):
        self._generate_policy_label(h, action)

        # Policy training
        if np.any(h):  # if any element is not 0
            self._single_update(neural_network, state_representation)
            print("feedback", h)

            # Add state action-label pair to memory buffer
            if transition_model.last_step(self.policy_action_label) is not None:
                self.buffer.add(transition_model.last_step(self.policy_action_label))

            # Train sampling from buffer
            if self.buffer.initialized():
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)  # TODO: probably this config thing should not be here
                self._batch_update(neural_network, transition_model, batch)

        # Train policy every k time steps from buffer
        if self.buffer.initialized() and t % self.buffer_sampling_rate == 0:
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            self._batch_update(neural_network, transition_model, batch)

    def new_episode(self):
        pass
