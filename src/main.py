import numpy as np
import time
from main_init import neural_network, transition_model, transition_model_type, agent, agent_type, exp_num,count_down, \
    max_num_of_episodes, env, render, max_time_steps_episode, human_feedback, save_results, eval_save_path, \
    render_delay, save_policy, save_transition_model

# Initialize variables
total_feedback, total_time_steps, trajectories_database, total_reward = [], [], [], []
t_total, h_counter, last_t_counter, omg_c, eval_counter, total_r = 1, 0, 0, 0, 0, 0
human_done, evaluation, random_agent, evaluation_started = False, False, False, False

init_time = time.time()

# Print general information
print('\nExperiment number:', exp_num)
print('Environment:', env)
print('Learning algorithm: ', agent_type)
print('Transition Model:l', transition_model_type, '\n')

time.sleep(2)

# Count-down before training if requested
if count_down:
    for i in range(10):
        print(' ' + str(10 - i) + '...')
        time.sleep(1)

# Start training loop
for i_episode in range(max_num_of_episodes):
    print('Starting episode number', i_episode)

    if not evaluation:
        agent.new_episode()
        transition_model.new_episode()

    observation = env.reset()  # reset environment at the beginning of the episode

    past_action, past_observation, episode_trajectory, h_counter, r = None, None, [], 0, 0  # reset variables for new episode

    # Iterate over the episode
    for t in range(int(max_time_steps_episode)):
        if render:
            env.render()  # Make the environment visible
            time.sleep(render_delay)  # Add delay to rendering if necessary

        # Map action from observation
        state_representation = transition_model.get_state_representation(neural_network, observation)
        action = agent.action(neural_network, state_representation)  # TODO: probably change agent to policy

        # Act
        observation, reward, done, info = env.step(action)

        # Compute new hidden state of LSTM
        transition_model.compute_lstm_hidden_state(neural_network, action)

        # Append transition to database
        if not evaluation:
            if past_action is not None and past_observation is not None:
                episode_trajectory.append([past_observation, past_action, transition_model.processed_observation])  # append o, a, o' (not really necessary to store it like this)

            past_observation, past_action = transition_model.processed_observation, action

            if t % 100 == 0:  # TODO: add this to config file
                trajectories_database.append(episode_trajectory)  # append episode trajectory to database
                episode_trajectory = []

        # Get feedback signal
        h = human_feedback.get_h()
        evaluation = human_feedback.evaluation

        if np.any(h):
            h_counter += 1

        # Update weights policy/transition model
        if not evaluation:
            agent.train(neural_network, transition_model, state_representation, action, h, t_total)
            transition_model.train(neural_network, t_total, trajectories_database)

            t_total += 1

        # Accumulate reward (not for learning purposes, only to quantify the performance of the agents)
        r += reward

        # End of episode
        if done or human_feedback.ask_for_done():
            if evaluation:
                total_r += r

                print('Episode Reward:', '%.3f' % r)
                print('\n', i_episode, 'avg reward:', '%.3f' % (total_r / (i_episode + 1)), '\n')
                print('Percentage of given feedback:', '%.3f' % ((h_counter / (t + 1e-6)) * 100))
                total_reward.append(r)
                total_feedback.append(h_counter/(t + 1e-6))
                total_time_steps.append(t_total)
                if save_results:
                    np.save(eval_save_path + exp_num + '_reward', total_reward)
                    np.save(eval_save_path + exp_num + '_feedback', total_feedback)
                    np.save(eval_save_path + exp_num + '_time', total_time_steps)

            if save_policy:
                neural_network.save_policy()

            if save_transition_model:
                neural_network.save_transition_model()

            if render:
                time.sleep(1)

            print('Total time (s):', '%.3f' % (time.time() - init_time))
            break
