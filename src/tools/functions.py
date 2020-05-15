import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')  # Necessary for fast plots, otherwise they may crash
import configparser


def load_config_data(config_dir):
    config = configparser.ConfigParser()
    config.read(config_dir)
    return config


def str_2_array(str_state_shape, type_n='int'):
    sep_str_state_shape = str_state_shape.split(',')
    state_n_dim = len(sep_str_state_shape)
    state_shape = []
    for i in range(state_n_dim):
        if type_n == 'int':
            state_shape.append(int(sep_str_state_shape[i]))
        elif type_n == 'float':
            state_shape.append(float(sep_str_state_shape[i]))
        else:
            print('Selected type for str_2_array not implemented.')
            exit()

    return state_shape


def observation_to_gray(observation, image_size):
    observation = np.array(observation).reshape(1, image_size, image_size, 3)
    observation_gray = np.mean(observation, axis=3)
    observation_gray = observation_gray.reshape(
        (-1, image_size, image_size, 1))
    observation_gray_norm = observation_gray / 255.0

    return observation_gray_norm


class FastImagePlot:
    def __init__(self, fig_num, observation, image_size, title_name, vmin=0, vmax=1):
        self.window = plt.figure(fig_num)
        self.image_size = image_size
        self.im = plt.imshow(np.reshape(observation, [self.image_size, self.image_size]),
                             cmap='gray', vmin=vmin, vmax=vmax)
        plt.show(block=False)
        self.window.canvas.set_window_title(title_name)
        self.window.canvas.draw()

    def refresh(self, observation):
        self.im.set_data(np.reshape(observation, [self.image_size, self.image_size]))
        self.window.draw_artist(self.im)
        self.window.canvas.blit()
        self.window.canvas.flush_events()


class Fast1DPlot:
    def __init__(self, plot_height, number_of_plots, titles):
        self.fig = plt.figure()
        self.axes = []
        self.feedback_plots = []
        self.return_plots = []
        self.axesBackground = []
        self.number_of_plots = number_of_plots
        self.plot_height = plot_height

        for i in range(number_of_plots):
            self.axes.append(self.fig.add_subplot(number_of_plots, 1, i+1))
            self.axes[i].set_ylim(0, 1)
            self.axes[i].set_xlim(-plot_height, 0)
            self.axes[i].set_title(titles[i])

        self.fig.canvas.draw()
        x_start = range(1-plot_height, 1)
        y_start = np.zeros(plot_height)

        for i in range(number_of_plots):
            self.feedback_plots.append(self.axes[i].plot(x_start, y_start))
            self.return_plots.append(self.axes[i].plot(x_start, y_start))
            self.axesBackground.append(self.fig.canvas.copy_from_bbox(self.axes[i].bbox))

    def refresh(self, feedback_plots, return_plots):
        for i in range(self.number_of_plots):
            zeros = np.zeros(self.plot_height)
            y_feedback = np.concatenate([zeros, feedback_plots[i]])
            y_return = np.concatenate([zeros, return_plots[i]])

            self.feedback_plots[i][0].set_ydata(y_feedback[-self.plot_height:])
            self.return_plots[i][0].set_ydata(y_return[-self.plot_height:])

            self.fig.canvas.restore_region(self.axesBackground[i])

            # redraw just the points
            self.axes[i].draw_artist(self.feedback_plots[i][0])
            self.axes[i].draw_artist(self.return_plots[i][0])

            # fill in the axes rectangle
            self.fig.canvas.blit(self.axes[i].bbox)
        plt.pause(1e-7)
