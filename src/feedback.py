from pyglet.window import key
from tools.functions import str_2_array

"""
Class that obtains the human feedback from the computer's keyboard.
"""


class Feedback:
    def __init__(self, env, key_type, h_up, h_down, h_right, h_left, h_null):
        if key_type == '1':
            env.unwrapped.viewer.window.on_key_press = self.key_press
            env.unwrapped.viewer.window.on_key_release = self.key_release
        elif key_type == '2':
            env.unwrapped.window.on_key_press = self.key_press
            env.unwrapped.window.on_key_release = self.key_release
        else:
            print('No valid feedback type selected!')
            exit()

        self.h = str_2_array(h_null)  # Human correction
        self.h_null = str_2_array(h_null)
        self.h_up = str_2_array(h_up)
        self.h_down = str_2_array(h_down)
        self.h_right = str_2_array(h_right)
        self.h_left = str_2_array(h_left)
        self.restart = False
        self.evaluation = False
        self.model_training = False

    def key_press(self, k, mod):
        if k == key.A:
            self.h = self.h_right
        if k == key.D:
            self.h = self.h_left
        if k == key.SPACE:
            self.restart = True
        if k == key.LEFT:
            self.h = self.h_left
        if k == key.RIGHT:
            self.h = self.h_right
        if k == key.NUM_1:
            self.h = self.h_left
        if k == key.NUM_3:
            self.h = self.h_right
        if k == key.NUM_6:
            self.h = self.h_left
        if k == key.NUM_4:
            self.h = self.h_right
        if k == key.UP:
            self.h = self.h_up
        if k == key.DOWN:
            self.h = self.h_down
        if k == key.E:
            self.evaluation = not self.evaluation
            if self.evaluation:
                print('EVALUATION STARTED')
            else:
                print('EVALUATION STOPPED')
        if k == key.S:
            self.model_training = not self.model_training
            if not self.model_training:
                print('MODEL TRAINING STOPPED')
            else:
                print('MODEL TRAINING STARTED')

    def key_release(self, k, mod):
        if k == key.LEFT or k == key.RIGHT or k == key.UP or k == key.DOWN \
                or k == key.A or k == key.D or k == key.NUM_1 or k == key.NUM_3 or k == key.NUM_4 or k == key.NUM_6:
            self.h = self.h_null

    def get_h(self):
        return self.h

    def ask_for_done(self):
        done = self.restart
        self.restart = False
        return done
