# Interactive Learning of Temporal Features for Control
Code of the paper "Interactive Learning of Temporal Features for Control" published in the IEEE Robotics & Automation Magazine (Special Issue on Deep Learning and Machine Learning in Robotics).

This code is based on the following publication:
1. [Interactive Learning of Temporal Features for Control](https://ieeexplore.ieee.org/document/9076630), preprint availabe [here](http://www.jenskober.de/publications/PerezDattari2020RAM.pdf).

**Authors:** Rodrigo Pérez-Dattari, Carlos Celemin, Giovanni Franzese, Javier Ruiz-del-Solar, Jens Kober.

[Link to paper video](https://youtu.be/4kWGfNdm21A)

This repository includes the code necessary to run the experiments done in simulated environments using human teachers.
## Installation

To use the code, it is necessary to first install the gym toolkit (release v0.9.6): https://github.com/openai/gym

Then, the files in the `gym` folder of this repository should be replaced/added in the installed gym folder in your PC. Two environments were added:

1. **Continuous Mountain Car:** the environment outputs an image as an observation.

1. **Inverted Pendulum:** the pendulum is bigger and the environments outputs an image as an observation.

### Requirements
* setuptools==38.5.1
* numpy==1.13.3
* opencv_python==3.4.0.12
* matplotlib==2.2.2
* tensorflow==1.4.0
* pyglet==1.3.2
* gym==0.9.6

## Usage

1. To run the main program type in the terminal (inside the folder `src`):

```bash 
python main.py --config-file <environment>
```

To be able to give feedback to the agent, the environment rendering window must be selected/clicked.

## Comments

This code has been tested in `Ubuntu 18.04` and `python >= 3.5`.

## Troubleshooting

If you run into problems of any kind, don't hesitate to [open an issue](https://github.com/rperezdattari/Interactive-Learning-of-Temporal-Features-for-Control/issues) on this repository. It is quite possible that you have run into some bug we are not aware of.

