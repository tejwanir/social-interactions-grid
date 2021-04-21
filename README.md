# Social Interaction Grid

## Requirements
- Python 3.6

We recommend to run in a new virtual environment, e.g. `conda create -n social-grid python=3.6`. You can clone this repo and run `pip install -r requirements.txt` to get the dependencies.

## Environment

The environment is adapt from "Help or Hinder: Bayesian Models of Social Goal Inference". The reimplementation and a GUI is in the `world` folder. You can check `social_world.py` to see how to initialize the environment and show the GUI. This implementation follows OPAI gym's interface to `step` and `reset` the environment.

- Sample a new environment config: `SocialWorldEnv.sample_init_grid()`
- Step with two agents' actions: `env.step(a_strong, a_weak)`. This returns two 4-tuples `(obs, reward, done, info)`, the first one is for the strong agent, the second one is for the weak agent.
- Resest the environment: `env.reset()`

