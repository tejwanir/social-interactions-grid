# The data is converted from this repo:
# https://github.com/tomeru/helpHinderShare/

from world.social_world import SocialWorldEnv

import numpy as np


grid_train = {}

# Transformations
# In the help or hinder paper
#   1-indexed grid
#   action, 4: down, 5: up, 2: left, 3: right, 1: stay
# In our social world
#   0-indexed grid
#   action, 0: donw, 1: up, 2: left, 3: right, 4: stay

# TODO: move this to a json file?
grid_train['help'] = [
  {
    # scenario 7: S comes long way to help W
    'strong': [5, 3], 'weak': [0, 0], 'FLOWER': [1, 5], 'TREE': [5, 5],
    'strong_goal': 'HELP',
    'weak_goal': 'FLOWER',
    'action_s': [0, 0, 2, 2, 2, 4, 2, 1, 1, 1, 1, 4, 4, 4, 4],
    'action_w': [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4],
    'path_s': [[5, 5, 5, 4, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [3, 2, 1, 1, 1, 1, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4]],
    'path_w': [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5]]
  },
  {
    # scenario 8: S gets out of the way to help W get to tree
    'strong': [5, 3], 'weak': [5, 1], 'FLOWER': [1, 5], 'TREE': [5, 5],
    'strong_goal': 'HELP',
    'weak_goal': 'TREE',
    'action_s': [1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    'action_w': [1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    'path_s': [[5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
               [3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]],
    'path_w': [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
               [1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]]
  },
  {
    # scenario 7, with idle boulder: S comes long way to help W
    'strong': [5, 3], 'weak': [0, 0], 'FLOWER': [1, 5], 'TREE': [5, 5], 'BOULDER': [4, 5],
    'strong_goal': 'HELP',
    'weak_goal': 'FLOWER',
    'action_s': [0, 0, 2, 2, 2, 0, 2, 1, 1, 1, 1, 4, 4, 4, 4],
    'action_w': [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4],
    'path_s': [[5, 5, 5, 4, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [3, 2, 1, 1, 1, 1, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4]],
    'path_w': [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5]]
  },
  {
    # scenario 20: Boulder-Help---trapped in by boulders on bottom level (pushing)
    'strong': [0, 0], 'weak': [3, 1], 'FLOWER': [1, 5], 'TREE': [5, 5], 'BOULDER': [2, 1],
    'strong_goal': 'HELP',
    'weak_goal': 'FLOWER',
    'action_s': [3, 3, 1, 2, 0, 4, 1, 1, 1, 1, 4, 4, 4, 4],
    'action_w': [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 4, 4, 4, 4],
    'path_s': [[0, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 1, 1, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4]],
    'path_w': [[3, 3, 3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5]]
  },
  {
    # scenario 21: Boulder-Help---trapped in by boulders on top level (no pushing)
    'strong': [0, 6], 'weak': [3, 5], 'FLOWER': [1, 5], 'TREE': [5, 5], 'BOULDER': [2, 5],
    'strong_goal': 'HELP',
    'weak_goal': 'FLOWER',
    'action_s': [3, 3, 0, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    'action_w': [3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4],
    'path_s': [[0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
               [6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]],
    'path_w': [[3, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
               [5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4]]
  },
  {
    # scenario 24: boulder tricky help
    # S is a helper and starts in top-right near W's goal.
    # S pushes boulder down and then pushes W up to goal.
    'strong': [6, 5], 'weak': [4, 2], 'FLOWER': [1, 5], 'TREE': [5, 5], 'BOULDER': [5, 3],
    'strong_goal': 'HELP',
    'weak_goal': 'TREE',
    'action_s': [2, 0, 0, 0, 0, 3, 0, 2, 1, 1, 1, 1, 4, 4, 4],
    'action_w': [0, 0, 0, 2, 2, 2, 3, 1, 1, 1, 1, 4, 4, 4, 4],
    'path_s': [[6, 5, 5, 5, 5, 5, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5],
               [5, 5, 4, 3, 2, 1, 1, 0, 0, 1, 2, 3, 4, 4, 4, 4]],
    'path_w': [[4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
               [2, 2, 2, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5]]
  },
  {
    # scenario 202: helping basic
    'strong': [5, 5], 'weak': [0, 0], 'FLOWER': [1, 5], 'TREE': [5, 5],
    'strong_goal': 'HELP',
    'weak_goal': 'FLOWER',
    'action_s': [0, 0, 2, 2, 2, 0, 2, 1, 1, 1, 1, 4, 4, 4, 4],
    'action_w': [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4],
    'path_s': [[5, 5, 5, 4, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [3, 2, 1, 1, 1, 1, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4]],
    'path_w': [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5]]
  }
]

grid_train['hinder'] = [
  {
    # scenario 6: S comes from top right to hinder W, blocking at the tunnel
    'strong': [6, 6], 'weak': [0, 0], 'FLOWER': [1, 5], 'TREE': [5, 5],
    'strong_goal': 'HINDER',
    'weak_goal': 'FLOWER',
    'action_s': [0, 2, 2, 2, 2, 2, 0, 0, 4, 4, 4, 4, 4, 4, 4],
    'action_w': [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'path_s': [[6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [6, 5, 5, 5, 5, 5, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3]],
    'path_w': [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
  },
  {
    # scenario 9: S pushes W to its goal in order to set up a hinder push 
    'strong': [6, 6], 'weak': [4, 5], 'FLOWER': [1, 5], 'TREE': [5, 5],
    'strong_goal': 'HINDER',
    'weak_goal': 'FLOWER',
    'action_s': [0, 2, 2, 2, 2, 1, 2, 0, 0, 0, 0, 0, 4, 4, 4],
    'action_w': [2, 2, 2, 2, 2, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1],
    'path_s': [[6, 6, 5, 4, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [6, 5, 5, 5, 5, 5, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1]],
    'path_w': [[4, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 2, 1, 0, 0, 0, 0]]
  },
  {
    # scenario 12
    'strong': [0, 0], 'weak': [6, 0], 'FLOWER': [1, 5], 'TREE': [5, 5],
    'strong_goal': 'HINDER',
    'weak_goal': 'TREE',
    'action_s': [1, 3, 3, 3, 3, 3, 1, 0, 2, 4, 4, 4, 4, 3, 4],
    'action_w': [1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 3, 1, 1],
    'path_s': [[0, 0, 1, 2, 3, 4, 5, 5, 5, 4, 4, 4, 4, 4, 5, 5],
               [0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]],
    'path_w': [[6, 6, 6, 5, 4, 5, 6, 5, 5, 4, 4, 4, 4, 5, 5, 5],
               [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
  },
  {
    # scenario 17: boulder gets pushed onto flower.
    'strong': [5, 4], 'weak': [1, 1], 'FLOWER': [1, 5], 'TREE': [5, 5],
    'strong_goal': 'HINDER',
    'weak_goal': 'FLOWER',
    'action_s': [1, 2, 2, 2, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    'action_w': [1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    'path_s': [[5, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
               [4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]],
    'path_w': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]
  },
  {
    # scenario 19: a locking-in hinder
    'strong': [5, 4], 'weak': [1, 0], 'FLOWER': [1, 5], 'TREE': [5, 5], 'BOULDER': [4, 5],
    'strong_goal': 'HINDER',
    'weak_goal': 'FLOWER',
    'action_s': [1, 2, 2, 2, 1, 2, 0, 0, 0, 4, 4, 4, 4, 4],
    'action_w': [1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4],
    'path_s': [[5, 5, 4, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [4, 5, 5, 5, 5, 6, 6, 5, 4, 4, 4, 4, 4, 4, 4]],
    'path_w': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
  },
  {
    # scenario 25: boulder sadistic imprisonment
    # S is a hinder and starts in top-right near W's goal.
    # S pushes boulder down and then locks W into corner
    'strong': [6, 5], 'weak': [4, 2], 'FLOWER': [1, 5], 'TREE': [5, 5], 'BOULDER': [5, 3],
    'strong_goal': 'HINDER',
    'weak_goal': 'TREE',
    'action_s': [2, 0, 0, 0, 0, 1, 2, 0, 3, 1, 3, 0, 4, 4, 4],
    'action_w': [0, 0, 0, 2, 2, 2, 3, 0, 0, 0, 2, 2, 1, 1, 1],
    'path_s': [[6, 5, 5, 5, 5, 5, 5, 4, 4, 5, 5, 6, 6, 6, 6, 6],
               [5, 5, 4, 3, 2, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1]],
    'path_w': [[4, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6],
               [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
  },
  {
    # scenario 203: hinder basic
    'strong': [6, 6], 'weak': [0, 0], 'FLOWER': [1, 5], 'TREE': [5, 5],
    'strong_goal': 'HINDER',
    'weak_goal': 'FLOWER',
    'action_s': [0, 2, 2, 2, 2, 2, 0, 0, 4, 4, 4, 4, 4, 4, 4],
    'action_w': [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'path_s': [[6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [6, 5, 5, 5, 5, 5, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3]],
    'path_w': [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
  }
]


def convert_grid(config):
    grid = np.zeros((7, 7, len(SocialWorldEnv.Items)))
    # add walls
    walls = [[3,0], [0,3], [3,3], [3,6], [6,3], [2,3], [3,4], [3,2], [4,3]]
    for x,y in walls:
        grid[x,y,SocialWorldEnv.Items.WALL] = 1
    # add other items and agents from the config
    strong_pos = None; weak_pos = None
    strong_goal = None; weak_goal = None
    for key, val in config.items():
        if 'action' in key or 'goal' in key or 'path' in key:
            continue
        if key == 'strong':
            strong_pos = val
            strong_goal = SocialWorldEnv.Goals[config['strong_goal']]
        elif key == 'weak':
            weak_pos = val
            weak_goal = SocialWorldEnv.Goals[config['weak_goal']]
        else:
            x, y = val
            grid[x][y][SocialWorldEnv.Items[key]] = 1
    return grid, strong_pos, strong_goal, weak_pos, weak_goal


if __name__ == '__main__':
    mode = 'help'; config_idx = 0
    config = grid_train[mode][config_idx]
    grid, strong_pos, strong_goal, weak_pos, weak_goal = convert_grid(config)
    env = SocialWorldEnv(grid=grid, agent_strong=strong_pos,
            agent_strong_goal=strong_goal, agent_weak=weak_pos, agent_weak_goal=weak_goal)
    for i in range(len(config['action_s'])):
        path_s = config['path_s']
        path_w = config['path_w']
        # needs to use position control because the weak agent may fail to perform an action
        agent_s, agent_w = env.step(config['action_s'][i],
                                    config['action_w'][i],
                                    pos_ctrl=True,
                                    pos_s=[path_s[0][i], path_s[1][i]],
                                    pos_w=[path_w[0][i], path_w[1][i]])
        env.gui.draw()
    while True:  # make sure the GUI still shows up
        env.gui.draw(move=True)
