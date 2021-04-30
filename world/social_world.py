from enum import IntEnum
from gym import spaces
from PIL import Image

import gym
import numpy as np
import os
import pygame


class SocialGui(object):
    def __init__(self, env, width=7, height=7,
                 is_headless=False, width_px=300, height_px=300,
                 img_path='world/images/',
                 caption='SocialWorld Simulator'):
        if is_headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        self._env = env
        self._width = width
        self._height = height
        self._cell_width = width_px / width
        self._cell_height = height_px / height
        # load icon images
        self._sprites = {}
        for item in SocialWorldEnv.Items:
            filename = img_path + item.name.lower() + '.bmp'
            self._sprites[item.value] = pygame.image.load(filename)
        self._sprites['strong'] = pygame.image.load(img_path + 'strong.bmp')
        self._sprites['weak'] = pygame.image.load(img_path + 'weak.bmp')
        # pygame related init
        self._screen = pygame.display.set_mode((width_px, height_px), 0, 32)
        pygame.display.set_caption(caption)
        self._clock = pygame.time.Clock()

    def move(self, action_strong, action_weak, reset_after_done=False):
        agent_strong, agent_weak = self._env.step(action_strong, action_weak)
        print(' strong', agent_strong)
        print(' weak', agent_weak)
        if reset_after_done and (agent_strong[2] or agent_weak[2]):
            self._env.reset()

    def draw(self, move=False, reset_after_done=False):
        bg_color = (255, 255, 255)
        self._screen.fill(bg_color)
        row = 0
        cell_size = (int(self._cell_width-1), int(self._cell_height-1))
        for y in reversed(range(self._height)):
            for x in range(self._width):
                px_x = x*self._cell_width
                px_y = row*self._cell_height
                if self._env.grid[x, y, :].any() \
                        or [x, y] == self._env.strong_pos \
                        or [x, y] == self._env.weak_pos:
                    if [x, y] == self._env.strong_pos:
                        thing = 'strong'
                    elif [x, y] == self._env.weak_pos:
                        thing = 'weak'
                    else:
                        thing = self._env.grid[x, y, :].argmax()
                    picture = pygame.transform.scale(self._sprites[thing], cell_size)
                    self._screen.blit(picture, (px_x, px_y))
            row += 1
        pygame.display.update()
        # input action to move the agent
        if move:
            print('Input the actions for the strong agent, 0:Down, 1:Up, 2:Left, 3:Right, 4:Stay')
            action_strong = int(input())
            print('Input the actions for the weak agent, 0:Down, 1:Up, 2:Left, 3:Right, 4:Stay')
            action_weak = int(input())
            self.move(action_strong, action_weak, reset_after_done)


class SocialWorldEnv(gym.Env):
    class Actions(IntEnum):
        DOWN  = 0
        UP    = 1
        LEFT  = 2
        RIGHT = 3
        STAY  = 4

    class Items(IntEnum):
        TREE    = 0
        FLOWER  = 1
        BOULDER = 2
        WALL    = 3

    class Goals(IntEnum):
        TREE   = 0
        FLOWER = 1
        HELP   = 2
        HINDER = 3

    def __init__(self, grid,
                 agent_strong, agent_strong_goal,
                 agent_weak, agent_weak_goal, weak_success_p=0.6,
                 strong_rho_g=1, strong_delta_g=0.5, strong_rho_o=5,
                 weak_rho_g=1, weak_delta_g=0.5,
                 use_gui=True, is_headless=False):
        self.init_grid = grid.copy()
        self.init_strong_pos = agent_strong.copy()
        self.init_weak_pos = agent_weak.copy()
        # agent/grid configurations
        # TODO: refactor this into two dictionaries for two agents
        self.grid = grid
        self.strong_pos = agent_strong
        self.strong_goal = agent_strong_goal
        self.strong_goal_pos = self._get_goal_pos(self.strong_goal)
        self.strong_rho_g = strong_rho_g
        self.strong_delta_g = strong_delta_g
        self.strong_rho_o = strong_rho_o
        self.weak_pos = agent_weak
        self.weak_goal = agent_weak_goal
        self.weak_goal_pos = self._get_goal_pos(self.weak_goal)
        self.weak_prob = weak_success_p
        self.weak_rho_g = weak_rho_g
        self.weak_delta_g = weak_delta_g
        self.grid_size = grid.shape[0]
        # set up the action space
        self.actions = SocialWorldEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        self.del2dir = {(-1,0): int(SocialWorldEnv.Actions.LEFT),
                        (1,0): int(SocialWorldEnv.Actions.RIGHT),
                        (0,-1): int(SocialWorldEnv.Actions.DOWN),
                        (0,1): int(SocialWorldEnv.Actions.UP),
                        (0,0): int(SocialWorldEnv.Actions.STAY)}
        self.dir2del = {x:y for y,x in self.del2dir.items()}
        # set up the observation space
        self.obj_pos = self._get_obj_pos()
        # set up the gui
        self.use_gui = use_gui
        self.is_headless = is_headless
        if use_gui:
            self.gui = SocialGui(self, is_headless=self.is_headless)

    @staticmethod
    def sample_init_grid(width=7, height=7):
        grid = np.zeros((width, height, len(SocialWorldEnv.Items)))
        occupied = []
        # add walls
        walls = [[3,0], [0,3], [3,3], [3,6], [6,3], [2,3], [3,4], [3,2], [4,3]]
        for x,y in walls:
            grid[x,y,SocialWorldEnv.Items.WALL] = 1
        occupied.extend(walls)
        # sample agent pos
        tunnels = [[1,3], [3,1], [3,5], [5,3]]
        rooms_range = [((0,2),(0,2)), ((0,2),(4,6)), ((4,6),(0,2)), ((4,6),(4,6))]
        def sample_room_pos(room_id=None, is_boulder=False):
            if room_id is None:
                p = [0.2 for _ in range(5)] if is_boulder \
                        else [0.23,0.23,0.23,0.23,0.08]
                room_id = np.random.choice([0,1,2,3,4], p=p)
            pos = None
            while pos is None or pos in occupied:
                if room_id < 4:
                    range_x, range_y = rooms_range[room_id]
                    x = np.random.randint(range_x[0], range_x[1]+1)
                    y = np.random.randint(range_y[0], range_y[1]+1)
                    pos = [x,y]
                else:
                    tunnel_id = np.random.randint(0,4)
                    pos = tunnels[tunnel_id]
            return pos
        strong_pos = sample_room_pos()
        weak_pos = sample_room_pos()
        # sample item configurations
        rooms = [0,1,2,3]
        np.random.shuffle(rooms)
        item_rooms = rooms[:2]
        for i, room in enumerate(item_rooms):
            x,y = sample_room_pos(room)
            grid[x][y][i] = 1
        # has boulder or not
        if np.random.choice([True, False]):
            x,y = sample_room_pos()
            grid[x][y][SocialWorldEnv.Items.BOULDER.value] = 1
        # sample agent's goal
        strong_goal = np.random.randint(4)
        weak_goal = np.random.randint(2)
        return grid, strong_pos, strong_goal, weak_pos, weak_goal

    def reset(self):
        self.grid = self.init_grid.copy()
        self.strong_pos = self.init_strong_pos.copy()
        self.weak_pos = self.init_weak_pos.copy()

    def _get_goal_pos(self, goal):
        if goal > 1:
            return None  # no goal position for help/hinder
        grid_shape = self.grid.shape
        for x in range(grid_shape[0]):
            for y in range(grid_shape[1]):
                if self.grid[x, y, :].any():
                    thing = self.grid[x, y, :].argmax()
                    if thing == goal:
                        return [x,y]
        return None

    def _get_obj_pos(self):
        obj_pos = {}
        grid_shape = self.grid.shape
        for x in range(grid_shape[0]):
            for y in range(grid_shape[1]):
                if self.grid[x, y, :].any():
                    thing = self.grid[x, y, :].argmax()
                    if thing >= 2:
                        continue
                    obj_pos[SocialWorldEnv.Items(thing)] = (x, y)
        return obj_pos

    def _move_boulder(self, delta):
        x, y = self.strong_pos
        boulder = SocialWorldEnv.Items.BOULDER.value
        if self.grid[x,y,boulder] == 1:
            self.grid[x,y,boulder] = 0
            x += delta[0]
            y += delta[1]
            self.grid[x,y,boulder] = 1

    def _move_weak_agent(self, delta):
        if self.strong_pos == self.weak_pos:
            self.weak_pos[0] += delta[0]
            self.weak_pos[1] += delta[1]

    def _collision_free(self, x, y, is_weak=False):
        out_boundary = x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size
        if out_boundary:  # handling the boundary conditions
            return False
        if is_weak:  # the weak agent cannot push the boulder or the strong agent
            if self.grid[x][y][SocialWorldEnv.Items.BOULDER]:
                return False
            if self.strong_pos[0] == x and self.strong_pos[1] == y:
                return False
        return self.grid[x][y][SocialWorldEnv.Items.WALL] == 0

    def _get_obs(self, agent_pos):
        features = []
        for item in range(2):
            obj_pos = self.obj_pos[SocialWorldEnv.Items(item)]
            features.append(obj_pos[0] - agent_pos[0])
            features.append(obj_pos[1] - agent_pos[1])
        return np.array(features)

    def _action_cost(self, action):
        if SocialWorldEnv.Actions(action) == SocialWorldEnv.Actions.STAY:
            return 0.1
        return 1

    def _object_reward(self, agent_pos, goal_pos, rho, delta):
        agent_pos = np.array(agent_pos)
        goal_pos = np.array(goal_pos)
        dist = np.linalg.norm(agent_pos-goal_pos)
        return max(rho * (1.-dist/delta), 0)

    def state2pos(self, state_idx):
        grid_size = self.grid_size**2
        strong_state_idx = state_idx % grid_size
        weak_state_idx = state_idx / grid_size
        xs = strong_state_idx % self.grid_size
        ys = strong_state_idx / self.grid_size
        xw = weak_state_idx % self.grid_size
        yw = weak_state_idx / self.grid_size
        return [xs, ys], [xw, yw]

    def pos2state(self, strong_pos, weak_pos):
        xs = strong_pos[0]; ys = strong_pos[1]
        xw = weak_pos[0];   yw = weak_pos[1]
        state_idx = (ys*self.grid_size + xs) + (self.grid_size**2) * (yw*self.grid_size + xw)
        return int(state_idx)

    def next_pos(self, pos, intend_action, is_weak):
        intend_pos = None; other_pos = []
        for action, delta in self.dir2del.items():
            new_pos = pos.copy()
            new_pos[0] += delta[0]; new_pos[0] = int(new_pos[0])
            new_pos[1] += delta[1]; new_pos[1] = int(new_pos[1])
            if not self._collision_free(new_pos[0], new_pos[1], is_weak=is_weak):
                new_pos = pos.copy()
                new_pos[0] = int(new_pos[0])
                new_pos[1] = int(new_pos[1])
            if action == intend_action:
                intend_pos = new_pos
            else:
                other_pos.append(new_pos)
        return intend_pos, other_pos

    def get_trans_matrix(self, agent_type):
        # returns a transition matrix: S x Ai x Aj x S'
        n_actions = len(self.dir2del.keys())
        trans_prob = np.zeros(((self.grid_size**2)**2, n_actions, n_actions, (self.grid_size**2)**2))
        for i in range((self.grid_size**2)**2):
            strong_pos, weak_pos = self.state2pos(i)
            for action_s in self.dir2del.keys():
                new_strong_pos, _ = self.next_pos(strong_pos, action_s, False)
                for action_w in self.dir2del.keys():
                    new_weak_pos, other_weak_pos = self.next_pos(weak_pos, action_w, True)
                    # if the weak succeeds
                    new_state_idx = self.pos2state(new_strong_pos, new_weak_pos)
                    trans_prob[i][action_s][action_w][new_state_idx] += self.weak_prob
                    # if the weak fails
                    for new_weak_pos in other_weak_pos:
                        new_state_idx = self.pos2state(new_strong_pos, new_weak_pos)
                        trans_prob[i][action_s][action_w][new_state_idx] += \
                            (1.-self.weak_prob) / (len(self.dir2del.keys())-1)
        if agent_type == 'strong':
            return trans_prob
        else:  # swap axis if looking up from the weak agent's view
            return np.swapaxes(trans_prob, 1, 2)

    def get_observe_matrix(self, agent_type):
        # returns a observation matrix of the other agent: S x Ai x S
        n_actions = len(self.dir2del.keys())
        weak_fail_prob = (1.-self.weak_prob) / (len(self.dir2del.keys())-1)
        observe_prob = np.zeros(((self.grid_size**2)**2, n_actions, (self.grid_size**2)**2))
        for i in range((self.grid_size**2)**2):
            strong_pos, weak_pos = self.state2pos(i)
            if agent_type == 'strong':
                for action_s in self.dir2del.keys():
                    sum_prob = 0
                    new_strong_pos, _ = self.next_pos(strong_pos, action_s, False)
                    for action_w in self.dir2del.keys():
                        new_weak_pos, other_weak_pos = self.next_pos(weak_pos, action_w, True)
                        # if the weak succeeds
                        new_state_idx = self.pos2state(new_strong_pos, new_weak_pos)
                        observe_prob[i][action_s][new_state_idx] += self.weak_prob
                        sum_prob += self.weak_prob
                        # if the weak fails
                        for new_weak_pos in other_weak_pos:
                            new_state_idx = self.pos2state(new_strong_pos, new_weak_pos)
                            observe_prob[i][action_s][new_state_idx] += weak_fail_prob
                            sum_prob += weak_fail_prob
                    # normalize the probability
                    observe_prob[i][action_s] /= sum_prob
            else:
                for action_w in self.dir2del.keys():
                    sum_prob = 0
                    new_weak_pos, other_weak_pos = self.next_pos(weak_pos, action_w, True)
                    # if the weak succeeds
                    for action_s in self.dir2del.keys():
                        new_strong_pos, _ = self.next_pos(strong_pos, action_s, False)
                        new_state_idx = self.pos2state(new_strong_pos, new_weak_pos)
                        observe_prob[i][action_w][new_state_idx] += self.weak_prob
                    # if the weak fails
                    for new_weak_pos in other_weak_pos:
                        for action_s in self.dir2del.keys():
                            new_strong_pos, _ = self.next_pos(strong_pos, action_s, False)
                            new_state_idx = self.pos2state(new_strong_pos, new_weak_pos)
                            observe_prob[i][action_w][new_state_idx] += weak_fail_prob
                    # normalize the probability
                    observe_prob[i][action_w] /= sum_prob
        return observe_prob

    def get_reward_matrix(self, agent_type):
        # returns a reward matrix S x Ai x Aj x 1
        n_actions = len(self.dir2del.keys())
        reward = np.zeros(((self.grid_size**2)**2, n_actions, n_actions, 1))
        for i in range((self.grid_size**2)**2):
            strong_pos, weak_pos = self.state2pos(i)
            for action_s in self.dir2del.keys():
                new_strong_pos, _ = self.next_pos(strong_pos, action_s, False)
                for action_w in self.dir2del.keys():
                    new_weak_pos, _ = self.next_pos(weak_pos, action_w, True)
                    if agent_type == 'strong':
                        reward[i][action_s][action_w][0] = self.compute_reward(agent_type,
                            new_strong_pos, self.strong_goal_pos, action_s,
                            new_weak_pos, self.weak_goal_pos)
                    else:
                        reward[i][action_w][action_s][0] = self.compute_reward(agent_type,
                            new_weak_pos, self.weak_goal_pos, action_w)
        return reward

    def compute_reward(self, agent_type, agent_pos, goal_pos, action,
                       other_pos=None, other_goal_pos=None):
        if agent_type == 'strong':
            rho_g = self.strong_rho_g
            delta_g = self.strong_delta_g
        else:
            rho_g = self.weak_rho_g
            delta_g = self.weak_delta_g
        reward = 0
        if goal_pos is None:  # compute the social reward for the strong agent
            other_reward = self._object_reward(other_pos, other_goal_pos,
                self.weak_rho_g, self.weak_delta_g)
            max_reward = self.weak_rho_g * 1/self.weak_delta_g
            # currently, the reward is a simple one purely based on the other agent's
            # TODO: should reason about the possibility of the other agent reaching its goal
            #       to attribute rewards correctly
            if self.strong_goal == SocialWorldEnv.Goals.HELP:
                reward = other_reward
            else:
                reward = max_reward - other_reward
        else:
            reward += self._object_reward(agent_pos, goal_pos, rho_g, delta_g)
        reward -= self._action_cost(action)
        return reward

    def step(self, action_s, action_w, pos_ctrl=False, pos_s=None, pos_w=None):
        # compute the agent movement
        if pos_ctrl:  # use position control or not, used to get training data
            d_strong = np.array(pos_s) - np.array(self.strong_pos)
            d_weak = np.array(pos_w) - np.array(self.weak_pos)
        else:
            # strong agent always succeed
            d_strong = self.dir2del[action_s]
            # weak agent successfully perform actions at probability p
            # TODO: define the failure actions
            d_weak = self.dir2del[action_w] \
                    if np.random.choice([True, False], p=[self.weak_prob, 1.-self.weak_prob]) \
                    else (0,0)
        # update weak agent's position
        if self._collision_free(self.weak_pos[0]+d_weak[0], self.weak_pos[1]+d_weak[1], is_weak=True):
            self.weak_pos[0] += d_weak[0]
            self.weak_pos[1] += d_weak[1]
        # update strong agent's position and related objects
        if self._collision_free(self.strong_pos[0]+d_strong[0], self.strong_pos[1]+d_strong[1]):
            self.strong_pos[0] += d_strong[0]
            self.strong_pos[1] += d_strong[1]
            self._move_boulder(d_strong)
            self._move_weak_agent(d_strong)
        # compute reward and check reaching the goal
        done_strong = self.strong_pos == self.strong_goal_pos or \
            (self.strong_goal == SocialWorldEnv.Goals.HELP and \
                self.weak_pos == self.weak_goal_pos)
        done_weak = self.weak_pos == self.weak_goal_pos
        reward_strong = self.compute_reward('strong', self.strong_pos,
            self.strong_goal_pos, action_s, self.weak_pos, self.weak_goal_pos)
        reward_weak = self.compute_reward('weak', self.weak_pos, self.weak_goal_pos, action_w)
        # extract features
        obs_strong = self._get_obs(self.strong_pos)
        obs_weak = self._get_obs(self.weak_pos)
        return (obs_strong, reward_strong, done_strong, {}), \
            (obs_weak, reward_weak, done_weak, {})

    def save_fig(self, path):
        self.gui.draw()
        img_str = pygame.image.tostring(self.gui._screen, 'RGB')
        img = Image.frombytes('RGB', self.gui._screen.get_size(), img_str)
        img.save(path)

if __name__ == '__main__':
    grid, strong_pos, strong_goal, weak_pos, weak_goal = \
            SocialWorldEnv.sample_init_grid()
    print("Strong agent's goal", SocialWorldEnv.Goals(strong_goal))
    print("Weak agent's goal", SocialWorldEnv.Goals(weak_goal))
    env = SocialWorldEnv(grid=grid, agent_strong=strong_pos,
            agent_strong_goal=strong_goal, agent_weak=weak_pos, agent_weak_goal=weak_goal)
    while True:
        env.gui.draw(move=True, reset_after_done=True)
