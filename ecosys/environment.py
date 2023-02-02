import gym
import random
import numpy
import itertools
import pygame
from ecosys.entities import Resource, Herbivor

GRID_DIM = 20
N_RESOURCES = 50
N_DISCRETE_ACTIONS = 4
FOV = (5, 5)

class Ecosystem(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(Ecosystem, self).__init__()
        # Generate resources and herbivor
        self._herb, self._res = self._generate_entities()
        self._step = 0
        # Define action and observation space
        # The herbivor can move in four directions: up, down, left, right
        self.action_space = gym.spaces.Discrete(N_DISCRETE_ACTIONS)
        # Herbivor's field of view: square image around it
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=FOV, dtype=numpy.uint8)

    def step(self, action):
        # Execute one time step within the environment
        self._step += 1
        # Perform the action
        if self._herb.pos[0] == 0 and action == 3:
            self._herb.energy -= 0.1
        elif self._herb.pos[0] == GRID_DIM-1 and action == 1:
            self._herb.energy -= 0.1
        elif self._herb.pos[1] == 0 and action == 0:
            self._herb.energy -= 0.1
        elif self._herb.pos[1] == GRID_DIM-1 and action == 2:
            self._herb.energy -= 0.1
        else:
            self._herb.move(action)
        # Interact with resources
        has_eaten = False
        for r in self._res:
            if self._herb.interact(r):
                self._res.remove(r)
                self._herb.energy == 1
                has_eaten = True
        # Make observation
        obs = self._make_observation()
        # Calculate reward
        reward = -0.1
        if has_eaten:
            reward = 5
        # Check if done
        done = len(self._res) == 0 or self._herb.energy <= 0
        return obs, reward, done
        
    def reset(self):
        # Reset the state of the environment to an initial state
        self._herb, self._res = self._generate_entities()
        self._step = 0

    def render(self, screen, mode='human', close=False):
        # Render the environment to the screen
        screen.fill((0,0,0))
        resolution = screen.get_size()
        width = resolution[0]/GRID_DIM
        for r in self._res:
            pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(r.pos[0]*width, r.pos[1]*width, width, width))
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(self._herb.pos[0]*width, self._herb.pos[1]*width, width, width))
        
    def _generate_entities(self):
        prod = itertools.product(range(GRID_DIM), range(GRID_DIM))
        coords = random.sample(list(prod), N_RESOURCES+1)
        herb = Herbivor(coords[0])
        res = [Resource(pos) for pos in coords[1:]]
        return herb, res
    
    def _make_observation(self):
        # Board representation
        board = numpy.zeros(shape=(GRID_DIM, GRID_DIM), dtype=numpy.uint8)
        for r in self._res:
            board[r.pos[0], r.pos[1]] = 1
        board[self._herb.pos[0], self._herb.pos[1]] = 2
        # Create observation
        obs = numpy.zeros(shape=FOV, dtype=numpy.uint8)
        hx, hy = self._herb.pos
        def is_out_of_bounds(idx):
            if idx < 0 or idx > GRID_DIM-1:
                return True
            return False
        i=0
        for x in range(hx-FOV[0]//2, hx+FOV[0]//2+1):
            j=0
            if is_out_of_bounds(x):
                i+=1
                continue
            for y in range(hy-FOV[1]//2, hy+FOV[1]//2+1):
                if is_out_of_bounds(y):
                    j+=1
                    continue
                obs[i, j] = board[x, y]
                j+=1
            i+=1
        return obs
        