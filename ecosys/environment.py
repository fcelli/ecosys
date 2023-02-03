import numpy
import numpy.typing as npt
import gym
import random
import itertools
import pygame
from ecosys.entities import Resource, Herbivor

class Ecosystem(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(
        self,
        grid_dim: int,
        n_resources: int,
        fov: int):
        super(Ecosystem, self).__init__()
        # Grid dimension
        self.grid_dim = grid_dim
        # Number of resources to be generated inside the grid
        self.n_resources = n_resources
        # Herbivor's field of view (must be odd number)
        self.fov = fov
        # Define action space (up, down, left, right)
        self.action_space = gym.spaces.Discrete(4)
        # Define observation space (grid cells within hebivor's fov)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.fov, self.fov), dtype=numpy.uint8)
    
    def step(self, action: int) -> tuple[npt.NDArray, float, bool]:
        '''Execute one time step within the environment'''
        self._has_moved = False
        self._has_eaten = False
        self._counter += 1
        # Perform the action
        if self._herb.pos[0] == 0 and action == 3:
            self._herb.energy -= 0.05
        elif self._herb.pos[0] == self.grid_dim-1 and action == 1:
            self._herb.energy -= 0.05
        elif self._herb.pos[1] == 0 and action == 0:
            self._herb.energy -= 0.05
        elif self._herb.pos[1] == self.grid_dim-1 and action == 2:
            self._herb.energy -= 0.05
        else:
            self._herb.move(action)
            self._has_moved = True
        # Interact with resources
        for r in self._res:
            if self._herb.interact(r):
                self._res.remove(r)
                self._herb.energy = 1
                self._has_eaten = True
        # Make observation
        obs = self._make_observation()
        # Calculate reward
        reward = self._calculate_reward()
        # Determine if done
        done = ((len(self._res) == 0) or (self._herb.energy <= 0))
        return obs, reward, done
        
    def reset(self) -> npt.NDArray:
        # Reset the state of the environment to an initial state
        self._herb, self._res = self._generate_entities()
        self._counter = 0
        return self._make_observation()

    def render(self, screen: pygame.Surface, mode: str='human', close: bool=False) -> None:
        # Render the environment to the screen
        screen.fill((0,0,0))
        resolution = screen.get_size()
        width = resolution[0]/self.grid_dim
        for r in self._res:
            pygame.draw.rect(screen, r.color, pygame.Rect(r.pos[0]*width, r.pos[1]*width, width, width))
        pygame.draw.rect(screen, self._herb.color, pygame.Rect(self._herb.pos[0]*width, self._herb.pos[1]*width, width, width))
        
    def _generate_entities(self) -> tuple[Herbivor, list[Resource]]:
        prod = itertools.product(range(self.grid_dim), range(self.grid_dim))
        coords = random.sample(list(prod), self.n_resources+1)
        herb = Herbivor(coords[0])
        res = [Resource(pos) for pos in coords[1:]]
        return herb, res
    
    def _make_observation(self) -> npt.NDArray:
        # Board representation
        board = numpy.zeros(shape=(self.grid_dim, self.grid_dim), dtype=numpy.uint8)
        for r in self._res:
            board[r.pos[0], r.pos[1]] = 1
        # Create observation
        obs = numpy.zeros(shape=(self.fov, self.fov), dtype=numpy.uint8)
        hx, hy = self._herb.pos
        def is_out_of_bounds(idx):
            if idx < 0 or idx > self.grid_dim-1:
                return True
            return False
        i=0
        for x in range(hx-self.fov//2, hx+self.fov//2+1):
            j=0
            if is_out_of_bounds(x):
                i+=1
                continue
            for y in range(hy-self.fov//2, hy+self.fov//2+1):
                if is_out_of_bounds(y):
                    j+=1
                    continue
                obs[i, j] = board[x, y]
                j+=1
            i+=1
        return obs
    
    def _calculate_reward(self) -> float:
        reward = 0
        if self._has_eaten:
            reward += 10
        if self._herb.energy <= 0:
            reward += -10
        if not self._has_moved:
            reward += -5
        return reward
    
    @property
    def counter(self) -> int:
        return self._counter