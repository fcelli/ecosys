import numpy
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
            n_resources: int):
        super(Ecosystem, self).__init__()
        # Grid dimension
        self.grid_dim = grid_dim
        # Number of resources to be generated inside the grid
        self.n_resources = n_resources
        # Action space
        self.action_space = gym.spaces.Discrete(4)
        # Observation space
        self.observation_space = gym.spaces.MultiBinary([2, 4])

    def step(self, action: int) -> tuple[numpy.ndarray, float, bool]:
        '''Execute one time step within the environment'''
        # Perform the action
        self._herb.move(action)
        # Determine if in danger
        self._is_in_danger = any([i not in range(0, self.grid_dim) for i in self._herb.pos])
        # Interact with resources
        self._has_eaten = False
        for r in self._res:
            if self._herb.interact(r):
                self._res.remove(r)
                self._has_eaten = True
        # Make observation
        state = self._make_observation()
        # Calculate reward
        reward = self._calculate_reward()
        # Determine if done
        done = (len(self._res) == 0) or self._is_in_danger
        # increase the counter
        self._counter += 1
        return state, reward, done

    def reset(self) -> numpy.ndarray:
        '''Reset the environment state'''
        # Reset the environment state
        self._herb, self._res = self._generate_entities()
        # Reset counter
        self._counter = 0
        return self._make_observation()

    def render(self, screen: pygame.Surface, mode: str = 'human', close: bool = False) -> None:
        '''Render the environment to screen'''
        screen.fill((0, 0, 0))
        resolution = screen.get_size()
        width = resolution[0]/self.grid_dim
        for r in self._res:
            pygame.draw.rect(screen, r.color, pygame.Rect(r.x*width, r.y*width, width, width))
        pygame.draw.rect(screen, self._herb.color, pygame.Rect(self._herb.x*width, self._herb.y*width, width, width))

    def _generate_entities(self) -> tuple[Herbivor, list[Resource]]:
        '''Randomly generate the herbivor and resources on the grid'''
        prod = itertools.product(range(self.grid_dim), range(self.grid_dim))
        coords = random.sample(list(prod), self.n_resources+1)
        herb = Herbivor(coords[0])
        res = [Resource(pos) for pos in coords[1:]]
        return herb, res

    def _make_observation(self) -> numpy.ndarray:
        '''Return the current state of the environment'''
        state = numpy.zeros((2, 4))
        if len(self._res) != 0:
            # Compute the food array
            food = numpy.zeros((4,), dtype=float)
            for res in self._res:
                inv_square_dist = 1./float(self._herb.distance(res)**2)
                if self._herb.y > res.y:  # food up
                    food[0] += inv_square_dist
                if self._herb.x < res.x:  # food right
                    food[1] += inv_square_dist
                if self._herb.y < res.y:  # food down
                    food[2] += inv_square_dist
                if self._herb.x > res.x:  # food left
                    food[3] += inv_square_dist
            idx = numpy.argmax(food)
            food = numpy.zeros((4,), dtype=numpy.uint8)
            food[idx] = 1
            # Compute the danger array
            danger = numpy.array(
               [
                    self._herb.y == 0,              # danger up
                    self._herb.x == self.grid_dim,  # danger right
                    self._herb.y == self.grid_dim,  # danger down
                    self._herb.x == 0               # danger left
               ],
               dtype=numpy.uint8
            )
            # Create state array
            state = numpy.array([food, danger], dtype=numpy.uint8)
        return state

    def _calculate_reward(self) -> float:
        '''Calculate the reward'''
        if len(self._res) == 0:
            return 100.
        elif self._has_eaten:
            return 10.
        elif self._is_in_danger:
            return -100.
        else:
            return -1./(2*(self.grid_dim - 1))

    @property
    def counter(self) -> int:
        return self._counter
