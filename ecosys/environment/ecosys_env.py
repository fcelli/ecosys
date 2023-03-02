import sys
import numpy
import random
import itertools
from typing import Optional
import gym
from gym.error import DependencyNotInstalled
from ecosys.environment.entities import Resource, Herbivore


class EcosysEnv(gym.Env):
    '''
    ### Description
    This environment describes describes the movement of a herbivore scavenging for resources on a square grid world.
    '''

    metadata = {
        'render_modes': ['human'],
        'render_fps': 10,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None
    ):
        super(EcosysEnv, self).__init__()
        # Grid dimension
        self.grid_dim = 10
        # Number of resources to be generated inside the grid
        self.n_resources = 20
        # Action space
        self.action_space = gym.spaces.Discrete(4)
        # Observation space
        self.observation_space = gym.spaces.MultiBinary([2, 4])
        # Initialize state and info
        self.state = None
        self.info = {}
        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.screen_width = 600
        self.screen_height = 600
        self.clock = None
        self.isopen = True

    def step(
        self,
        action: int
    ) -> tuple[numpy.ndarray, float, bool]:
        '''Execute one time step within the environment'''
        # Error handling
        err_msg = f'{action!r} ({type(action)}) invalid'
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, 'Call reset before using step method.'
        # Perform the action
        self._herb.move(action)
        # Determine if in wall
        self._is_in_wall = any([i not in range(0, self.grid_dim) for i in self._herb.pos])
        # Interact with resources
        self._has_eaten = False
        for r in self._res:
            if self._herb.interact(r):
                self._res.remove(r)
                self._has_eaten = True
        # Make observation
        self.state = self._get_obs()
        # Calculate reward
        reward = self._get_rw()
        # Determine if done
        terminated = (len(self._res) == 0) or self._is_in_wall
        # Update info
        self.info = self._get_info()
        # Render step
        if self.render_mode == 'human':
            self.render()
        return self.state, reward, terminated, False, self.info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> numpy.ndarray:
        '''Reset the environment state'''
        super().reset(seed=seed)
        # Parse options
        if options is not None:
            self.grid_dim = options.get('grid_dim') if 'grid_dim' in options else self.grid_dim
            self.n_resources = options.get('n_resources') if 'n_resources' in options else self.n_resources
        # Randomly generate entities on the grid
        self._herb, self._res = self._gen_ent()
        if self.render_mode == "human":
            self.render()
        # Update state and info
        self.state = self._get_obs()
        self.info = self._get_info()
        return self.state, self.info

    def render(self) -> None:
        '''Render the environment to screen'''
        # Check render mode has been set
        if self.render_mode is None:
            gym.logger.warn(
                'You are calling render method without specifying any render mode. '
                'You can specify the render_mode at initialization, '
                f'e.g. gym(\'{self.spec.id}\', render_mode=\'human\')'
            )
            return
        # Check pygame installation
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )
        # Initialize screen
        if self.screen is None:
            pygame.init()
            if self.render_mode == 'human':
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption('Ecosys-v0')
        # Initialize clock
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.state is None:
            return None
        # Draw objects to screen
        self.screen.fill((0, 0, 0))
        width = self.screen_width/self.grid_dim
        for r in self._res:
            pygame.draw.rect(
                self.screen,
                r.color,
                pygame.Rect(r.x*width, r.y*width, width, width)
            )
        pygame.draw.rect(
            self.screen,
            self._herb.color,
            pygame.Rect(self._herb.x*width, self._herb.y*width, width, width)
        )
        # Update screen
        if self.render_mode == 'human':
            pygame.event.pump()
            self.clock.tick(self.metadata['render_fps'])
            pygame.display.flip()
        # Handle quit button
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

    def close(self) -> None:
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def _gen_ent(self) -> tuple[Herbivore, list[Resource]]:
        '''Randomly generate the herbivor and resources on the grid'''
        prod = itertools.product(range(self.grid_dim), range(self.grid_dim))
        coords = random.sample(list(prod), self.n_resources+1)
        herb = Herbivore(coords[0])
        res = [Resource(pos) for pos in coords[1:]]
        return herb, res

    def _get_obs(self) -> numpy.ndarray:
        '''Return the current state of the environment'''
        state = numpy.zeros((2, 4))
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
        # Compute the wall array
        wall = numpy.array(
            [
                self._herb.y == 0,              # wall up
                self._herb.x == self.grid_dim,  # wall right
                self._herb.y == self.grid_dim,  # wall down
                self._herb.x == 0               # wall left
            ],
            dtype=numpy.uint8
        )
        # Create state array
        state = numpy.array([food, wall], dtype=numpy.uint8)
        return state

    def _get_info(self) -> dict:
        return {
            'herbivor_pos': self._herb.pos,
            'resources_remaining': len(self._res)
        }

    def _get_rw(self) -> float:
        '''Calculate the reward'''
        if len(self._res) == 0:
            return 100.
        elif self._has_eaten:
            return 10.
        elif self._is_in_wall:
            return -100.
        else:
            return -1./(2*(self.grid_dim - 1))
