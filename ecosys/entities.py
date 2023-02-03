import numpy
from typing import TypeVar
Entity = TypeVar("Entity", bound='Entity')

class Entity:
    def __init__(self, pos: tuple[int, int]):
        # x, y cartesian coordinates
        self._pos = numpy.array(pos, dtype=numpy.uint8)
        # diet (list of types)
        self._diet = set()
        # color
        self._color = (0, 0, 0)
        # energy
        self._energy = 1

    def distance(self, ent: Entity) -> int:
        '''
        Returns the distance between two entities.
        '''
        if not isinstance(ent, Entity):
            raise TypeError
        delta_x = max(self.pos[0], ent.pos[0]) - min(self.pos[0], ent.pos[0])
        delta_y = max(self.pos[1], ent.pos[1]) - min(self.pos[1], ent.pos[1])
        return delta_x + delta_y

    def interact(self, ent: Entity) -> bool:
        '''
        Returns True if self and ent occupy the same position.
        '''
        if not isinstance(ent, Entity):
            raise TypeError
        return all(self.pos == ent.pos)

    def add_to_diet(self, obj_type: Entity) -> None:
        if not isinstance(obj_type, type(Entity)):
            raise TypeError
        self._diet.add(obj_type)
         
    def remove_from_diet(self, obj_type: Entity) -> None:
        if not isinstance(obj_type, type(Entity)):
            raise TypeError
        self._diet.discard(obj_type)

    def is_eaten_by(self, ent: Entity) -> bool:
        return type(self) in ent.diet
    
    def move(self, action: int) -> None:
        if action == 0:  # up
            self.pos[1] -= 1
        if action == 1:  # right
            self.pos[0] += 1
        if action == 2:  # down
            self.pos[1] += 1
        if action == 3:  # left
            self.pos[0] -= 1
        self.energy -= 0.05

    @property
    def pos(self) -> tuple[int, int]:
        return self._pos

    @pos.setter
    def pos(self, value: tuple[int, int]) -> None:
        self._pos = value

    @property
    def diet(self) -> set:
        return self._diet

    @property
    def color(self) -> tuple[int, int, int]:
        return self._color

    @color.setter
    def color(self, value: tuple[int, int, int]) -> None:
        self._color = value

    @property
    def energy(self) -> float:
        return self._energy

    @energy.setter
    def energy(self, value: float) -> None:
        self._energy = value


class Resource(Entity):
    def __init__(self, pos: tuple[int, int]):
        super().__init__(pos)
        # color
        self.color = (255, 255, 255)


class Herbivor(Entity):
    def __init__(self, pos: tuple[int, int]):
        super().__init__(pos)
        # resources are eaten by Herbivors
        self.add_to_diet(Resource)
        # color
        self.color = (0, 255, 0)
