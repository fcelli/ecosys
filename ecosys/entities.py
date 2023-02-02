import math
import numpy

class Entity:
    def __init__(self, pos:tuple, vel:float=0):
        # x, y cartesian coordinates
        self._pos = numpy.array(pos, dtype=numpy.uint8)
        # intensity of velocity vector
        self._vel = vel
        # diet (list of types)
        self._diet = set()
        # color
        self._color = None
        # energy
        self._energy = 1
        # size
        self._size = 1
        
    def distance(self, ent):
        '''
        Returns the distance between two entities.
        '''
        if not isinstance(ent, Entity):
            raise TypeError
        x, y = self.pos - ent.pos
        return math.sqrt(x*x + y*y)
    
    def rel_cart_coord(self, ent):
        '''
        Cartesian coordinates of Entity ent with self as origin.
        '''
        if not isinstance(ent, Entity):
            raise TypeError
        return ent.pos - self.pos
    
    def interact(self, ent):
        '''
        Returns True if the distance between self and ent is within the sum of their sizes.
        '''
        if not isinstance(ent, Entity):
            raise TypeError
        return all(self.pos == ent.pos)
    
    def add_to_diet(self, obj_type):
        if not isinstance(obj_type, type(Entity)):
            raise TypeError
        self._diet.add(obj_type)
                
    def remove_from_diet(self, obj_type):
        if not isinstance(obj_type, type(Entity)):
            raise TypeError
        self._diet.discard(obj_type)
        
    def is_eaten_by(self, ent):
        return type(self) in ent.diet
    
    def move(self, action):
        if action == 0: # up
            self.pos[1] -= 1
        if action == 1: # right
            self.pos[0] += 1
        if action == 2: # down
            self.pos[1] += 1
        if action == 3: # left
            self.pos[0] -= 1
        self.energy -= 0.1
    
    @property
    def pos(self):
        return self._pos
    
    @pos.setter
    def pos(self, value):
        self._pos = value
    
    @property
    def vel(self):
        return self._vel
    
    @vel.setter
    def vel(self, value):
        self._vel = value
    
    @property
    def diet(self):
        return self._diet
    
    @property
    def color(self):
        return self._color
    
    @color.setter
    def color(self, value):
        self._color = value
        
    @property
    def energy(self):
        return self._energy
    
    @energy.setter
    def energy(self, value):
        self._energy = value
        
    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, value):
        self._size = value
    
class Resource(Entity):
    def __init__(self, pos):
        super().__init__(pos, vel=0)
        
        # color
        self.color = (255, 255, 255)
        
class Herbivor(Entity):
    def __init__(self, pos):
        super().__init__(pos, vel=1)
        
        # resources are eaten by Herbivors
        self.add_to_diet(Resource)
        
        # color
        self.color = (0, 255, 0)