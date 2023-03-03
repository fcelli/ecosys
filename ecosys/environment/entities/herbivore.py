from ecosys.environment.entities import Entity, Resource


class Herbivore(Entity):
    def __init__(self, pos: tuple[int, int]):
        super().__init__(pos)
        # resources are eaten by Herbivors
        self.add_to_diet(Resource)
        # color
        self.color = (0, 255, 0)
