from ecosys.environment.entities import Entity


class Resource(Entity):
    def __init__(self, pos: tuple[int, int]):
        super().__init__(pos)
        # color
        self.color = (255, 255, 255)
