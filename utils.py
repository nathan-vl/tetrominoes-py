from dataclasses import dataclass


@dataclass
class Vec2:
    x: float
    y: float

    def __eq__(self, pos):
        return self.x == pos.x and self.y == pos.y

    def __add__(self, pos):
        return Vec2(self.x + pos.x, self.y + pos.y)

    def __sub__(self, pos):
        return Vec2(self.x - pos.x, self.y - pos.y)

    def __str__(self):
        return f'({self.x}, {self.y})'
