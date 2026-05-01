from pydantic import BaseModel


class PointModel(BaseModel):
    """Representation of a ordered pair"""

    x: float
    y: float

    def astuple(self):
        return (self.x, self.y)
