from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, PrivateAttr


class Bar(BaseModel):
    counter: int = Field(default=0)
    _foo: "Foo" = PrivateAttr()

    def bind(self, foo: "Foo") -> None:
        self._foo = foo

    def perform_task(self) -> None:
        self.counter += 1


class Foo(BaseModel):
    value: float = Field(description="A Value")
    bar: Bar = Field(default_factory=Bar)

    def model_post_init(self, context: Any) -> None:
        self.bar.bind(self)

    def to_json(
        self, directory: Path | str | None = None, name: str | None = None
    ) -> None:
        if directory is None:
            directory = Path.cwd()
        if name is None:
            name = "test"
        filename = Path(directory, name).with_suffix(".json")
        with open(filename, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, fp: Path | str) -> Self:
        with open(fp, "r") as f:
            return cls.model_validate_json(f.read())


f = Foo(value=10.0)
for _ in range(400):
    f.bar.perform_task()
f.to_json()

f2 = Foo.from_json("./test.json")
print(f2.bar.counter)
