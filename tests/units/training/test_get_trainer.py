# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass, field

from omegaconf import DictConfig

from lcm.train.common import get_trainer


@dataclass
class Foo:
    a: float = 0
    b: float = 0
    c: float = field(init=False)

    def __post_init__(self):
        self.c = self.a + self.b


@dataclass
class Config:
    foobar: str = "test"
    cfg: Foo = field(default_factory=lambda: Foo())
    c: float = field(init=False)

    def __post_init__(self):
        self.c = 10.0


class TrainerClass:
    def __init__(self, config: Config) -> None:
        self.config = config
        pass


def trainer_builder(config: Config):
    return TrainerClass(config)


def test_get_trainer_fn():
    conf_dict = DictConfig(
        {
            "_trainer_": f"{trainer_builder.__module__}.trainer_builder",
            "foobar": "bar",
            "cfg": {
                "a": 1,
                "b": 3,
            },
        },
    )
    tr = get_trainer(conf_dict)
    assert isinstance(tr, TrainerClass)
    assert tr.config.foobar == "bar"
    assert tr.config.cfg.c == 4.0


def test_get_trainer_class():
    conf_dict = DictConfig(
        {
            "_trainer_": f"{TrainerClass.__module__}.TrainerClass",
            "foobar": "bar",
        }
    )
    tr = get_trainer(conf_dict)
    assert isinstance(tr, TrainerClass)
    assert tr.config.foobar == "bar"
