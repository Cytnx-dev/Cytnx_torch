from dataclasses import dataclass


@dataclass
class AbstractAccessInterface:
    pass


@dataclass
class RegularAccessInterface(AbstractAccessInterface):
    pass
