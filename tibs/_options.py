import os
import sys

__all__ = ["Options"]


def is_interactive() -> bool:
    return hasattr(sys, "ps1")


class Options:
    """Returns the singleton module options instance.

    To query and change module options create an instance and set or get properties of it.

    .. code-block:: python

        if Options().byte_aligned:
            # ...
            Options().byte_aligned = False

    """

    _instance = None
    _byte_aligned: bool
    _no_color: bool
    _indent_size: int

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Options, cls).__new__(cls)
            cls._byte_aligned = False
            cls._verbose_bits_repr = is_interactive()
            no_color = os.getenv("NO_COLOR")
            cls._no_color = True if no_color else not is_interactive()
            cls._indent_size = 4
        return cls._instance

    def __init__(self):
        pass

    def __setattr__(self, name, value):
        if hasattr(self, name):
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"Cannot add new attribute '{name}' to Options. "
                                 f"Only existing attributes can be modified:\n{self!r}")

    def __repr__(self) -> str:
        attributes = {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith("_") and not callable(getattr(self, attr))
        }
        return "\n".join(f"{attr}: {value!r}" for attr, value in attributes.items())

    @property
    def byte_aligned(self) -> bool:
        """Governs the default byte alignment option in methods that use it."""
        return self._byte_aligned

    @byte_aligned.setter
    def byte_aligned(self, value: bool) -> None:
        self._byte_aligned = bool(value)

    @property
    def no_color(self) -> bool:
        """If True then no ANSI color codes will be used in output."""
        return self._no_color

    @no_color.setter
    def no_color(self, value: bool) -> None:
        self._no_color = bool(value)

    @property
    def indent_size(self) -> int:
        """The number of spaces used for indentation. Defaults to 4."""
        return self._indent_size

    @indent_size.setter
    def indent_size(self, value: int) -> None:
        value = int(value)
        if value < 0:
            raise ValueError("Indent size cannot be negative.")
        self._indent_size = value

