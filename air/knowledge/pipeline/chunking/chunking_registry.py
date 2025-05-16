"""
Contains the ChunkingRegistry class that stores references to chunking classes.
"""


class ChunkingRegistry:
    """
    A global registry that keeps track of all chunking classes
    (subclasses of BaseChunking).
    """

    _registry = {}

    @classmethod
    def register(cls, subclass):
        """
        Register a subclass in the global registry. The key is the class name,
        but you can choose any naming scheme (e.g., subclass.__name__ or
        a subclass-level variable).
        """
        name = subclass.__name__
        if name not in cls._registry:
            cls._registry[name] = subclass

    @classmethod
    def get(cls, name):
        """
        Retrieve a chunking class by its name. Returns None if not found.
        """
        return cls._registry.get(name, None)

    @classmethod
    def all_chunkings(cls):
        """
        Return a list of all registered chunking classes.
        """
        return list(cls._registry.values())
