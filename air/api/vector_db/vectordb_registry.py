"""
Contains the VectorDBRegistry class that stores references to vector DB classes.
"""


class VectorDBRegistry:
    """
    A global registry that keeps track of all vector db classes
    (subclasses of BaseVectorDB).
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
        Retrieve a vector DB class by its name. Returns None if not found.
        """
        return cls._registry.get(name, None)

    @classmethod
    def all_vectordbs(cls):
        """
        Return a list of all registered vector DB classes.
        """
        return list(cls._registry.values())
