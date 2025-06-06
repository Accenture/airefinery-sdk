"""
Contains the KnowledgeGraphRegistry: class that stores references to knowledge graph classes.
"""


class KnowledgeGraphRegistry:
    """
    A global registry that keeps track of all knowledge graph classes
    (subclasses of BaseKnowledgeGraph).
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
    def get(cls, name: str):
        """
        Retrieve a knowledge graph class by its name. Returns None if not found.
        """
        return cls._registry.get(name, None)

    @classmethod
    def all_knowledgegraphs(cls):
        """
        Return a list of all registered knowledge graph classes.
        """
        return list(cls._registry.keys())
