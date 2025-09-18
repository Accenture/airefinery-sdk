from collections.abc import MutableMapping
from typing import (
    Any,
    AsyncIterator,
    Generic,
    ItemsView,
    Iterator,
    KeysView,
    List,
    Optional,
    TypeVar,
    ValuesView,
    override,
)

from pydantic import BaseModel


class CustomBaseModel(BaseModel, MutableMapping):
    """Custom Pydantic BaseModel that behaves like a dictionary and a pretty
    ``__str__``.

    Every subclass of `CustomBaseModel` is a fully-validated Pydantic model *and*
    a `MutableMapping`.  Therefore the usual dict operations work:

        >>> class User(CustomBaseModel):
        ...     id: int
        ...     name: str
        ...
        >>> u = User(id=1, name='Alice')
        >>> u['name']
        'Alice'
        >>> 'id' in u
        True
        >>> u.update(name='Bob')
        >>> dict(u)
        {'id': 1, 'name': 'Bob'}
    """

    # ------------------------------------------------------------------ #
    #                          Internal helpers                          #
    # ------------------------------------------------------------------ #
    def _field_keys(self) -> KeysView[str]:
        """Return a view with all field names (Pydantic 1 & 2 safe).
        Returns:
            KeysView[str]: Dynamic view of the model’s field names.
        """
        cls = type(self)
        return cls.model_fields.keys()

    # ------------------------------------------------------------------ #
    #                          Pretty printing                           #
    # ------------------------------------------------------------------ #
    @override
    def __str__(self) -> str:
        """Return ``ModelName(field=value, ...)``."""
        return f"{self.__repr_name__()}({self.__repr_str__(', ')})"

    # ------------------------------------------------------------------ #
    #               Mandatory `MutableMapping` interface                 #
    # ------------------------------------------------------------------ #
    def __getitem__(self, key: str) -> Any:
        """Return the value for *key*."""
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __setitem__(self, key: str, value: Any) -> None:
        """Assign *value* to *key* after Pydantic validation."""
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:  # noqa: D401
        """Field deletion is not supported.
        The model’s schema is fixed; removing a field would break it.
        Convert the model to a plain dict first::
            data = self.model_dump()            # Pydantic 2
            # or: data = self.dict()            # Pydantic 1
            del data["field"]
        """
        raise TypeError(
            "Deletion of fields from CustomBaseModel is not supported.  "
            "Call `model_dump()` (Pydantic 2) or `dict()` (Pydantic 1) to obtain "
            "a plain dictionary before deleting keys."
        )

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Iterate over *keys* only (dict-compatible).
        Note:
            `pydantic.BaseModel.__iter__` is typed to yield
            ``Iterator[tuple[str, Any]]``—(key, value) pairs.
            For backward-compatibility with earlier releases of our API we
            deliberately deviate and return ``Iterator[str]`` so that
                >>> for key in model:
                ...     ...
            behaves exactly as it would for a plain ``dict``.
        """
        return iter(self._field_keys())

    def __len__(self) -> int:
        """Return the number of fields."""
        return len(self._field_keys())

    # ------------------------------------------------------------------ #
    #                    Optional convenience helpers                    #
    # ------------------------------------------------------------------ #
    def __contains__(self, item: object) -> bool:
        """Return ``True`` if *item* is a valid field name."""
        return item in self._field_keys()

    def get(self, key: str, default: Any | None = None) -> Any:
        """Return the value for *key* or *default* when absent."""
        return self[key] if key in self else default

    def keys(self) -> KeysView[str]:
        """Return a dynamic view of the model’s field names."""
        return self._field_keys()

    def items(self) -> ItemsView[str, Any]:
        """Return a dynamic view of ``(key, value)`` pairs."""
        # Build a *temporary* real dict (cheap – small and contiguous in C)
        return {k: getattr(self, k) for k in self._field_keys()}.items()

    def values(self) -> ValuesView[Any]:
        """Return a dynamic view of the field values."""
        return {k: getattr(self, k) for k in self._field_keys()}.values()


T = TypeVar("T")


class PageBase(CustomBaseModel, Generic[T]):
    """Common fields and iteration support for SyncPage/AsyncPage objects.

    Attributes:
        object: String label indicating the object type (e.g., "list").
        data: The payload, typically a list of items of type T.
        first_id: Optional identifier for the first item in the list.
        last_id: Optional identifier for the last item in the list.
        has_more: Whether there are more items available to fetch.
    """

    object: str
    data: List[T]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: Optional[bool] = None

    def __iter__(self) -> Iterator[T]:  # type: ignore[override]
        """Allows iteration directly over the data list."""
        return iter(self.data)


class SyncPage(PageBase[T]):
    """A synchronously fetched page of items."""


class AsyncPage(PageBase[T]):
    """An asynchronously fetched page of items.

    Adds asynchronous iteration support::

        async for m in page:
            ...
    """

    def __aiter__(self) -> AsyncIterator[T]:
        """Enables async iteration over the data in this page."""

        async def _gen() -> AsyncIterator[T]:
            for item in self.data:
                yield item

        return _gen()
