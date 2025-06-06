from typing import override, Generic, List, Optional, TypeVar, AsyncIterator, Iterator
from pydantic import BaseModel


class CustomBaseModel(BaseModel):
    """An internal base model with a custom string representation.

    Extends Pydantic's BaseModel to override the string representation so that
    printing an instance will display the class name and key-value pairs of the
    model fields.
    """

    @override
    def __str__(self) -> str:
        """Returns a string like ModelName(field=value, field=value, ...)."""
        return f"{self.__repr_name__()}({self.__repr_str__(', ')})"


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
