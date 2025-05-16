import asyncio


async def async_input(query: str):
    """Asynchronous input"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, input, query)


async def async_print(*args):
    """Asynchronous print"""
    print(*args)
