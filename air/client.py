# pylint: disable=too-many-instance-attributes, too-few-public-methods
"""
A module that defines top-level clients (synchronous and asynchronous)
to interact with sub-clients such as chat, embeddings, models, and distiller.
"""

from air.chat import (
    AsyncChatClient,
    ChatClient,
)
from air.distiller import AsyncDistillerClient
from air.embeddings import (
    AsyncEmbeddingsClient,
    EmbeddingsClient,
)
from air.images import (
    AsyncImagesClient,
    ImagesClient,
)
from air.knowledge import AsyncKnowledgeClient, KnowledgeClient
from air.models import (
    AsyncModelsClient,
    ModelsClient,
)


class AsyncAIRefinery:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    A top-level client that exposes various sub-clients in a single interface,
    operating asynchronously.

    Example usage:

        client = AsyncAIRefinery(
            api_key="...",
            base_url="...",
            default_headers={"X-Client-Version": "1.2.3"}
        )

        # Use chat
        response = await client.chat.completions.create(
            model="model-name", messages=[...]
        )

        # Use embeddings
        embeddings_response = await client.embeddings.create(
            model="model-name", input=["Hello"]
        )

        # Use models
        models_list = await client.models.list()

        # Use distiller
        async with client.distiller(project="...", uuid="...") as dc:
            responses = await dc.query(query="hi")
            async for response in responses:
                print(response)

        # Use images
        embeddings_response = await client.images.generate(
            prompt="A cute baby sea otter", model="model-name"
        )

        # Use knowledge
        graph_client = client.graph
        graph_client.create_project(graph_config=...)
        status = await graph_client.build(files_path=..)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.airefinery.accenture.com",
        default_headers: dict[str, str] | None = None,
        **kwargs
    ):
        """
        Initializes the asynchronous unified client with sub-clients.

        Args:
            api_key (str): Your API key or token for authenticated requests.
            base_url (str, optional): Base URL for your API endpoints.
                Defaults to "https://api.airefinery.accenture.com".
            default_headers (dict[str, str] | None): Headers that apply to
                every request made by sub-clients (e.g., {"X-Client-Version": "1.2.3"}).
            **kwargs: Additional configuration parameters, if any.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.kwargs = kwargs

        # Provides async chat functionalities
        self.chat = AsyncChatClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
        )
        # Provides async embeddings functionalities
        self.embeddings = AsyncEmbeddingsClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
        )
        # Provides async models functionalities
        self.models = AsyncModelsClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
        )
        # Provides async distiller functionalities
        self.distiller = AsyncDistillerClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
        )
        # Provides async images functionalities
        self.images = AsyncImagesClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
        )

        # Provides async knowledge functionalities
        self.knowledge = AsyncKnowledgeClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
        )


class AIRefinery:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    A top-level client that exposes various sub-clients in a single interface,
    operating synchronously.

    Example usage:

        client = AIRefinery(
            api_key="...",
            base_url="...",
            default_headers={"X-Client-Version": "1.2.3"}
        )

        # Use chat
        response = client.chat.completions.create(
            model="model-name", messages=[...]
        )

        # Use embeddings
        embeddings_response = client.embeddings.create(
            model="model-name", input=["Hello"]
        )

        # Use models
        models_list = client.models.list()

        # Use images
        image_response = await client.images.generate(
            prompt="A cute baby sea otter", model="model-name"
        )

        # Use knowledge
        knowledge_client = client.knowledge
        knowledge_client.create_project(config)
        knowledge_response = await knowledge_client.document_processing.parse_documents(file_path='', model='')

        # Attempting to use client.distiller will raise an exception
        # (not supported in sync mode).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.airefinery.accenture.com",
        default_headers: dict[str, str] | None = None,
        **kwargs
    ):
        """
        Initializes the synchronous unified client with sub-clients.

        Args:
            api_key (str): Your API key or token for authenticated requests.
            base_url (str, optional): Base URL for your API endpoints.
                Defaults to "https://api.airefinery.accenture.com".
            default_headers (dict[str, str] | None): Headers that apply to
                every request made by sub-clients (e.g., {"X-Client-Version": "1.2.3"}).
            **kwargs: Additional configuration parameters, if any.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.kwargs = kwargs

        # Provides sync chat functionalities
        self.chat = ChatClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
        )
        # Provides sync embeddings functionalities
        self.embeddings = EmbeddingsClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
        )
        # Provides sync models functionalities
        self.models = ModelsClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
        )
        # Provides sync embeddings functionalities
        self.images = ImagesClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
        )

        # Provides sync knowledge functionalities
        self.knowledge = KnowledgeClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=self.default_headers,
        )

    @property
    def distiller(self):
        """
        Distiller is only supported in the asynchronous client.
        Accessing this property in the synchronous client will raise a NotImplementedError.
        """
        raise NotImplementedError(
            "Distiller is only available in async mode. Please use AsyncAIRefinery instead."
        )
