"""Module containing the SnowflakeExecutor for Snowflake Agent integration."""

import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict, Tuple

import requests

from air.distiller.executor.executor import Executor
from air.types.distiller.executor.snowflake_config import (
    SnowflakeAgentConfig,
    SnowflakeServicesConfig,
)

logger = logging.getLogger(__name__)


class SnowflakeExecutor(Executor):
    """
    Executor class for Snowflake Agent.
    """

    agent_class: str = "SnowflakeAgent"

    def __init__(
        self,
        func: Dict[str, Callable],
        send_queue: asyncio.Queue,
        account: str,
        project: str,
        uuid: str,
        role: str,
        utility_config: Dict[str, Any],
        return_string: bool = True,
    ):
        """Initializes the Snowflake Executor.

        Args:
            func: A dictionary mapping function names to callables.
            send_queue: An asyncio.Queue for sending output messages.
            account: The account identifier.
            project: The project identifier.
            uuid: A unique identifier for the session or request.
            role: The role identifier for this executor (e.g., "agent").
            utility_config: A configuration dictionary.
            return_string: Flag to determine if the result should be returned as a string.

        Raises:
            ValueError: If any required configuration key is missing.
        """
        logger.debug(
            "Initializing SnowflakeExecutor with role=%r, account=%r, project=%r, uuid=%r",
            role,
            account,
            project,
            uuid,
        )

        # Casting utility config to class-specific pydantic BaseModel
        self.snowflake_agent_config = SnowflakeAgentConfig(**utility_config)

        # Validate required fields in utility_config.
        self._snowflake_services: SnowflakeServicesConfig
        self._snowflake_services = self.snowflake_agent_config.snowflake_services
        self._snowflake_base_url = self.snowflake_agent_config.snowflake_base_url
        self._account_url = f"{self._snowflake_base_url}/api/v2/cortex/agent:run"
        self._sql_url = f"{self._snowflake_base_url}/api/v2/statements"

        # Initialize the headers once as it is static throughout the agent execution
        self.headers = {
            "Authorization": f"Bearer {os.getenv(self.snowflake_agent_config.snowflake_password)}",
            "Content-Type": "application/json",
        }

        # Initialize the base Executor with our specialized execution method.
        super().__init__(
            func=self._execute_agent,
            send_queue=send_queue,
            account=account,
            project=project,
            uuid=uuid,
            role=role,
            return_string=return_string,
        )

    def _execute_agent(self, **kwargs) -> str:
        """
        Executes the Snowflake agent using a prompt.
        """
        prompt = kwargs.get("prompt")
        if not prompt:
            raise ValueError(
                "Missing 'prompt' parameter in SnowflakeExecutor._execute_agent."
            )

        logger.debug("Running Snowflake agent with prompt=%r", prompt)
        # Execute the asynchronous agent run in a synchronous context.
        return asyncio.run(self._process_agent_request(prompt))

    async def _process_agent_request(self, prompt: str) -> str:
        """Processes the agent request asynchronously."""
        response = self.send_message(prompt)
        response_text = self.parse_response(response)
        logger.info("Snowflake agent response received (length=%d)", len(response_text))

        return response_text

    def send_message(self, prompt: str) -> requests.Response:
        """
        Sends the query to the Snowflake agent.
        """
        # Create body of the API request
        request_body = self.create_request_body(prompt)

        # Post request
        try:
            response = requests.post(
                self._account_url,
                headers=self.headers,
                json=request_body,
            )
            response.raise_for_status()  # Raise an HTTPError for bad responses
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to post message to the Snowflake agent: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise

        return response

    def create_request_body(self, prompt: str) -> dict:
        """
        Creates the request body for the Snowflake agent.
        """
        tools = []
        tool_resources = {}

        # Handle search services
        for service in self._snowflake_services.search:
            name = service.name
            tools.append({"tool_spec": {"type": "cortex_search", "name": name}})
            full_name = f"{service.database}.{service.db_schema}.{service.service_name}"
            tool_resources[name] = {"name": full_name}

        # Handle analyst services
        for service in self._snowflake_services.analyst:
            name = service.name
            tools.append(
                {"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": name}}
            )
            semantic_model_path = (
                f"@{service.database}."
                f"{service.db_schema}."
                f"{service.stage}/"
                f"{service.file_name}"
            )
            tool_resources[name] = {"semantic_model_file": semantic_model_path}

        request_body = {
            "model": self.snowflake_agent_config.snowflake_model,
            "response_instruction": self.snowflake_agent_config.system_prompt,
            "experimental": self.snowflake_agent_config.snowflake_experimental,
            "tools": tools,
            "tool_resources": tool_resources,
            "tool_choice": {"type": self.snowflake_agent_config.snowflake_tool_choice},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        }

        return request_body

    def parse_response(self, response: requests.Response) -> str:
        """
        Parses the response of the Snowflake agent.
        """
        response_text = ""
        if response.status_code == 200:
            try:
                response_text, tracing_text = self.parse_response_text(response)
                if self.snowflake_agent_config.thought_process_tracing:
                    response_text += tracing_text
            except Exception as e:
                print("An error occurred while processing the response:", e)
        else:
            logger.exception("Failed to retrieve messages from the Snowflake agent.")
            raise Exception("Failed to retrieve messages from the Snowflake agent.")

        return response_text

    def parse_response_text(self, response: requests.Response) -> Tuple[str, str]:
        """
        Extracts the textual part of the Snowflake agent's response. In addition,
        extracts the intermediate thought processing information of the agent.
        """
        response_text = ""
        tracing_text = ""
        self.tracing_index = 0  # For tracking the number of thought process blocks
        sql_info = {}

        # Split the full response text by double newlines into blocks
        event_blocks = response.content.decode("utf8").strip().split("\n\n")
        if not event_blocks:
            raise ValueError("Response text is empty or improperly formatted.")
        # Validate that 'event_blocks' is a list
        if not isinstance(event_blocks, list):
            raise TypeError(
                f"'event_blocks' must be of type list, but got {type(event_blocks).__name__}."
            )

        for event in event_blocks:
            # Ensure it's a message.delta event
            if not event.startswith("event: message.delta"):
                continue
            data_dict = self.parse_event_block(event)
            content = self.parse_data_dictionary(data_dict)

            for block in content:
                if "type" not in block:
                    raise KeyError(
                        "The block dictionary does not contain the 'type' key."
                    )
                if block["type"] == "tool_results":
                    parsed_tracing_text, sql_info = self.handle_tool_results_block(
                        block
                    )
                    tracing_text += parsed_tracing_text

                elif block["type"] == "text":
                    parsed_response_text = self.handle_text_block(block)
                    response_text += parsed_response_text

        # In case the request needs SQL execution:
        if not response_text:
            sql_response = self.send_sql_message(sql_info)
            parsed_sql_text = self.parse_sql_response(sql_response)
            response_text += parsed_sql_text

        return response_text, tracing_text

    def parse_event_block(self, event: str) -> dict:
        """
        Parses the data dictionary from a single SSE event.
        """
        # Process the event block to extract the JSON string
        json_str = None
        for line in event.splitlines():
            if line.startswith("data: "):
                json_str = line[len("data: ") :]
                break
        if json_str is None:
            raise ValueError("No 'data: ' line found in response.")
        # Parse the JSON string to a data dictionary
        try:
            data_dict = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decoding failed: {e}") from e

        return data_dict

    def parse_data_dictionary(self, data_dict: dict) -> list:
        """
        Parses the data dictionary and returns its content.
        """
        # Extract the content information from the data dictionary
        try:
            delta = data_dict.get("delta")
            if not delta or "content" not in delta:
                raise ValueError(
                    "Missing 'delta' or 'content' key in the response data."
                )
            content = delta["content"]
            # Validate that 'content' is a list
            if not isinstance(content, list):
                raise TypeError(
                    f"'content' must be of type list, but got {type(content).__name__}."
                )

        except Exception as e:
            raise ValueError(
                f"Failed to extract 'content' from response data: {e}"
            ) from e

        return content

    def handle_tool_results_block(self, block) -> Tuple[str, dict]:
        """
        For a single content block that relates to thought processing steps,
        extracts its textual information and any corresponding SQL commands.
        """
        sql_info = {}
        # Validate that 'tool_results' exists and is a dictionary
        tool_results = block.get("tool_results")
        if not isinstance(tool_results, dict):
            raise ValueError(
                f"'tool_results' is missing or not a dictionary in block: {block}"
            )

        # Validate that 'type' exists and is a string
        tool_type = tool_results.get("type")
        if not isinstance(tool_type, str):
            raise ValueError(
                f"'type' is missing or not a string in tool_results: {tool_results}"
            )

        parsed_tracing_text = "\n" + "-" * 30
        if tool_type == "cortex_search":
            parsed_tracing_text = self.parse_search_tracing_text(
                tool_results, parsed_tracing_text
            )
        elif tool_type == "cortex_analyst_text_to_sql":
            parsed_tracing_text, sql_info = self.parse_analyst_tracing_text(
                tool_results, parsed_tracing_text
            )

        return parsed_tracing_text, sql_info

    def parse_search_tracing_text(
        self, tool_results: dict, parsed_tracing_text: str
    ) -> str:
        """
        Extracts the thought processing information from a Cortex Search
        service result.
        """
        parsed_tracing_text += f"\n***Thought process of Cortex Search services:***\n"

        # Validate that 'content' exists in tool_results
        if "content" not in tool_results:
            raise KeyError("'content' key is missing in tool_results dictionary.")

        tool_contents = tool_results["content"]
        # Validate that 'tool_contents' is a list
        if not isinstance(tool_contents, list):
            raise TypeError(
                f"'content' key must be of type list, but got {type(tool_contents).__name__}."
            )

        for tool_content in tool_contents:
            # Validate that 'type' exists
            if "type" not in tool_content:
                raise KeyError("'type' key is missing in tool_content dictionary.")

            if tool_content["type"] == "json":
                # Validate that 'searchResults' exists in tool_content['json']
                if "searchResults" not in tool_content["json"]:
                    raise KeyError(
                        "'searchResults' key is missing in tool_content['json'] dictionary."
                    )
                # Validate that 'searchResults' is a list
                search_results = tool_content["json"]["searchResults"]
                if not isinstance(search_results, list):
                    raise TypeError(
                        f"'searchResults' key must be of type list, but got {type(search_results).__name__}."
                    )

                # Process each search result
                for search_result in search_results:
                    if not isinstance(search_result, dict):
                        raise TypeError(
                            f"Each search result must be of type dict, but got {type(search_result).__name__}."
                        )
                    if "text" not in search_result:
                        raise KeyError(
                            "'text' key is missing in search_result dictionary."
                        )

                    self.tracing_index += 1
                    parsed_tracing_text += f"\nTracing block #{self.tracing_index}:\n"
                    parsed_tracing_text += search_result["text"] + "\n"

        return parsed_tracing_text

    def parse_analyst_tracing_text(
        self, tool_results: dict, parsed_tracing_text: str
    ) -> Tuple[str, dict]:
        """
        Extracts the thought processing information from a Cortex Analyst
        service result.
        """
        sql_query = ""
        service_name = ""
        parsed_tracing_text += f"\n***Thought process of Cortex Analyst services:***\n"

        # Validate that 'content' exists in tool_results
        if "content" not in tool_results:
            raise KeyError("'content' key is missing in tool_results dictionary.")

        tool_contents = tool_results["content"]
        # Validate that 'tool_contents' is a list
        if not isinstance(tool_contents, list):
            raise TypeError(
                f"'content' key must be of type list, but got {type(tool_contents).__name__}."
            )

        for tool_content in tool_contents:
            # Validate that 'type' exists
            if "type" not in tool_content:
                raise KeyError("'type' key is missing in tool_content dictionary.")

            if tool_content["type"] == "json":
                self.tracing_index += 1
                parsed_tracing_text += f"\nTracing block #{self.tracing_index}:\n"

                # Validate that 'text' exists
                if "text" not in tool_content["json"]:
                    raise KeyError(
                        "'text' key is missing in tool_content['json'] dictionary."
                    )
                sql_text = tool_content["json"]["text"]
                parsed_tracing_text += sql_text + "\n"

                # Validate that 'sql' exists
                if "sql" not in tool_content["json"]:
                    raise KeyError(
                        "'sql' key is missing in tool_content['json'] dictionary."
                    )
                sql_query += tool_content["json"]["sql"]
                parsed_tracing_text += sql_query + "\n"

                # Validate that 'name' exists in tool_results
                if "name" not in tool_results:
                    raise KeyError("'name' key is missing in tool_results dictionary.")
                service_name = tool_results["name"]

        sql_info = {"sql_query": sql_query, "service_name": service_name}
        return parsed_tracing_text, sql_info

    def handle_text_block(self, block: dict) -> str:
        """
        Extracts textual response from a single content block.
        """
        if "text" not in block:
            raise KeyError("The block dictionary does not contain the 'text' key.")
        parsed_response_text = block["text"]

        return parsed_response_text

    def send_sql_message(self, sql_info: dict) -> requests.Response:
        """
        Sends a SQL query to the Snowflake service and returns the response of the request.
        """
        sql_query = sql_info["sql_query"]
        if not sql_query:
            raise ValueError("SQL query is empty or improperly formatted.")
        service_name = sql_info["service_name"]
        analyst_service = next(
            (
                service
                for service in self._snowflake_services.analyst
                if service.name == service_name
            ),
            None,
        )
        if analyst_service is None:
            raise ValueError(f"Analyst service '{service_name}' not found in config.")

        sql_request_body = {
            "statement": sql_query,
            "timeout": self.snowflake_agent_config.sql_timeout,
            "database": analyst_service.database,
            "schema": analyst_service.db_schema,
            "warehouse": analyst_service.warehouse,
            "role": analyst_service.user_role,
        }

        sql_response = requests.post(
            self._sql_url,
            headers=self.headers,
            json=sql_request_body,
        )

        return sql_response

    def parse_sql_response(self, sql_response: requests.Response) -> str:
        """
        Parses the response of the SQL request and returns the textual part of it.
        """
        if sql_response.status_code == 200:
            try:
                sql_content = sql_response.content.decode("utf-8")
                if not sql_content:
                    raise ValueError("SQL Response is empty or improperly formatted.")

                sql_content_json = json.loads(sql_content)
                sql_data = sql_content_json.get("data")
                if sql_data is None:
                    raise ValueError("'data' field is missing in SQL response.")

                return str(sql_data)

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"An error occurred while decoding the SQL response: {e}"
                )
            except Exception as e:
                raise ValueError(
                    f"An error occurred while processing the SQL response: {e}"
                )
        else:
            raise ValueError(
                f"SQL Request Failed: {sql_response.status_code} {sql_response.text}"
            )
