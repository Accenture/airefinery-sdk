# AI Refinery™ SDK  
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)


## Get Your API Key from AI Refinery

To start using AI Refinery™, you must first create an account and obtain an API key. Contact us through AI Refinery™ [official website](https://airefinery.accenture.com/contact) to submit your request and start your journey with AI Refinery™ SDK.

## Introduction

AI Refinery™ by Accenture is a Cloud Service for developing and executing AI multi-agent solutions. It is a platform designed to help organizations:

- Adopt and customize large language models (LLMs) to meet specific business needs.

- Integrate generative AI across various enterprise functions using a robust AI stack.

- Foster continuous innovation with minimal human intervention.

- Ensure seamless integration and ongoing advancements in AI capabilities within your projects.

The AI Refinery™ SDK is engineered to facilitate the development of AI solutions by leveraging features supported by AI Refinery such as:

- The Distiller Framework: A framework designed to streamline complex workflows by orchestrating various agents that handle different tasks.
- Inference API: An Api designed to seamlessly connect to any Language Model supported by AI Refinery™.
- Knowledge Extraction: An API that is designed to extract knowledge from various formats of input documents containing text, tables, and figures. 


## Documentation

Comprehensive documentation for the SDK is available on the [official documentation website](https://sdk.airefinery.accenture.com/).

All the APIs integrated in the AI Refinery™ SDK are detailed in [API page](https://sdk.airefinery.accenture.com/setup/api-reference/distiller-index/).

## Installation

We recommend using a MacOS or a Linux system to install the SDK. For Windows setup, we recommend using **WSL (Windows Subsystem Linux)**, a Linux kernel you can access from Windows. For instructions on installing WSL, please visit [this page](https://documentation.ubuntu.com/wsl/en/latest/guides/install-ubuntu-wsl2/). Please use **Ubuntu Distro 22.04** or above.

```bash
pip install "git+https://github.com/Accenture/airefinery-sdk.git@main"
```

## Examples

### 1. Design your Stock Investment Strategy Advisor

This [example](https://sdk.airefinery.accenture.com/tutorial/flow_superagent/tutorial_flow_superagent/) uses the AI Refinery™ SDK to create and run an AI system that can provide suggestions on investing in stocks. This example demonstrates:

- How to manage AI agents using a "Directed Acyclic Graph (DAG)" workflow representation.
- Usecases of AI Refinery™ `UtilityAgent` and `SuperAgent`.
- AI Agent parallel processing, allowing all tasks to begin execution as soon as they receive the necessary input information.

### 2. Integrate Your Custom Python Functions with the Tool Use Agent
   
The [**Tool Use Agent**](https://sdk.airefinery.accenture.com/tutorial/tutorial_tool_use/) is a utility agent designed to perform function calls using provided tools. It enables dynamic execution of functions based on user queries, allowing for a flexible and extensible system. By integrating both built-in and custom tools, the agent can process a wide range of tasks—from simple calculations to complex data processing.  

### 3. Integrate Responsible AI (RAI) into your project

The RAI module is a [**Responsible AI**](https://sdk.airefinery.accenture.com/tutorial/tutorial_rai_module/) framework designed to help you define, load, and apply safety or policy rules to user queries via a Large Language Model (LLM). This module automatically applies system base rules for RAI checks and allows users to create and add custom rules for specific needs.


### 4. Process your documents to extract the knowledge for your project

This exmaple introduces [**The Knowledge Extraction API**](https://sdk.airefinery.accenture.com/tutorial/knowledge_extraction/knowledge_extraction/) that allows you to send a document and then extract the knowledge/information contained within the documents. This example demonstrates:

- How to perform knowledge extraction tasks from your PPTX, PDF, DOCX, PPT, and DOC files. 
- The [DocumentProcessingClient](https://sdk.airefinery.accenture.com/setup/api-reference/knowledge-extraction-index/).
- How to construct a knowledge database for your project.


### 5. For more examples, check all our tutorials [here](https://sdk.airefinery.accenture.com/tutorial/).


## Quickstart

 
1. Create your first project

- Create a directory named `sdk-project`.  
- Inside the`sdk-project` directory, create a Python file named `example.py` to place the Python code needed to run the SDK Distiller client.  
- Also, within the `sdk-project` directory, create a YAML file named `example.yaml` to provide your project configuration.  

This gives you the following project structure:

```
sdk-project/  
│  
├── example.py  
├── example.yaml
```

2. Configure your project with a single YAML file

You can start by configuring the orchestrator for this project to have access to
  - A `CustomAgent` with the name `Assistant Agent` that you implement and add to executor_dict `executor_dict`
  - An AIRefienry built-in utility agent named `Search Agent.` that you can call and use as-is 

The settings for each of these utility agents are specified under `utility_agents`. You have the flexibility to expand your project based on your requirements. You can add additional custom agents that you define in the future or integrate built-in agents from our [agent library](../../distiller/agent-library/). 

```yaml
orchestrator:
  # Agent name list that the Orchestrator would route the user's queries to
  agent_list:
    - agent_name: "Assistant Agent"
    - agent_name: "Search Agent"

# List of all utility_agents active in the project
utility_agents:

  # Agent 1 configuration
  - agent_class: CustomAgent
    agent_name: "Assistant Agent"
    agent_description: "The Assistant Agent helps the users in their projects."
    config: {}

  # Agent 2 configuration
  - agent_class: SearchAgent
    agent_name: "Search Agent"

```

3. Create you first DistillerClient

[`DistillerClient`](../api-reference/distiller-index.md) API creates a distiller client. This client will interface with the AI Refinery™ service to run your project. Below is a function that sets up the distiller client. Here's what it does:  

- Authenticate you using `AIREFINERY_ACCOUNT` and `AIREFINERY_API_KEY` from your os envenvironment variables.
- Defines the `simple_agent` function that will cover the scope of the `Assistant Agent` using AI Refinery™ Inference-as-a-service.  
- Instantiates a `DistillerClient`.  
- Creates a project named `my_first_project` using the configuration specified in the `example.yaml` file.
- Adds the `simple_agent` to the `executor_dict` under the name `Assistant Agent`.
- Runs the project in `interactive` mode.  

```python
import asyncio
import os

from openai import AsyncOpenAI

from air import login, DistillerClient

auth = login(
    account=str(os.getenv("AIREFINERY_ACCOUNT")),
    api_key=str(os.getenv("AIREFINERY_API_KEY")),
)

async def simple_agent(query: str):
    global auth

    prompt = "You are an AI assistant that helps users navigate their projects."
    client = AsyncOpenAI(**auth.openai())

    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="meta-llama/Llama-3.1-70B-Instruct",
    )
    return response.choices[0].message.content

def distiller_client_test():
    distiller_client = DistillerClient()
    project_name = "my_first_project"

    # upload your config file to register a new distiller project
    distiller_client.create_project(config_path="example.yaml", project=project_name) 

    # Define a mapping between your custom agent to Callable.
    # When the custom agent is summoned by the super agent / orchestrator,
    # distiller-sdk will run the custom agent and send its response back to the
    # multi-agent system.
    executor_dict = {
        "Assistant Agent": simple_agent,
    }
    async with distiller_client(
        project=project_name,
        uuid="test_user",
        executor_dict=executor_dict,
    ) as dc:
        responses = await dc.query("Hello")
        async for response in responses:
            print(response["content"])  

if __name__ == "__main__":

    distiller_client_test()

```


### Execution

To execute `my_first_project`, run the following commands on your terminal:

```cmd
cd sdk-project/
python example.py
```

Running these commands will create your project on the AI Refinery™ server and run your Distiller client. 

## Releases & Versioning
`airefinery-sdk` is currently on version `1.MINOR.PATCH`.
The `airefinery-sdk` package defines the main interfaces and runtime logic for the entire AIRefinery Platform. To maintain stability, we will clearly announce any breaking changes in advance and reflect them through appropriate version updates and deprecation announcement.
As a rule, any changes that break compatibility in stable parts of the API will result in a minor or major version update, depending on the scope of the change.
### Minor version increases will occur for:
- Introduction of new agents or capabilities
- Additions to supported features
### Patch version increases will occur for:
- Bug fixes
- Minor improvements or refinements that do not affect API stability