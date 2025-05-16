from air.distiller.executor.executor import Executor
from air.distiller.executor.analytics_executor import AnalyticsExecutor
from air.distiller.executor.tool_executor import ToolExecutor
from air.distiller.executor.google_executor import GoogleExecutor
from air.distiller.executor.azure_executor import AzureExecutor
from air.distiller.executor.mcp_executor import MCPExecutor

agent_class_to_executor = {}
executor_list = [
    Executor,
    AnalyticsExecutor,
    ToolExecutor,
    GoogleExecutor,
    AzureExecutor,
    MCPExecutor,
]

for executor in executor_list:
    agent_class_to_executor[executor.agent_class] = executor
