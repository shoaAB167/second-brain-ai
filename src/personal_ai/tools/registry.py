from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from personal_ai.core.logger import get_logger
from personal_ai.domain.tool import (
    BaseTool,
    ToolDefinition,
    ToolExecutionContext,
    ToolResult,
)

logger = get_logger(__name__)


class ToolRegistry:
    """Registry responsible for discovering, retrieving, and safely executing registered tools.

    Invariants:
    - Duplicate tool registrations are strictly rejected.
    - All registered tools must declare a valid Pydantic BaseModel input_schema.
    - Unknown or unregistered tools fail safely with structured ToolResult.
    - Never executes unregistered capabilities or raw functions.
    - Preserves provider-agnostic, deterministic lookup and permission metadata.
    """

    def __init__(self, tools: Optional[List[BaseTool]] = None) -> None:
        """Initialize registry with an optional list of tools."""
        self._tools: Dict[str, BaseTool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance in the registry.

        Args:
            tool: BaseTool instance to register.

        Raises:
            TypeError: If the tool is not an instance of BaseTool or lacks a valid Pydantic input_schema.
            ValueError: If the tool name is empty or already registered.
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Expected BaseTool instance, received {type(tool).__name__}.")

        name = tool.name.strip() if getattr(tool, "name", None) else ""
        if not name:
            raise ValueError("Tool name must be a non-empty string.")

        if name in self._tools:
            raise ValueError(f"Tool with name '{name}' is already registered in ToolRegistry.")

        schema = getattr(tool, "input_schema", None)
        if schema is None or not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError(f"Tool '{name}' must declare a valid Pydantic BaseModel as its input_schema.")

        self._tools[name] = tool
        logger.info(
            "Registered tool capability in ToolRegistry [name=%s, permission=%s]",
            name,
            tool.permission.value,
        )

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieve a registered tool by its exact name.

        Args:
            name: Exact tool identifier.

        Returns:
            The registered BaseTool or None if not found.
        """
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check whether a tool is currently registered.

        Args:
            name: Exact tool identifier.

        Returns:
            True if registered, False otherwise.
        """
        return name in self._tools

    def list_tools(self) -> List[BaseTool]:
        """Return a list of all registered tool instances."""
        return list(self._tools.values())

    def list_definitions(self) -> List[ToolDefinition]:
        """Return public ToolDefinitions for all registered tools without exposing implementations."""
        return [tool.get_definition() for tool in self._tools.values()]

    async def execute_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        context: Optional[ToolExecutionContext] = None,
    ) -> ToolResult:
        """Safely execute a registered tool by name with arguments and application context.

        Unknown or unregistered tools return a structured ToolResult failure.
        Internal execution exceptions return a safe generic failure without leaking stack traces.

        Args:
            name: Tool name to execute.
            arguments: Dictionary of input arguments to pass to the tool.
            context: Application-controlled execution context (e.g. authenticated user_id).

        Returns:
            ToolResult: Structured execution output or structured error.
        """
        tool = self.get(name)
        if tool is None:
            logger.warning("Execution rejected for unregistered tool [name=%s]", name)
            return ToolResult(
                success=False,
                error=f"Tool '{name}' is not registered in the ToolRegistry.",
                metadata={"tool_name": name},
            )

        try:
            return await tool.execute(arguments, context=context)
        except Exception as exc:
            logger.error("Unexpected failure executing tool [name=%s]: %s", name, exc, exc_info=True)
            return ToolResult(
                success=False,
                error="Tool execution failed.",
                metadata={"tool_name": name, "permission": tool.permission.value},
            )


def create_tool_registry(retrieval_service: Optional[Any] = None) -> ToolRegistry:
    """Factory function constructing and configuring the production ToolRegistry.

    Serves as the explicit composition boundary for registering approved capabilities.
    If retrieval_service is provided, registers SearchPersonalMemoryTool.
    """
    from personal_ai.tools.memory_search import SearchPersonalMemoryTool

    registry = ToolRegistry()
    if retrieval_service is not None:
        registry.register(SearchPersonalMemoryTool(retrieval_service=retrieval_service))
    return registry
