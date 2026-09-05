from typing import Any, Dict, List, Optional

from personal_ai.core.logger import get_logger
from personal_ai.domain.tool import BaseTool, ToolDefinition, ToolResult

logger = get_logger(__name__)


class ToolRegistry:
    """Registry responsible for discovering, retrieving, and safely executing registered tools.

    Invariants:
    - Duplicate tool registrations are strictly rejected.
    - Unknown or unregistered tools fail safely with structured ToolResult.
    - Never executes unregistered capabilities or raw functions.
    - Preserves provider-agnostic, deterministic lookup.
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
            TypeError: If the tool is not an instance of BaseTool.
            ValueError: If the tool name is empty or already registered.
        """
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Expected BaseTool instance, received {type(tool).__name__}.")

        name = tool.name.strip() if tool.name else ""
        if not name:
            raise ValueError("Tool name must be a non-empty string.")

        if name in self._tools:
            raise ValueError(f"Tool with name '{name}' is already registered in ToolRegistry.")

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

    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Safely execute a registered tool by name with arguments.

        Unknown or unregistered tools return a structured ToolResult failure.

        Args:
            name: Tool name to execute.
            arguments: Dictionary of input arguments to pass to the tool.

        Returns:
            ToolResult: Structured execution output or structured error.
        """
        tool = self.get(name)
        if tool is None:
            logger.warning("Execution rejected for unregistered tool [name=%s]", name)
            return ToolResult(
                success=False,
                error=f"Tool '{name}' is not registered in the ToolRegistry.",
            )

        try:
            return await tool.execute(arguments)
        except Exception as exc:
            logger.error("Unexpected failure executing tool [name=%s]: %s", name, exc, exc_info=True)
            return ToolResult(
                success=False,
                error=f"Unexpected error executing tool '{name}': {str(exc)}",
            )
