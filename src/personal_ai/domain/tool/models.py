from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import inspect
from typing import Any, Dict, Optional, Tuple, Type
import uuid

from pydantic import BaseModel, ValidationError

from personal_ai.core.logger import get_logger

logger = get_logger(__name__)


class ToolPermission(str, Enum):
    """Capability classification for safety and authorization boundaries."""

    READ_ONLY = "READ_ONLY"
    WRITE = "WRITE"
    EXTERNAL_SIDE_EFFECT = "EXTERNAL_SIDE_EFFECT"


@dataclass(frozen=True)
class ToolExecutionContext:
    """Application-controlled execution context passed to tools.

    Contains security boundaries such as authenticated user identity.
    Cannot be specified or overridden by LLM-controlled tool arguments.
    """

    user_id: uuid.UUID
    conversation_id: Optional[uuid.UUID] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ToolResult:
    """Structured result container returned by all tool executions.

    Guarantees clean representation without leaking raw exceptions as output.
    """

    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ToolDefinition:
    """Public, model-agnostic tool definition container used for discovery and LLM exposure."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    permission: ToolPermission


class BaseTool(ABC):
    """Abstract base class representing a controlled capability.

    Provides declarative Pydantic input schema validation, safety categorization,
    and safe execution wrapping. All concrete tools must declare a Pydantic input_schema
    subclassing BaseModel (use an empty BaseModel for zero-input tools).
    """

    name: str
    description: str
    permission: ToolPermission = ToolPermission.READ_ONLY
    input_schema: Type[BaseModel]

    def get_definition(self) -> ToolDefinition:
        """Return the public ToolDefinition without exposing internal implementation details."""
        if (
            self.input_schema is not None
            and isinstance(self.input_schema, type)
            and issubclass(self.input_schema, BaseModel)
        ):
            schema = self.input_schema.model_json_schema()
        else:
            schema = {"type": "object", "properties": {}}

        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=schema,
            permission=self.permission,
        )

    def validate_arguments(self, arguments: Any) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate input arguments against declared Pydantic input_schema.

        Returns:
            Tuple of (is_valid, error_message, validated_kwargs_dict).
        """
        if not isinstance(arguments, dict):
            return (
                False,
                f"Invalid arguments for tool '{self.name}': expected dictionary of arguments, received {type(arguments).__name__}.",
                None,
            )

        if (
            self.input_schema is None
            or not isinstance(self.input_schema, type)
            or not issubclass(self.input_schema, BaseModel)
        ):
            return (
                False,
                f"Tool '{self.name}' must declare a valid Pydantic BaseModel as its input_schema.",
                None,
            )

        try:
            validated_model = self.input_schema.model_validate(arguments)
            return True, None, validated_model.model_dump()
        except ValidationError as exc:
            errors = exc.errors()
            formatted_errors = "; ".join(
                f"{'.'.join(str(loc) for loc in err.get('loc', []))}: {err.get('msg')}"
                for err in errors
            )
            return (
                False,
                f"Argument validation failed for tool '{self.name}': {formatted_errors}",
                None,
            )
        except Exception as exc:
            logger.error("Unexpected validation exception for tool '%s': %s", self.name, exc, exc_info=True)
            return (
                False,
                f"Argument validation failed for tool '{self.name}'.",
                None,
            )

    @abstractmethod
    async def _run(self, **kwargs: Any) -> Any:
        """Internal execution method implemented by concrete tools."""
        pass

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: Optional[ToolExecutionContext] = None,
    ) -> ToolResult:
        """Execute the tool with argument validation and exception safety wrapping.

        Guarantees that raw exceptions are caught, logged with stack trace,
        and returned as a safe generic ToolResult failure without leaking internal details.
        """
        is_valid, error_msg, validated_kwargs = self.validate_arguments(arguments)
        if not is_valid:
            logger.warning("Tool validation rejected [tool=%s, error=%s]", self.name, error_msg)
            return ToolResult(
                success=False,
                error=error_msg,
                metadata={"tool_name": self.name, "permission": self.permission.value},
            )

        kwargs = dict(validated_kwargs) if validated_kwargs is not None else {}

        # Forward application-controlled context if accepted by _run
        try:
            sig = inspect.signature(self._run)
            if "context" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                kwargs["context"] = context
        except Exception as exc:
            logger.debug("Failed to inspect _run signature for %s: %s", self.name, exc)

        try:
            if inspect.iscoroutinefunction(self._run):
                output = await self._run(**kwargs)
            else:
                output = self._run(**kwargs)
                if inspect.isawaitable(output):
                    output = await output

            return ToolResult(
                success=True,
                output=output,
                metadata={"tool_name": self.name, "permission": self.permission.value},
            )
        except Exception as exc:
            logger.error("Error during tool execution [tool=%s]: %s", self.name, exc, exc_info=True)
            return ToolResult(
                success=False,
                error="Tool execution failed.",
                metadata={"tool_name": self.name, "permission": self.permission.value},
            )
