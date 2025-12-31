# -*- coding: utf-8 -*-
"""
This file provides the FunctionTool class.
"""

import asyncio
import inspect
from typing import Any, Callable, Dict, Optional, Type

from typing_extensions import override

from agentsdk.message import ContentElement, MimeType
from agentsdk.tools.base_tool import BaseTool, FunctionDeclarationDict
from agentsdk.tools.tool_context import ToolContext
from agentsdk.tools.tool_response import ToolResponse

# docstring parsing is optional (useful for descriptions), tolerate absence
try:
    from docstring_parser import parse as parse_docstring
except Exception:
    parse_docstring = None

# pydantic (for args_schema)
try:
    from pydantic import BaseModel, ValidationError
except Exception:
    BaseModel = None
    ValidationError = Exception  # fallback to generic


class FunctionTool(BaseTool):
    """
    FunctionTool supports two declaration/validation modes:
      1. args_schema (Pydantic BaseModel class) — recommended for complex/nested params.
      2. func (callable) — fallback to inspect.signature + optional docstring parse.
    """

    def __init__(
        self,
        func: Optional[Callable[..., Any]] = None,
        args_schema: Optional[Type["BaseModel"]] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        # Determine name/description
        if name is None:
            name = (
                getattr(func, "__name__", None)
                if func
                else (args_schema.__name__ if args_schema else "unnamed_tool")
            )
        if description is None:
            if func:
                # use docstring of func as default description
                description = inspect.cleandoc(getattr(func, "__doc__", "")) or ""
            elif args_schema and hasattr(args_schema, "__doc__"):
                description = (
                    inspect.cleandoc(getattr(args_schema, "__doc__", "")) or ""
                )
            else:
                description = ""

        super().__init__(name=name, description=description)
        self.func = func
        self.args_schema = args_schema

    @override
    def get_declaration(self) -> Optional[FunctionDeclarationDict]:
        """
        Returns a JSON-schema-like declaration dict for the tool.

        Priority:
        - Case A: Use args_schema (Pydantic model) if provided — preferred for complex/nested schemas.
        - Case B: Fallback to inspecting function signature and docstring — for lightweight tools.
        """
        # Case A: pydantic args_schema provided -> use model JSON schema
        if self.args_schema is not None and BaseModel is not None:
            # model_json_schema() in pydantic v2; schema() in v1
            try:
                raw_schema = self.args_schema.model_json_schema()
            except Exception:
                # fallback to pydantic v1 style
                raw_schema = getattr(self.args_schema, "schema", lambda: {})()

            expanded_schema = self.resolve_refs(raw_schema)

            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": expanded_schema.get("properties", {}),
                    "required": expanded_schema.get("required", []),
                },
            }

        # Case B: fallback to func signature + docstring (best-effort)
        if self.func is None:
            return {
                "name": self.name,
                "description": self.description,
            }

        # parse docstring for short desc and per-param descriptions (if docstring_parser available)
        short_desc = self.description
        param_docs = {}
        returns_desc = ""
        if parse_docstring is not None:
            doc = inspect.getdoc(self.func) or ""
            try:
                parsed = parse_docstring(doc)
                short_desc = parsed.short_description or short_desc
                param_docs = {p.arg_name: p.description for p in parsed.params}
                if parsed.returns:
                    returns_desc = parsed.returns.description or ""
            except Exception:
                # parsing errors -> ignore, keep defaults
                pass

        sig = inspect.signature(self.func)
        properties: Dict[str, Any] = {}
        required = []
        for pname, param in sig.parameters.items():
            if pname == "tool_context":
                # skip framework-injected param from schema
                continue

            ann = param.annotation
            if ann in (int, "int"):
                schema = {"type": "integer"}
            elif ann in (float, "float"):
                schema = {"type": "number"}
            elif ann in (bool, "bool"):
                schema = {"type": "boolean"}
            elif ann in (str, "str"):
                schema = {"type": "string"}
            elif ann in (dict, "dict"):
                schema = {"type": "object"}
            elif ann in (list, "list", tuple, "tuple"):
                schema = {"type": "array"}
            else:
                schema = {"type": "string"}  # fallback

            # attach description from parsed docstring if present
            if pname in param_docs and param_docs[pname]:
                schema["description"] = param_docs[pname]

            properties[pname] = schema
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        decl = {
            "name": self.name,
            "description": short_desc or self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
        if returns_desc:
            decl["returns"] = returns_desc
        return decl

    @override
    async def arun(
        self, *, args: Dict[str, Any], tool_context: Optional[ToolContext] = None
    ) -> ToolResponse:
        """Run the tool asynchronously, returning ToolResponse.
        Args:
        Args:
            args (Dict[str, Any]): Arguments to pass to the tool.
            tool_context (Optional[ToolContext]): An optional ToolContext instance to use for running the tool.
        Returns:
            ToolResponse:
        """
        validated_args, error = self._prepare_call_args(args, tool_context)
        if error:
            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.TEXT_PLAIN,
                        metadata={"error": True},
                        data=error,
                    )
                ],
                err_msg=error,
            )

        try:
            result = await self._maybe_await(self.func(**validated_args))
            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.APPLICATION_JSON,
                        data=result,
                        metadata={"tool": self.name},
                    )
                ]
            )
        except Exception as e:
            err = str(e)
            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.TEXT_PLAIN,
                        data=err,
                        metadata={"error": True},
                    )
                ],
                err_msg=err,
            )

    @override
    def run(
        self, *, args: Dict[str, Any], tool_context: Optional[ToolContext] = None
    ) -> ToolResponse:
        """Run the tool synchronously, returning ToolResponse."""

        validated_args, error = self._prepare_call_args(args, tool_context)
        if error:
            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.TEXT_PLAIN,
                        metadata={"error": True},
                        data=error,
                    )
                ],
                err_msg=error,
            )

        try:
            result = self.func(**validated_args)
            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.APPLICATION_JSON,
                        data=result,
                        metadata={"tool": self.name},
                    )
                ]
            )
        except Exception as e:
            err = str(e)
            return ToolResponse(
                content=[
                    ContentElement(
                        mime_type=MimeType.TEXT_PLAIN,
                        data=err,
                        metadata={"error": True},
                    )
                ],
                err_msg=err,
            )

    def _prepare_call_args(
        self, args: dict[str, Any], tool_context: Optional[ToolContext] = None
    ) -> tuple[dict[str, Any] | None, str | None]:
        """
        Validate/prepare call arguments for the tool.

        Returns:
            - (validated_args, None) on success
            - (None, error_msg) on failure
        """
        args_copy = dict(args or {})

        # Case A: Pydantic args_schema provided
        if self.args_schema is not None and BaseModel is not None:
            try:
                model_instance = self.args_schema(**args_copy)
            except ValidationError as e:
                return None, f"Invalid args for {self.name}: {str(e)}"

            # extract plain dict
            if hasattr(model_instance, "model_dump"):
                validated = model_instance.model_dump()
            elif hasattr(model_instance, "dict"):
                validated = model_instance.dict()
            else:
                validated = dict(model_instance.__dict__)

            # inject tool_context if func expects it
            if self.func is not None:
                sig = inspect.signature(self.func)
                if "tool_context" in sig.parameters and tool_context is not None:
                    validated["tool_context"] = tool_context
            return validated, None

        # Case B: no args_schema, fallback to function signature
        if self.func is None:
            return args_copy, None

        sig = inspect.signature(self.func)
        var_kwargs_param = None
        filtered_params = []
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                var_kwargs_param = param
            else:
                filtered_params.append(param)

        # inject tool_context if needed
        if "tool_context" in sig.parameters and tool_context is not None:
            args_copy["tool_context"] = tool_context

        # check missing mandatory parameters
        mandatory = [
            p.name
            for p in filtered_params
            if p.default is inspect.Parameter.empty and p.name != "tool_context"
        ]
        missing = [m for m in mandatory if m not in args_copy]
        if missing:
            return (
                None,
                f"Missing mandatory parameters for {self.name}: {', '.join(missing)}",
            )

        # build call kwargs
        call_kwargs = {}
        for param in filtered_params:
            if param.name == "tool_context":
                continue
            if param.name in args_copy:
                call_kwargs[param.name] = args_copy[param.name]

        # handle **kwargs
        if var_kwargs_param is not None:
            call_kwargs[var_kwargs_param.name] = args_copy.get(
                var_kwargs_param.name, {}
            )

        # inject tool_context if needed
        if "tool_context" in sig.parameters:
            call_kwargs["tool_context"] = tool_context

        return call_kwargs, None

    async def _maybe_await(self, result):
        """Helper: await result if it is a coroutine."""
        if asyncio.iscoroutine(result):
            return await result
        return result

    def resolve_refs(self, schema: dict) -> dict:
        """Recursively expand $ref in Pydantic schema."""
        defs = schema.get("$defs", {})

        def _expand(obj):
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_name = obj["$ref"].split("/")[-1]
                    return _expand(defs.get(ref_name, {}))
                return {k: _expand(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_expand(v) for v in obj]
            else:
                return obj

        return _expand(schema)
