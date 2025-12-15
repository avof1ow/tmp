from __future__ import annotations

import contextlib
import inspect
import logging
import re
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Literal, get_args, get_origin, get_type_hints
from functools import lru_cache

from griffe import Docstring, DocstringSectionKind
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from .exceptions import UserError
from .run_context import RunContextWrapper
from .strict_schema import ensure_strict_json_schema
from .tool_context import ToolContext


@dataclass
class FuncSchema:
    """
    Captures the schema for a python function, in preparation for sending it to an LLM as a tool.
    """

    name: str
    """The name of the function."""
    description: str | None
    """The description of the function."""
    params_pydantic_model: type[BaseModel]
    """A Pydantic model that represents the function's parameters."""
    params_json_schema: dict[str, Any]
    """The JSON schema for the function's parameters, derived from the Pydantic model."""
    signature: inspect.Signature
    """The signature of the function."""
    takes_context: bool = False
    """Whether the function takes a RunContextWrapper argument (must be the first argument)."""
    strict_json_schema: bool = True
    """Whether the JSON schema is in strict mode. We **strongly** recommend setting this to True,
    as it increases the likelihood of correct JSON input."""

    def to_call_args(self, data: BaseModel) -> tuple[list[Any], dict[str, Any]]:
        """
        Converts validated data from the Pydantic model into (args, kwargs), suitable for calling
        the original function.
        """
        positional_args: list[Any] = []
        keyword_args: dict[str, Any] = {}
        seen_var_positional = False

        # Use enumerate() so we can skip the first parameter if it's context.
        for idx, (name, param) in enumerate(self.signature.parameters.items()):
            # If the function takes a RunContextWrapper and this is the first parameter, skip it.
            if self.takes_context and idx == 0:
                continue

            value = getattr(data, name, None)
            if param.kind == param.VAR_POSITIONAL:
                # e.g. *args: extend positional args and mark that *args is now seen
                positional_args.extend(value or [])
                seen_var_positional = True
            elif param.kind == param.VAR_KEYWORD:
                # e.g. **kwargs handling
                keyword_args.update(value or {})
            elif param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                # Before *args, add to positional args. After *args, add to keyword args.
                if not seen_var_positional:
                    positional_args.append(value)
                else:
                    keyword_args[name] = value
            else:
                # For KEYWORD_ONLY parameters, always use keyword args.
                keyword_args[name] = value
        return positional_args, keyword_args


@dataclass
class FuncDocumentation:
    """Contains metadata about a Python function, extracted from its docstring."""

    name: str
    """The name of the function, via `__name__`."""
    description: str | None
    """The description of the function, derived from the docstring."""
    param_descriptions: dict[str, str] | None
    """The parameter descriptions of the function, derived from the docstring."""


DocstringStyle = Literal["google", "numpy", "sphinx"]

# Предкомпилированные регулярные выражения
_SPHINX_PATTERNS = [re.compile(r"^:param\s", re.MULTILINE),
                    re.compile(r"^:type\s", re.MULTILINE),
                    re.compile(r"^:return:", re.MULTILINE),
                    re.compile(r"^:rtype:", re.MULTILINE)]
_NUMPY_PATTERNS = [re.compile(r"^Parameters\s*\n\s*-{3,}", re.MULTILINE),
                   re.compile(r"^Returns\s*\n\s*-{3,}", re.MULTILINE),
                   re.compile(r"^Yields\s*\n\s*-{3,}", re.MULTILINE)]
_GOOGLE_PATTERNS = [re.compile(r"^(Args|Arguments):", re.MULTILINE),
                    re.compile(r"^(Returns):", re.MULTILINE),
                    re.compile(r"^(Raises):", re.MULTILINE)]


# As of Feb 2025, the automatic style detection in griffe is an Insiders feature. This
# code approximates it.
def _detect_docstring_style(doc: str) -> DocstringStyle:
    scores: dict[DocstringStyle, int] = {"sphinx": 0, "numpy": 0, "google": 0}

    # Sphinx style detection: look for :param, :type, :return:, and :rtype:
    for pattern in _SPHINX_PATTERNS:
        if pattern.search(doc):
            scores["sphinx"] += 1

    # Numpy style detection: look for headers like 'Parameters', 'Returns', or 'Yields' followed by
    # a dashed underline
    for pattern in _NUMPY_PATTERNS:
        if pattern.search(doc):
            scores["numpy"] += 1

    # Google style detection: look for section headers with a trailing colon
    for pattern in _GOOGLE_PATTERNS:
        if pattern.search(doc):
            scores["google"] += 1

    max_score = max(scores.values())
    if max_score == 0:
        return "google"

    # Priority order: sphinx > numpy > google in case of tie
    if scores["sphinx"] == max_score:
        return "sphinx"
    if scores["numpy"] == max_score:
        return "numpy"
    return "google"


@contextlib.contextmanager
def _suppress_griffe_logging():
    # Suppresses warnings about missing annotations for params
    logger = logging.getLogger("griffe")
    previous_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


# Кэшированные версии часто используемых функций
@lru_cache(maxsize=128)
def _cached_signature(func: Callable[..., Any]) -> inspect.Signature:
    """Кэшированная версия inspect.signature."""
    return inspect.signature(func)


@lru_cache(maxsize=128)
def _cached_get_type_hints(func: Callable[..., Any]) -> dict[str, Any]:
    """Кэшированная версия get_type_hints."""
    return get_type_hints(func, include_extras=True)


@lru_cache(maxsize=128)
def _cached_getdoc(func: Callable[..., Any]) -> str | None:
    """Кэшированная версия inspect.getdoc."""
    return inspect.getdoc(func)


@lru_cache(maxsize=128)
def generate_func_documentation(
    func: Callable[..., Any], style: DocstringStyle | None = None
) -> FuncDocumentation:
    """
    Extracts metadata from a function docstring, in preparation for sending it to an LLM as a tool.

    Args:
        func: The function to extract documentation from.
        style: The style of the docstring to use for parsing. If not provided, we will attempt to
            auto-detect the style.

    Returns:
        A FuncDocumentation object containing the function's name, description, and parameter
        descriptions.
    """
    doc = _cached_getdoc(func)
    if not doc:
        return FuncDocumentation(name=func.__name__, description=None, param_descriptions=None)

    with _suppress_griffe_logging():
        docstring = Docstring(doc, lineno=1, parser=style or _detect_docstring_style(doc))
        parsed = docstring.parse()

    description: str | None = next(
        (section.value for section in parsed if section.kind == DocstringSectionKind.text), None
    )

    param_descriptions: dict[str, str] = {
        param.name: param.description
        for section in parsed
        if section.kind == DocstringSectionKind.parameters
        for param in section.value
    }

    return FuncDocumentation(
        name=func.__name__,
        description=description,
        param_descriptions=param_descriptions or None,
    )


def _strip_annotated(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    """Returns the underlying annotation and any metadata from typing.Annotated."""

    metadata: tuple[Any, ...] = ()
    ann = annotation

    while get_origin(ann) is Annotated:
        args = get_args(ann)
        if not args:
            break
        ann = args[0]
        metadata = (*metadata, *args[1:])

    return ann, metadata


def _extract_description_from_metadata(metadata: tuple[Any, ...]) -> str | None:
    """Extracts a human readable description from Annotated metadata if present."""

    for item in metadata:
        if isinstance(item, str):
            return item
    return None


def function_schema(
    func: Callable[..., Any],
    docstring_style: DocstringStyle | None = None,
    name_override: str | None = None,
    description_override: str | None = None,
    use_docstring_info: bool = True,
    strict_json_schema: bool = True,
) -> FuncSchema:
    """
    Given a Python function, extracts a `FuncSchema` from it, capturing the name, description,
    parameter descriptions, and other metadata.

    Args:
        func: The function to extract the schema from.
        docstring_style: The style of the docstring to use for parsing. If not provided, we will
            attempt to auto-detect the style.
        name_override: If provided, use this name instead of the function's `__name__`.
        description_override: If provided, use this description instead of the one derived from the
            docstring.
        use_docstring_info: If True, uses the docstring to generate the description and parameter
            descriptions.
        strict_json_schema: Whether the JSON schema is in strict mode. If True, we'll ensure that
            the schema adheres to the "strict" standard the OpenAI API expects. We **strongly**
            recommend setting this to True, as it increases the likelihood of the LLM producing
            correct JSON input.

    Returns:
        A `FuncSchema` object containing the function's name, description, parameter descriptions,
        and other metadata.
    """

    # 1. Grab docstring info
    if use_docstring_info:
        doc_info = generate_func_documentation(func, docstring_style)
        param_descs = dict(doc_info.param_descriptions or {})
    else:
        doc_info = None
        param_descs = {}

    # Используем кэшированную версию get_type_hints
    type_hints_with_extras = _cached_get_type_hints(func)
    type_hints: dict[str, Any] = {}
    annotated_param_descs: dict[str, str] = {}

    for name, annotation in type_hints_with_extras.items():
        if name == "return":
            continue

        stripped_ann, metadata = _strip_annotated(annotation)
        type_hints[name] = stripped_ann

        description = _extract_description_from_metadata(metadata)
        if description is not None:
            annotated_param_descs[name] = description

    for name, description in annotated_param_descs.items():
        param_descs.setdefault(name, description)

    # Ensure name_override takes precedence even if docstring info is disabled.
    func_name = name_override or (doc_info.name if doc_info else func.__name__)

    # 2. Inspect function signature and get type hints
    # Используем кэшированную версию signature
    sig = _cached_signature(func)
    params = list(sig.parameters.items())
    takes_context = False
    filtered_params = []

    if params:
        first_name, first_param = params[0]
        # Prefer the evaluated type hint if available
        ann = type_hints.get(first_name, first_param.annotation)
        if ann != inspect._empty:
            origin = get_origin(ann) or ann
            if origin is RunContextWrapper or origin is ToolContext:
                takes_context = True  # Mark that the function takes context
            else:
                filtered_params.append((first_name, first_param))
        else:
            filtered_params.append((first_name, first_param))

    # For parameters other than the first, raise error if any use RunContextWrapper or ToolContext.
    for name, param in params[1:]:
        ann = type_hints.get(name, param.annotation)
        if ann != inspect._empty:
            origin = get_origin(ann) or ann
            if origin is RunContextWrapper or origin is ToolContext:
                raise UserError(
                    f"RunContextWrapper/ToolContext param found at non-first position in function"
                    f" {func.__name__}"
                )
        filtered_params.append((name, param))

    # We will collect field definitions for create_model as a dict:
    #   field_name -> (type_annotation, default_value_or_Field(...))
    fields: dict[str, Any] = {}

    for name, param in filtered_params:
        ann = type_hints.get(name, param.annotation)
        default = param.default

        # If there's no type hint, assume `Any`
        if ann == inspect._empty:
            ann = Any

        # If a docstring param description exists, use it
        field_description = param_descs.get(name, None)

        # Handle different parameter kinds
        if param.kind == param.VAR_POSITIONAL:
            # e.g. *args: extend positional args
            if get_origin(ann) is tuple:
                # e.g. def foo(*args: tuple[int, ...]) -> treat as List[int]
                args_of_tuple = get_args(ann)
                if len(args_of_tuple) == 2 and args_of_tuple[1] is Ellipsis:
                    ann = list[args_of_tuple[0]]  # type: ignore
                else:
                    ann = list[Any]
            else:
                # If user wrote *args: int, treat as List[int]
                ann = list[ann]  # type: ignore

            # Default factory to empty list
            fields[name] = (
                ann,
                Field(default_factory=list, description=field_description),
            )

        elif param.kind == param.VAR_KEYWORD:
            # **kwargs handling
            if get_origin(ann) is dict:
                # e.g. def foo(**kwargs: dict[str, int])
                dict_args = get_args(ann)
                if len(dict_args) == 2:
                    ann = dict[dict_args[0], dict_args[1]]  # type: ignore
                else:
                    ann = dict[str, Any]
            else:
                # e.g. def foo(**kwargs: int) -> Dict[str, int]
                ann = dict[str, ann]  # type: ignore

            fields[name] = (
                ann,
                Field(default_factory=dict, description=field_description),
            )

        else:
            # Normal parameter
            if default == inspect._empty:
                # Required field
                fields[name] = (
                    ann,
                    Field(..., description=field_description),
                )
            elif isinstance(default, FieldInfo):
                # Parameter with a default value that is a Field(...)
                fields[name] = (
                    ann,
                    FieldInfo.merge_field_infos(
                        default, description=field_description or default.description
                    ),
                )
            else:
                # Parameter with a default value
                fields[name] = (
                    ann,
                    Field(default=default, description=field_description),
                )

    # 3. Dynamically build a Pydantic model
    dynamic_model = create_model(f"{func_name}_args", __base__=BaseModel, **fields)

    # 4. Build JSON schema from that model
    json_schema = dynamic_model.model_json_schema()
    if strict_json_schema:
        json_schema = ensure_strict_json_schema(json_schema)

    # 5. Return as a FuncSchema dataclass
    return FuncSchema(
        name=func_name,
        # Ensure description_override takes precedence even if docstring info is disabled.
        description=description_override or (doc_info.description if doc_info else None),
        params_pydantic_model=dynamic_model,
        params_json_schema=json_schema,
        signature=sig,
        takes_context=takes_context,
        strict_json_schema=strict_json_schema,
    )