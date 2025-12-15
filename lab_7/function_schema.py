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
    """Оптимизированная детекция стиля docstring."""
    # Быстрая проверка: если строка пустая или короткая
    if not doc or len(doc) < 10:
        return "google"

    # Быстрый поиск по префиксам
    if doc.startswith(":param") or doc.startswith(":type") or ":return:" in doc[:100]:
        return "sphinx"

    # Поиск шаблонов с помощью предкомпилированных regex
    for pattern in _SPHINX_PATTERNS:
        if pattern.search(doc):
            return "sphinx"

    for pattern in _NUMPY_PATTERNS:
        if pattern.search(doc):
            return "numpy"

    for pattern in _GOOGLE_PATTERNS:
        if pattern.search(doc):
            return "google"

    return "google"


@contextlib.contextmanager
def _suppress_griffe_logging():
    """Контекстный менеджер для подавления логов griffe."""
    logger = logging.getLogger("griffe")
    previous_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


# Кэшированные версии часто используемых функций
@lru_cache(maxsize=256)  # Увеличили размер кэша
def _cached_signature(func: Callable[..., Any]) -> inspect.Signature:
    """Кэшированная версия inspect.signature."""
    return inspect.signature(func, eval_str=True)  # Добавили eval_str для лучшей поддержки


@lru_cache(maxsize=256)
def _cached_get_type_hints(func: Callable[..., Any]) -> dict[str, Any]:
    """Кэшированная версия get_type_hints."""
    try:
        return get_type_hints(func, include_extras=True)
    except (NameError, TypeError):
        # Если есть проблемы с аннотациями, возвращаем пустой dict
        return {}


@lru_cache(maxsize=256)
def _cached_getdoc(func: Callable[..., Any]) -> str | None:
    """Кэшированная версия inspect.getdoc."""
    return inspect.getdoc(func)


@lru_cache(maxsize=256)
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
    # Быстрая проверка: если это не Annotated, возвращаем как есть
    origin = get_origin(annotation)
    if origin is not Annotated:
        return annotation, ()

    args = get_args(annotation)
    if not args:
        return annotation, ()

    # Извлекаем основную аннотацию и метаданные
    main_annotation = args[0]
    metadata = args[1:]

    # Обрабатываем вложенные Annotated
    while get_origin(main_annotation) is Annotated:
        nested_args = get_args(main_annotation)
        if not nested_args:
            break
        main_annotation = nested_args[0]
        metadata = (*metadata, *nested_args[1:])

    return main_annotation, metadata


def _extract_description_from_metadata(metadata: tuple[Any, ...]) -> str | None:
    """Extracts a human readable description from Annotated metadata if present."""
    if not metadata:
        return None

    # Проверяем первый элемент
    if isinstance(metadata[0], str):
        return metadata[0]

    # Ищем строку среди остальных элементов
    for item in metadata[1:]:
        if isinstance(item, str):
            return item
        # Также проверяем Field объекты
        if isinstance(item, FieldInfo) and item.description:
            return item.description

    return None


# Кэш для созданных моделей Pydantic с LRU политикой
_MODEL_CACHE: dict[str, tuple[type[BaseModel], dict[str, Any]]] = {}
_MAX_CACHE_SIZE = 512  # Максимальный размер кэша


def _get_cache_key(
        func_name: str,
        strict_json_schema: bool,
        use_docstring_info: bool,
        docstring_style: str | None
) -> str:
    """Создает ключ для кэша моделей."""
    style_str = docstring_style or "auto"
    return f"{func_name}|{strict_json_schema}|{use_docstring_info}|{style_str}"


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

    # Оптимизированная обработка аннотаций
    for name, annotation in type_hints_with_extras.items():
        if name == "return":
            continue

        stripped_ann, metadata = _strip_annotated(annotation)
        type_hints[name] = stripped_ann

        # Быстрая проверка на наличие описания в метаданных
        if metadata:
            description = _extract_description_from_metadata(metadata)
            if description is not None:
                annotated_param_descs[name] = description

    # Быстрое обновление словаря описаний параметров
    if annotated_param_descs:
        param_descs.update(annotated_param_descs)

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

    # Проверяем, есть ли уже модель в кэше
    cache_key = _get_cache_key(func_name, strict_json_schema, use_docstring_info, docstring_style)

    if cache_key in _MODEL_CACHE:
        dynamic_model, cached_json_schema = _MODEL_CACHE[cache_key]
        json_schema = cached_json_schema
    else:
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

        # Кэшируем модель и схему
        _MODEL_CACHE[cache_key] = (dynamic_model, json_schema)

        # Очищаем кэш если он слишком большой (LRU политика)
        if len(_MODEL_CACHE) > _MAX_CACHE_SIZE:
            # Удаляем первый элемент (самый старый в Python 3.7+)
            first_key = next(iter(_MODEL_CACHE))
            del _MODEL_CACHE[first_key]

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


# Добавляем функцию для очистки кэша (полезно для тестирования)
def clear_caches():
    """Очищает все внутренние кэши."""
    _cached_signature.cache_clear()
    _cached_get_type_hints.cache_clear()
    _cached_getdoc.cache_clear()
    generate_func_documentation.cache_clear()
    _MODEL_CACHE.clear()


# Добавляем функцию для получения статистики кэша
def get_cache_stats() -> dict[str, Any]:
    """Возвращает статистику использования кэшей."""
    return {
        "signature_cache_size": _cached_signature.cache_info().currsize,
        "type_hints_cache_size": _cached_get_type_hints.cache_info().currsize,
        "getdoc_cache_size": _cached_getdoc.cache_info().currsize,
        "doc_cache_size": generate_func_documentation.cache_info().currsize,
        "model_cache_size": len(_MODEL_CACHE),
        "max_model_cache_size": _MAX_CACHE_SIZE,
    }