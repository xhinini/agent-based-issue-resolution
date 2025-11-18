import os
import json
from typing import Any, List, Dict, Optional, Type
from pydantic import BaseModel
from openai import OpenAI

# Configuration for OpenRouter
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

# Expose model slugs (overridable via env)
MODEL_GPT4O_MINI = os.environ.get("MODEL_GPT4O_MINI", "openai/gpt-4o-mini-2024-07-18")
MODEL_GPT5_MINI = os.environ.get("MODEL_GPT5_MINI", "openai/gpt-5-mini")
MODEL_GPT4O = os.environ.get("MODEL_GPT4O", "openai/gpt-4o")

# Create a single client used throughout the project
client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


def responses_create(
    *,
    model: str,
    input: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> Any:
    """Create a completion using the Responses API with fallback to Chat Completions.
    Returns an object with .output_text populated when falling back.
    """
    try:
        kwargs = {"model": model, "input": input}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        return client.responses.create(**kwargs)
    except Exception as e:
        # Retry without temperature if unsupported
        if temperature is not None and "Unsupported parameter" in str(e) and "temperature" in str(e):
            try:
                kwargs = {"model": model, "input": input}
                if max_output_tokens is not None:
                    kwargs["max_output_tokens"] = max_output_tokens
                return client.responses.create(**kwargs)
            except Exception:
                pass
        # Fallback: map to chat.completions
        try:
            cc_kwargs = {"model": model, "messages": input}
            if temperature is not None:
                cc_kwargs["temperature"] = temperature
            if max_output_tokens is not None:
                cc_kwargs["max_tokens"] = max_output_tokens
            resp = client.chat.completions.create(**cc_kwargs)
            class _Compat:
                output_text: str
            o = _Compat()
            o.output_text = resp.choices[0].message.content if resp.choices else ""
            return o
        except Exception as ee:
            raise ee


def responses_parse(
    *,
    model: str,
    input: List[Dict[str, str]],
    text_format: Type[BaseModel],
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> Any:
    """Parse structured outputs using Responses API, with robust fallback that parses JSON/text.
    Returns an object with .output_parsed populated with an instance of text_format.
    """
    try:
        kwargs = {"model": model, "input": input, "text_format": text_format}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        return client.responses.parse(**kwargs)
    except Exception as e:
        # Retry without temperature if unsupported
        if temperature is not None and "Unsupported parameter" in str(e) and "temperature" in str(e):
            try:
                kwargs = {"model": model, "input": input, "text_format": text_format}
                if max_output_tokens is not None:
                    kwargs["max_output_tokens"] = max_output_tokens
                return client.responses.parse(**kwargs)
            except Exception:
                pass
        # Fallback: call responses_create, then coerce into the schema
        resp = responses_create(
            model=model,
            input=input,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        output_text = getattr(resp, "output_text", "") or ""
        data: Any
        try:
            data = json.loads(output_text)
        except Exception:
            data = output_text
        try:
            if isinstance(data, dict):
                parsed = text_format.model_validate(data)
            else:
                # Heuristic: if the schema has a single field, map the text to it
                fields = list(text_format.model_fields.keys())
                if len(fields) == 1:
                    parsed = text_format.model_validate({fields[0]: data})
                else:
                    # Give up with a minimal mapping
                    parsed = text_format.model_validate({fields[0]: str(data)})
            class _Compat:
                output_parsed: Any
            o = _Compat()
            o.output_parsed = parsed
            return o
        except Exception as pe:
            raise pe
