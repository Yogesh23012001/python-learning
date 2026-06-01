"""Single-tool invocation against Gemini.

Hour 1: simplest possible loop. Send a prompt + tool schema, get back either
a text response OR a tool call. If tool call, execute it, send the result
back, get the final answer. Hard-coded to one round-trip for clarity.

Hour 3 will generalize this to an iterative agent loop.
"""

from __future__ import annotations

from typing import Any

import structlog
from api.config import get_settings  # ✅ correct
from google import genai
from google.genai import types

from agent.tools import TOOL_SCHEMAS, execute_tool

logger = structlog.get_logger(__name__)


async def run_single_tool_call(user_prompt: str) -> str:
    """Run one tool-use round-trip against Gemini.

    Returns the LLM's final text response. Logs every step.
    """
    settings = get_settings()
    if settings.gemini_api_key is None:
        raise RuntimeError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    model_name = "gemini-2.5-flash"

    # Convert our tool schemas into Gemini's tool format. The SDK accepts a
    # list of Tool | Callable | dict (invariant), so widen the local type.
    gemini_tools: list[types.Tool | Any] = [
        types.Tool(
            function_declarations=[types.FunctionDeclaration(**schema) for schema in TOOL_SCHEMAS]
        )
    ]

    logger.info("agent_call_started", prompt=user_prompt[:200])

    # STEP 1: send prompt + tools
    response = await client.aio.models.generate_content(
        model=model_name,
        contents=[user_prompt],
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    # STEP 2: did the LLM ask to call a tool?
    if not response.candidates:
        raise RuntimeError("model returned no candidates")
    candidate = response.candidates[0]
    if candidate.content is None:
        raise RuntimeError("candidate had no content")
    parts = candidate.content.parts or []

    function_calls = [part.function_call for part in parts if part.function_call is not None]

    if not function_calls:
        # No tool requested — LLM answered directly
        final_text = response.text or ""
        logger.info("agent_completed_no_tools", response_length=len(final_text))
        return final_text

    # STEP 3: execute the requested tool(s)
    tool_responses: list[types.Part] = []
    for fc in function_calls:
        if fc.name is None:
            raise RuntimeError("function call had no name")
        name: str = fc.name
        args = dict(fc.args) if fc.args else {}
        logger.info("llm_requested_tool", tool=name, args=args)

        try:
            result = await execute_tool(name, args)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}

        tool_responses.append(types.Part.from_function_response(name=name, response=result))

    # STEP 4: send results back, get final answer
    follow_up = await client.aio.models.generate_content(
        model=model_name,
        contents=[
            user_prompt,
            candidate.content,
            types.Content(role="user", parts=tool_responses),
        ],
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    final_text = follow_up.text or ""
    logger.info(
        "agent_completed_with_tools",
        tools_called=[fc.name for fc in function_calls],
        response_length=len(final_text),
    )
    return final_text
