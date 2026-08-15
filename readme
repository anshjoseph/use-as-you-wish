use-as-you-wish
================

A LangChain-compatible chat model wrapper around the Anthropic Messages API,
used to drive Claude with a Claude-Code-style system prompt and OAuth access
token.

Install
-------

    pip install -e .

Usage
-----

    from use_as_you_wish import AnthropicLLM, ClaudeModel
    from langchain_core.messages import HumanMessage

    llm = AnthropicLLM(
        access_token="<your-anthropic-access-token>",
        model=ClaudeModel.HAIKU_4_5,
        max_tokens=256,
    )

    result = llm.invoke([HumanMessage(content="Say hello in one short sentence.")])
    print(result.content)

See examples/test_anthropic_llm.py for a runnable example (reads the token
from the ANTHROPIC_ACCESS_TOKEN environment variable).

Project layout
--------------

    src/use_as_you_wish/
        __init__.py    exports AnthropicLLM, ClaudeModel
        anthropic.py   AnthropicLLM implementation and ClaudeModel enum
