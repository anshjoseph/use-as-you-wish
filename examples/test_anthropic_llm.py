"""Example usage of AnthropicLLM with the ClaudeModel enum.

Requires an Anthropic access token:
    set ANTHROPIC_ACCESS_TOKEN=sk-ant-...   (Windows)
    export ANTHROPIC_ACCESS_TOKEN=sk-ant-... (bash)
"""
from use_as_you_wish import AnthropicLLM, ClaudeModel
from langchain_core.messages import HumanMessage


def main() -> None:
    access_token = "sk-ant-<your claude token>"

    llm = AnthropicLLM(
        access_token=access_token,
        model=ClaudeModel.HAIKU_4_5,
        max_tokens=256,
    )

    result = llm.invoke([HumanMessage(content="Say hello in one short sentence.")])
    print(result.content)


if __name__ == "__main__":
    main()
