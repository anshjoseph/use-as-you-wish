import requests
from typing import List, Dict, Any, Optional, Union
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import ConfigDict, Field, PrivateAttr, field_validator
from enum import Enum


class ClaudeModel(Enum):
    HAIKU_3_5 = "claude-3-5-haiku-20241022"
    HAIKU_4_5 = "claude-haiku-4-5-20251001"
    SONNET_3_5 = "claude-3-5-sonnet-20241022"
    SONNET_3_7 = "claude-3-7-sonnet-20250219"
    SONNET_4_0 = "claude-sonnet-4-20250514"
    SONNET_4_5 = "claude-sonnet-4-5-20250929"
    SONNET_4_6 = "claude-sonnet-4-6"
    OPUS_4_0 = "claude-opus-4-20250514"
    OPUS_4_1 = "claude-opus-4-1-20250805"
    OPUS_4_5 = "claude-opus-4-5-20251101"
    OPUS_4_6 = "claude-opus-4-6"




class AnthropicLLM(BaseChatModel):
    """Anthropic LLM wrapper that maintains the Claude Code specific configuration."""
    
    access_token: str = Field(description="Anthropic API access token")
    model: Union[ClaudeModel, str] = Field(default=ClaudeModel.SONNET_4_5, description="Model to use")
    max_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, description="Temperature for generation")
    thinking_budget: Optional[int] = Field(default=None, description="Budget for thinking tokens")
    system_prompt: str = Field(
        default="You are Claude Code, Anthropic's official CLI for Claude.",
        description="System prompt (must be exactly this for Claude Code compatibility)"
    )
    api_url: str = Field(
        default="https://api.anthropic.com/v1/messages",
        description="Anthropic API URL"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("model", mode="before")
    @classmethod
    def _normalize_model(cls, value: Union[ClaudeModel, str]) -> str:
        return value.value if isinstance(value, ClaudeModel) else value

    _bound_tools: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _tool_choice: Optional[Any] = PrivateAttr(default=None)
    
    @property
    def _llm_type(self) -> str:
        return "anthropic-claude-code"
    
    def _convert_messages_to_anthropic(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages to Anthropic format."""
        anthropic_messages = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                continue
            elif isinstance(message, HumanMessage):
                anthropic_messages.append({
                    "role": "user",
                    "content": message.content
                })
            elif isinstance(message, AIMessage):
                # Handle tool calls in AI messages
                content = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tool_call.get("id", f"call_{hash(tool_call['name'])}"),
                            "name": tool_call["name"],
                            "input": tool_call["args"]
                        })
                
                anthropic_messages.append({
                    "role": "assistant",
                    "content": content if content else message.content
                })
            elif isinstance(message, ToolMessage):
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content
                    }]
                })
        
        return anthropic_messages
    
    def _convert_anthropic_to_langchain(self, response: Dict[str, Any]) -> ChatResult:
        """Convert Anthropic response to LangChain ChatResult."""
        content = response.get("content", [])
        
        # Extract text content
        text_content = ""
        tool_calls = []
        
        for block in content:
            if block.get("type") == "text":
                text_content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "name": block.get("name"),
                    "args": block.get("input", {}),
                    "id": block.get("id")
                })
        
        message = AIMessage(
            content=text_content,
            additional_kwargs={
                "tool_calls": tool_calls,
                "stop_reason": response.get("stop_reason"),
                "usage": response.get("usage", {})
            }
        )
        
        # Add tool calls as attribute for LangChain compatibility
        if tool_calls:
            message.tool_calls = [
                {
                    "name": tc["name"],
                    "args": tc["args"],
                    "id": tc["id"],
                    "type": "tool_call"
                }
                for tc in tool_calls
            ]
        
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={
                "stop_reason": response.get("stop_reason"),
                "usage": response.get("usage", {})
            }
        )
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """Generate a response from Anthropic."""
        
        # Convert messages to Anthropic format
        anthropic_messages = self._convert_messages_to_anthropic(messages)

        # Anthropic takes the system prompt as a top-level field, not a message
        system_prompt = next(
            (m.content for m in messages if isinstance(m, SystemMessage)),
            self.system_prompt,
        )

        # Prepare the request
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "oauth-2025-04-20,claude-code-20250219",
            "content-type": "application/json",
        }
        
        # Extract tools from kwargs or use bound tools
        tools = kwargs.get("tools", [])
        if not tools and self._bound_tools:
            tools = self._bound_tools
        tools = self._convert_tools_to_anthropic(tools)
        
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_prompt,
            "messages": anthropic_messages,
        }
        
        # Add thinking budget if provided
        if self.thinking_budget:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget
            }
        
        if tools:
            body["tools"] = tools
        
        if stop:
            body["stop_sequences"] = stop
        
        # Make the API call
        response = requests.post(
            self.api_url,
            headers=headers,
            json=body,
            timeout=60
        )
        if not response.ok:
            raise requests.exceptions.HTTPError(
                f"{response.status_code} {response.reason} for url: {response.url}\n{response.text}",
                response=response,
            )

        return self._convert_anthropic_to_langchain(response.json())
    
    @staticmethod
    def _convert_tools_to_anthropic(tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert a list of tools (dicts or BaseTool objects) to Anthropic's tool format."""
        tool_list = []
        for tool in tools:
            if isinstance(tool, dict):
                tool_list.append(tool)
            elif hasattr(tool, 'name') and hasattr(tool, 'description') and hasattr(tool, 'args_schema'):
                tool_list.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.args_schema.model_json_schema() if tool.args_schema else {"type": "object", "properties": {}}
                })
        return tool_list

    def bind_tools(
        self,
        tools: Union[List[Dict[str, Any]], Any],
        **kwargs: Any,
    ) -> "AnthropicLLM":
        """Bind tools to the model."""
        tools = self._convert_tools_to_anthropic(tools) if tools else []

        # Create a new instance with bound tools
        new_instance = self.model_copy(deep=True)
        new_instance._bound_tools = tools
        
        # Handle tool_choice if provided
        if "tool_choice" in kwargs:
            # Store tool_choice for potential future use
            new_instance._tool_choice = kwargs["tool_choice"]
        
        return new_instance

