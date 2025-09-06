import pytest
from typing import Any, List

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, LLMMetadata
from llama_index.core.llms import MockLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow.workflow_events import AgentStream


class CountingLLM(MockLLM):
    calls: int = 0

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)

    async def achat_with_tools(
        self, chat_history: List[ChatMessage], tools: List[Any], **kwargs: Any
    ) -> ChatResponse:
        self.calls += 1
        if self.calls == 1:
            message = ChatMessage(
                role="assistant",
                content=None,
                additional_kwargs={
                    "tool_calls": [
                        ToolSelection(
                            tool_id="tool_1",
                            tool_name="direct_tool",
                            tool_kwargs={},
                        )
                    ]
                },
            )
        else:
            message = ChatMessage(role="assistant", content="final answer from llm")
        return ChatResponse(message=message, raw={})

    def get_tool_calls_from_response(
        self, response: ChatResponse, **kwargs: Any
    ) -> List[ToolSelection]:
        return response.message.additional_kwargs.get("tool_calls", [])


async def direct_tool() -> str:
    return "tool output"


@pytest.mark.asyncio
async def test_return_direct_triggers_final_llm_pass() -> None:
    llm = CountingLLM()
    agent = FunctionAgent(
        name="test_agent",
        description="test",
        tools=[FunctionTool.from_defaults(fn=direct_tool, return_direct=True)],
        llm=llm,
        streaming=False,
    )
    ctx = Context(agent)
    handler = agent.run("call the tool", ctx=ctx)
    async for _ in handler.stream_events():
        pass
    result = await handler
    assert result.response.content == "final answer from llm"
    assert llm.calls == 2


class CountingStreamingLLM(MockLLM):
    calls: int = 0

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)

    async def astream_chat_with_tools(
        self, chat_history: List[ChatMessage], tools: List[Any], **kwargs: Any
    ):
        self.calls += 1

        async def gen():
            if self.calls == 1:
                message = ChatMessage(
                    role="assistant",
                    content=None,
                    additional_kwargs={
                        "tool_calls": [
                            ToolSelection(
                                tool_id="tool_1",
                                tool_name="direct_tool",
                                tool_kwargs={},
                            )
                        ]
                    },
                )
                yield ChatResponse(message=message, raw={})
            else:
                yield ChatResponse(
                    message=ChatMessage(
                        role="assistant", content="final", additional_kwargs={}
                    ),
                    raw={},
                    delta="final",
                )
                yield ChatResponse(
                    message=ChatMessage(
                        role="assistant",
                        content="final answer from llm",
                        additional_kwargs={},
                    ),
                    raw={},
                    delta=" answer from llm",
                )

        return gen()

    def get_tool_calls_from_response(
        self, response: ChatResponse, **kwargs: Any
    ) -> List[ToolSelection]:
        return response.message.additional_kwargs.get("tool_calls", [])


@pytest.mark.asyncio
async def test_return_direct_streams_final_llm_pass() -> None:
    llm = CountingStreamingLLM()
    agent = FunctionAgent(
        name="test_agent",
        description="test",
        tools=[FunctionTool.from_defaults(fn=direct_tool, return_direct=True)],
        llm=llm,
        streaming=True,
    )
    ctx = Context(agent)
    handler = agent.run("call the tool", ctx=ctx)

    streamed = ""
    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            streamed += ev.delta or ""

    result = await handler
    assert result.response.content == "final answer from llm"
    assert llm.calls == 2
    assert streamed == "final answer from llm"
