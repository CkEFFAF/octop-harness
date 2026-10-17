"""A real nested graph call must not leak peer tokens into its caller's answer."""

from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.graph import END, START, MessagesState, StateGraph

from octop_harness.agent import HarnessAgent
from octop_harness.protocols.langgraph import LangGraphProtocol


@pytest.mark.asyncio
async def test_nested_peer_call_keeps_only_parent_tokens():
    peer_model = FakeListChatModel(responses=["CHILD_SECRET"])
    host_model = FakeListChatModel(responses=["HOST_FINAL"])

    async def child_node(state):
        return {"messages": [await peer_model.ainvoke(state["messages"])]}

    child = StateGraph(MessagesState)
    child.add_node("model", child_node)
    child.add_edge(START, "model")
    child.add_edge("model", END)
    child_protocol = LangGraphProtocol(graph=child.compile())
    facade = SimpleNamespace(_get_protocol=lambda _: child_protocol)
    tool_results = []

    async def peer_tool(state):
        messages, config, protocol = HarnessAgent._prepare_call(facade, "Read", None)
        tool_results.append(await protocol.call(messages, config))
        return {}

    async def host_node(state):
        return {"messages": [await host_model.ainvoke(state["messages"])]}

    parent = StateGraph(MessagesState)
    parent.add_node("tools", peer_tool)
    parent.add_node("model", host_node)
    parent.add_edge(START, "tools")
    parent.add_edge("tools", "model")
    parent.add_edge("model", END)
    protocol = LangGraphProtocol(graph=parent.compile())
    parent_facade = SimpleNamespace(_get_protocol=lambda _: protocol)
    messages, config, _ = HarnessAgent._prepare_call(parent_facade, "Question", None)
    events = [event async for event in protocol.stream(messages, config)]
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert text == "HOST_FINAL"
    assert tool_results[0]["messages"][-1].content == "CHILD_SECRET"


def test_invocation_identity_is_fresh_and_does_not_mutate_request_metadata():
    facade = SimpleNamespace(_get_protocol=lambda _: None)
    metadata = {"trace": "kept", "harness_invocation_id": "parent"}
    request = {"messages": "Hi", "metadata": metadata}
    _, first, _ = HarnessAgent._prepare_call(facade, request, None)
    _, second, _ = HarnessAgent._prepare_call(facade, request, None)
    assert first["metadata"]["harness_invocation_id"] != second["metadata"]["harness_invocation_id"]
    assert first["metadata"]["trace"] == "kept"
    assert metadata["harness_invocation_id"] == "parent"
