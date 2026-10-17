"""A real graph must route a correction through its own bound memory tools."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from octop_harness import HarnessAgent, HarnessAgentConfig, ModelConfig, ProviderConfig


class _ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.asyncio
async def test_agent_can_correct_its_fact_without_a_provider_call(tmp_path: Path) -> None:
    model = _ToolModel(responses=[AIMessage(content="ready")])
    cfg = HarnessAgentConfig(
        name="correction-graph",
        workspace_dir=tmp_path,
        backend={"type": "filesystem", "root_dir": str(tmp_path), "virtual_mode": True},
        providers=[
            ProviderConfig(
                id="test", base_url="https://example.test", api_key="unused", models=[ModelConfig(id="model")]
            )
        ],
        default_model="test/model",
        memory_backend={"type": "sqlite", "db_path": str(tmp_path / "memory.sqlite")},
        memory_enabled=True,
        memory_aux_model_enabled=False,
        memory_capture_enabled=False,
        memory_extract_on_session_end=False,
        session_log_enabled=False,
        checkpointer=False,
        pii_enabled=False,
        model_retry_enabled=False,
    )
    with patch("octop_harness.llm.factory.build_chat_model", return_value=model):
        agent = HarnessAgent(cfg)
    try:
        old, _, _ = agent.memory.create_atom("GPU memory inaccessible", entity_name="GPU")
        model.responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "memory_update",
                        "id": "correct",
                        "args": {
                            "path": f"atom/{old.id}.md",
                            "content": "GPU memory accessible since 2026-10-17",
                            "expected_content": old.assertion,
                            "reason": "verified read/write/readback",
                        },
                    }
                ],
            ),
            AIMessage(content="corrected"),
        ]
        with patch("octop_harness.middleware.model_router.ModelRouterMiddleware._select_model", return_value=model):
            result = await agent.graph.ainvoke(
                {"messages": [HumanMessage("Correct the obsolete GPU memory fact")]},
                {"configurable": {"thread_id": "correction"}},
            )
        receipts = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(receipts) == 1 and receipts[0].status != "error"
        receipt = json.loads(str(receipts[0].content))
        assert receipt["status"] == "replaced"
        assert agent.memory.get_atom(old.id).deprecated_at is not None
        assert agent.memory.get_atom(receipt["path"][5:-3]).assertion == "GPU memory accessible since 2026-10-17"
    finally:
        agent.close()
