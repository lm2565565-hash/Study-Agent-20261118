
"""
memory_middleware.py

只负责 Agent 执行结束后的长期记忆自动保存。

读取长期记忆已经交给 memory_prompt.py 中的
dynamic_prompt 完成。
"""

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import StateT
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from langgraph.typing import ContextT

from memory_manager import MemoryManager


logger = logging.getLogger(__name__)


class AutomaticMemoryMiddleware(AgentMiddleware):
    """
    Agent 完成任务后自动提取和更新长期记忆。
    """

    def __init__(self, memory_manager: MemoryManager):

        self.memory_manager = memory_manager

    @staticmethod
    def get_latest_user_text(messages) -> str:
        """
        只获取最近一条用户输入。
        不把 AI 回答或工具结果当成用户事实。
        """

        for message in reversed(messages):

            if not isinstance(message, HumanMessage):
                continue

            content = message.content

            if isinstance(content, list):

                text_parts = []

                for block in content:

                    if isinstance(block, dict):

                        if block.get("type") == "text":
                            text_parts.append(
                                block.get("text", "")
                            )

                content = "\n".join(text_parts)

            if isinstance(content, str):
                return content.strip()

            return str(content)

        return ""


    def after_agent(self, state, runtime):
        """
        Agent 完成执行后自动处理长期记忆。
        """

        print(
            "\n[MemoryMiddleware] "
            "Agent 执行完成，开始更新长期记忆。"
        )

        try:

            # 当前用户 ID。
            user_id = runtime.context.user_id

            # 长期记忆 Store。
            store = runtime.store

            if store is None:
                return None

            # 当前消息。
            messages = state.get(
                "messages",
                []
            )

            # 只分析用户本轮明确表达的信息。
            user_text = self.get_latest_user_text(
                messages
            )

            if not user_text:
                return None

            # 调用上一阶段的记忆管理器。
            results = self.memory_manager.process_conversation(
                store=store,
                user_id=user_id,
                conversation_text=f"用户：{user_text}"
            )

            for result in results:

                print(
                    "[MemoryMiddleware] ",
                    result
                )

        except Exception:

            # 记忆系统异常不能破坏已经完成的正常回答。
            logger.exception(
                "自动长期记忆保存失败。"
            )

        return None
