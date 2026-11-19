"""
agent_app.py

文件职责：
创建带自动长期记忆功能的通用 Agent。

系统包含：

1. DeepSeek 模型
2. LangGraph SqliteStore
3. LangGraph SqliteSaver
4. MemoryManager
5. AutomaticMemoryMiddleware

运行效果：
用户正常对话，Agent 自动提取并更新长期记忆。
"""

import logging

from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from langgraph.store.sqlite import SqliteStore
from langgraph.checkpoint.sqlite import SqliteSaver

from memory_prompt import memory_context_prompt
from config import (
    LONG_TERM_DB_PATH,
    CHECKPOINT_DB_PATH
)

from memory_manager import MemoryManager
from memory_middleware import AutomaticMemoryMiddleware

# =====================================================
# 第一部分：日志配置
# =====================================================

# 自动记忆处理过程中可能出现模型调用异常、
# 结构化输出失败、数据库操作异常等问题。
#
# 使用日志记录异常，便于排查问题。

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s "
        "%(message)s"
    )
)


# =====================================================
# 第二部分：定义 Agent 运行上下文
# =====================================================

@dataclass
class UserContext:
    """
    当前 Agent 调用的用户上下文。

    user_id：
        当前登录用户的唯一标识。

    为什么不直接把 user_id 放在 messages 里面？

    因为 user_id 属于运行时身份信息，
    不是用户对话内容。

    将其放入 context，
    可以让 Middleware 和 Agent 节点安全地获取当前用户身份。
    """

    user_id: str


# =====================================================
# 第三部分：创建 DeepSeek 模型
# =====================================================

def create_llm():
    """
    创建 DeepSeek 模型。

    这里使用 ChatOpenAI，
    因为 DeepSeek 提供兼容 OpenAI 的 API 接口。
    """
    api_key = 'xxx'
    llm = ChatOpenAI(
        model="deepseek-flash",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0.2,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        }
    )

    return llm


# =====================================================
# 第四部分：创建 Agent
# =====================================================

def create_memory_agent(
        llm,
        store,
        checkpointer
):
    """
    创建具备长期记忆读写能力的 Agent。
    """

    # 1. 创建记忆管理器。
    memory_manager = MemoryManager(llm)

    # 2. 创建自动记忆保存 Middleware。
    memory_write_middleware = AutomaticMemoryMiddleware(
        memory_manager=memory_manager
    )

    # 3. 创建 Agent。
    agent = create_agent(

        model=llm,

        tools=[],

        system_prompt="""
你是一个通用 AI 助手。
请根据用户的问题提供准确、清晰的帮助。
""",

        # 注册两个 Middleware。
        middleware=[
            memory_context_prompt,
            memory_write_middleware
        ],

        # 短期记忆。
        checkpointer=checkpointer,

        # 长期记忆。
        store=store,

        # 用户身份上下文。
        context_schema=UserContext
    )

    return agent


# =====================================================
# 第五部分：打印用户的当前长期记忆
# =====================================================

def print_user_memories(store, user_id: str):
    """
    查看某个用户当前保存的长期记忆。

    这里只用于教学调试。
    实际应用中可以设计一个管理后台查看记忆。
    """

    namespace = (
        "users",
        user_id,
        "memories"
    )

    items = store.search(
        namespace,
        limit=200
    )

    print("\n")
    print("=" * 60)
    print(f"用户 {user_id} 的长期记忆")
    print("=" * 60)

    current_count = 0
    history_count = 0

    for item in items:

        if item.key.startswith("__history__:"):
            history_count += 1
            continue

        current_count += 1

        print("\nKey：", item.key)

        print(
            "类别：",
            item.value.get("memory_type")
        )

        print(
            "内容：",
            item.value.get("content")
        )

        print(
            "更新时间：",
            item.value.get("updated_at")
        )

    print("\n当前记忆数量：", current_count)
    print("历史版本数量：", history_count)


# =====================================================
# 第六部分：执行一次 Agent 对话
# =====================================================

def run_chat(
        agent,
        user_id: str,
        thread_id: str,
        user_text: str
):
    """
    执行一次 Agent 对话。

    user_id：
        决定访问哪个用户的长期记忆空间。

    thread_id：
        决定恢复哪个会话的短期状态。

    两者必须分开管理。
    """

    print("\n")
    print("=" * 60)
    print("用户输入：")
    print(user_text)
    print("=" * 60)

    # thread_id 放在 configurable 中。
    # 用于 Checkpointer 恢复对应会话。

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # user_id 放在运行时 context 中。
    # 用于 Middleware 确定长期记忆属于哪个用户。

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_text
                }
            ]
        },
        config=config,
        context=UserContext(
            user_id=user_id
        )
    )

    # 输出 Agent 最后一条 AI 消息。
    # Middleware 的 after_agent 已在执行完成阶段触发。

    for message in reversed(result["messages"]):

        if message.type == "ai":
            print("\nAI 助手：")
            print(message.content)
            break


# =====================================================
# 第七部分：程序入口
# =====================================================

def main():
    print("=" * 60)
    print("Agent 自动长期记忆验证")
    print("=" * 60)

    print(
        "\n长期记忆数据库：",
        LONG_TERM_DB_PATH
    )

    print(
        "短期记忆数据库：",
        CHECKPOINT_DB_PATH
    )

    # -------------------------------------------------
    # 1. 创建 DeepSeek 模型
    # -------------------------------------------------

    llm = create_llm()

    # -------------------------------------------------
    # 2. 创建长期记忆 SQLite Store
    # -------------------------------------------------

    # SqliteStore 是 LangGraph 提供的 SQLite Store。
    # 它负责持久化跨会话的长期记忆。

    with SqliteStore.from_conn_string(
            str(LONG_TERM_DB_PATH)
    ) as store:
        # 初始化 Store 数据表和必要结构。
        # 第一次运行时创建，后续启动会检查并迁移。

        store.setup()

        # -------------------------------------------------
        # 3. 创建短期记忆 SQLite Checkpointer
        # -------------------------------------------------

        # SqliteSaver 保存每个 thread_id 对应的会话状态。
        # 注意使用不同的数据库文件。

        with SqliteSaver.from_conn_string(
                str(CHECKPOINT_DB_PATH)
        ) as checkpointer:
            # -------------------------------------------------
            # 4. 创建带自动记忆能力的 Agent
            # -------------------------------------------------

            agent = create_memory_agent(
                llm=llm,
                store=store,
                checkpointer=checkpointer
            )

            # -------------------------------------------------
            # 5. 开始测试
            # -------------------------------------------------

            user_id = "user_001"

            # 测试一：
            # 第一次对话，提取用户长期偏好。

            run_chat(
                agent=agent,
                user_id=user_id,
                thread_id="session_001",
                user_text=(
                    "我正在学习大模型应用开发。"
                    "以后讲解的时候希望循序渐进，"
                    "最好结合真实业务场景，"
                    "代码也要有详细注释。"
                )
            )

            # 测试二：
            # 换一个新的 thread_id。
            # 同一个 user_id 仍然使用相同的长期记忆空间。
            #
            # 这次的表达与第一次相似，
            # 用来测试系统是否能够识别重复和补充。

            run_chat(
                agent=agent,
                user_id=user_id,
                thread_id="session_002",
                user_text=(
                    "以后回答尽量详细一些，"
                    "最好按步骤解释，"
                    "代码要有注释。"
                )
            )

            # 测试三：
            # 明确更新用户的 Python 技能水平。
            #
            # 先假设数据库中已有 python_skill_level：
            # Python 初级。
            #
            # 如果没有这条旧记忆，
            # 系统会将其作为新增信息处理。

            run_chat(
                agent=agent,
                user_id=user_id,
                thread_id="session_003",
                user_text=(
                    "我现在已经能够独立使用 Python "
                    "开发常见项目了，不用再按照零基础的方式讲解。"
                )
            )
            print('隔离性测试-------不同用户之间不能串记忆')
            run_chat(
                agent=agent,
                user_id="user_002",
                thread_id="memory_test_003",
                user_text=(
                    "请给我讲解一下 RAG 的工作原理。"
                )
            )

            # -------------------------------------------------
            # 6. 查看最终长期记忆
            # -------------------------------------------------

            print_user_memories(
                store,
                user_id
            )


if __name__ == "__main__":
    main()
