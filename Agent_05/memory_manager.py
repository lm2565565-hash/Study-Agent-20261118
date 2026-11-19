"""
memory_manager.py

文件职责：
通用 Agent 长期记忆自动管理核心。

主要功能：
1. 自动提取候选记忆。
2. 判断记忆是否值得长期保存。
3. 读取用户已有记忆。
4. 识别重复信息。
5. 识别补充信息。
6. 识别明确冲突和更新。
7. 更新时保留旧版本。
8. 使用 LangGraph SqliteStore 持久化。

核心设计原则：
LLM 负责理解和提出建议。
Python 负责校验和执行数据库操作。
"""

import re
import logging

from datetime import datetime, timezone

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from pydantic import BaseModel, Field
from typing import Literal

logger = logging.getLogger(__name__)


# =====================================================
# 第一部分：定义候选记忆的数据结构
# =====================================================

class MemoryCandidate(BaseModel):
    memory_key: str = Field(
        description="标准化记忆键，使用英文 snake_case"
    )

    memory_type: Literal[
        "profile",
        "preference",
        "goal",
        "skill",
        "project",
        "constraint",
        "behavior",
        "other"
    ]

    content: str

    confidence: float = Field(
        ge=0,
        le=1
    )

    is_stable: bool

    is_explicit: bool

    is_sensitive: bool

    evidence: str


class MemoryExtraction(BaseModel):
    memories: list[MemoryCandidate] = Field(
        default_factory=list
    )


# =====================================================
# 第二部分：定义记忆处理决策
# =====================================================

class MemoryDecision(BaseModel):
    action: Literal[
        "add",
        "update",
        "merge",
        "skip"
    ]

    target_key: str | None = None

    final_content: str = ""

    confidence: float = Field(
        ge=0,
        le=1
    )

    reason: str


# =====================================================
# 第三部分：定义 MemoryManager
# =====================================================

class MemoryManager:
    """
    通用长期记忆管理器。

    extractor_llm：
        负责从对话中提取候选记忆。

    decision_llm：
        负责比较新旧记忆并提出处理建议。

    store：
        不在构造函数中创建。
        由外部传入，避免 MemoryManager 自己管理数据库连接。
    """

    def __init__(self, llm):

        # 使用同一个基础模型，分别构建两个结构化输出链。
        #
        # function_calling 模式可以通过工具调用格式
        # 约束模型输出结构，适合兼容 OpenAI API 的模型。
        #
        # 如果模型服务不支持 function calling，
        # 需要根据服务商能力调整 structured output 方法。
        # pydantic  对象、dict
        self.extractor = llm.with_structured_output(
            MemoryExtraction,
            method="function_calling"
        )

        self.decision_model = llm.with_structured_output(
            MemoryDecision,
            method="function_calling"
        )

    # =================================================
    # 4.1 构建用户 namespace
    # =================================================

    @staticmethod
    def build_namespace(user_id: str) -> tuple:
        """
        为每个用户构建独立的长期记忆空间。

        例如：
        ("users", "user_001", "memories")

        不同用户不能共享同一个用户 namespace。
        """

        if not user_id or not user_id.strip():
            raise ValueError("user_id 不能为空")

        return (
            "users",
            user_id.strip(),
            "memories"
        )

    # =================================================
    # 4.2 判断记忆 Key 是否符合规范
    # =================================================

    @staticmethod
    def normalize_key(memory_key: str) -> str:
        """
        将模型生成的 Key 转换为统一格式。

        例如：
            AnswerPreference
            answer-preference
            answer preference

        最终统一为：
            answer_preference

        这样可以降低 Key 漂移造成的重复记忆。
        """

        key = memory_key.strip().lower()

        # 将空格和连字符转换成下划线。
        key = re.sub(r"[\s\-]+", "_", key)

        # 删除不符合要求的字符。
        key = re.sub(r"[^a-z0-9_]", "", key)

        # 删除连续下划线。
        key = re.sub(r"_+", "_", key)

        # 删除首尾下划线。
        key = key.strip("_")

        # 限制 Key 长度，避免异常数据。
        if not re.fullmatch(
                r"[a-z][a-z0-9_]{1,63}",
                key
        ):
            raise ValueError(
                f"非法 memory_key：{memory_key}"
            )

        return key

    # =================================================
    # 4.3 敏感信息规则检查
    # =================================================

    @staticmethod
    def contains_sensitive_data(text: str) -> bool:
        """
        使用简单规则拦截明显的敏感信息。

        不能只依赖 LLM 判断敏感信息。
        因此增加 Python 层的确定性检查。

        当前拦截：
        1. 邮箱地址。
        2. 中国大陆手机号。
        3. 中国大陆身份证号码。
        4. 明显的密码、密钥、Token 表达。

        注意：
        这只是基础防护，不代表覆盖所有敏感信息。
        """

        patterns = [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            r"\b1[3-9]\d{9}\b",
            r"\b\d{17}[\dXx]\b",
        ]

        for pattern in patterns:

            if re.search(pattern, text):
                return True

        # 对明显的凭据关键词进行检查。
        sensitive_words = [
            "密码是",
            "我的密码",
            "银行卡号",
            "信用卡号",
            "身份证号",
            "api_key",
            "api key",
            "access_token",
            "secret_key",
            "private_key"
        ]

        lower_text = text.lower()

        for word in sensitive_words:

            if word.lower() in lower_text:
                return True

        return False

    # =================================================
    # 4.4 从对话中提取候选记忆
    # =================================================

    def extract_candidates(
            self,
            conversation_text: str
    ) -> list[MemoryCandidate]:
        """
        使用 LLM 从对话中提取候选长期记忆。

        这里不直接写入 Store。

        提取规则：
        1. 只提取有长期价值的信息。
        2. 不提取临时计划。
        3. 不提取假设和猜测。
        4. 不把一次性任务误认为长期目标。
        5. 不保存敏感个人信息。
        6. 同一主题尽量使用稳定的 Key。
        """

        system_prompt = """
你是一个通用 AI 助手的长期记忆提取模块。

你的工作不是总结对话，而是发现值得长期保存的用户信息。

允许的记忆类别：

1. profile：用户背景
2. preference：回答和使用偏好
3. goal：长期目标
4. skill：技能水平
5. project：长期项目背景
6. constraint：稳定约束
7. behavior：稳定行为习惯
8. other：其他长期有价值信息

提取原则：

一、只提取长期有价值的信息。
例如：
- 用户长期从事 Python 开发。
- 用户喜欢详细、循序渐进的解释。
- 用户正在长期学习 Agent 开发。

二、不要提取临时信息。
例如：
- 今天要去开会。
- 下周去上海。
- 这次帮我写一份报告。

三、不要把假设当成事实。
例如：
- 如果我以后转行做开发……
- 假设我有一家公司……

四、不要提取模型自己推测出来的信息。
必须有用户明确表达的证据。

五、不要保存敏感个人信息。
例如：
- 密码、API Key、Token。
- 银行卡号、身份证号。
- 精确住址。
- 医疗诊断和其他高度敏感个人信息。

六、Key 必须稳定。
例如：
“喜欢详细解释”“回答尽量详细”“希望按步骤讲”
应尽量归一化为 answer_style。

“Python 初级”“Python 基础较弱”
应尽量归一化为 python_skill_level。

七、如果没有值得保存的信息，返回空列表。

请严格依据对话内容，不要编造用户信息。
"""

        result = self.extractor.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=conversation_text)
            ]
        )

        return result.memories

    # =================================================
    # 4.5 获取用户现有的长期记忆
    # =================================================

    def get_existing_memories(
            self,
            store,
            user_id: str
    ) -> list[dict]:
        """
        查询指定用户的已有长期记忆。

        这里使用 Store.search() 查询 namespace。
        不传 query，表示列出 namespace 下的记录，
        并不是进行向量语义检索。

        当前项目规模较小时，直接加载全部当前记忆，
        交给 LLM 判断是否重复或冲突。

        历史版本以 __history__: 开头，
        不参与当前记忆的去重判断。
        """

        namespace = self.build_namespace(user_id)

        items = store.search(
            namespace,
            limit=200
        )

        memories = []

        for item in items:

            # 历史版本不参与当前记忆匹配。
            if item.key.startswith("__history__:"):
                continue

            memories.append({
                "memory_key": item.key,
                "memory_type": item.value.get(
                    "memory_type",
                    "other"
                ),
                "content": item.value.get(
                    "content",
                    ""
                ),
                "updated_at": item.value.get(
                    "updated_at",
                    ""
                )
            })

        return memories

    # =================================================
    # 4.6 使用 LLM 判断新旧记忆之间的关系
    # =================================================

    def decide_memory_action(
            self,
            candidate: MemoryCandidate,
            existing_memories: list[dict]
    ) -> MemoryDecision:
        """
        比较一条候选记忆和已有记忆。

        LLM 只能提出以下四种建议：

        add：
            新增一条全新的记忆。

        update：
            用户明确改变了旧事实。

        merge：
            新信息是对旧记忆的补充。

        skip：
            信息重复、模糊、临时或没有保存价值。
        """

        system_prompt = """
你是一个长期记忆治理模块。

你的任务是比较一条候选记忆和用户已有的长期记忆，
判断应该新增、更新、合并还是跳过。

必须遵守以下规则：

1. add：
   新信息与已有记忆没有重复或冲突。
   使用新的标准化 Key。

2. skip：
   新信息与已有记忆完全相同。
   或者信息模糊、不稳定、价值不足。

3. merge：
   新信息是对已有记忆的补充。
   应保留旧信息中仍然有效的内容，
   并形成一条完整、简洁的新记忆。

4. update：
   用户明确表达了新的事实，
   并且它与已有记忆发生了冲突。
   例如：
   旧记忆：Python 初级。
   新信息：我现在已经有多年 Python 开发经验。
   应更新原有技能记忆。

5. 不要错误更新：
   旧记忆：办公城市是南京。
   新信息：下周我要去上海出差。
   这不是办公城市发生变化，应该 skip。

6. 假设、计划、临时安排不能覆盖稳定事实。

7. 如果不确定新旧信息是否属于同一主题，
   不要强行覆盖，优先 skip。

8. update 和 merge 必须指定已有记忆的 target_key。

9. add 不应覆盖已有 Key。

10. final_content 必须是一条准确、简洁的长期记忆。
"""

        human_prompt = f"""
候选记忆：

{candidate.model_dump_json(indent=2)}

用户已有记忆：

{existing_memories}

请判断应该执行什么操作。
"""

        result = self.decision_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
        )

        return result

    # =================================================
    # 4.7 保存更新前的旧版本
    # =================================================

    @staticmethod
    def save_history(
            store,
            namespace: tuple,
            memory_key: str,
            old_value: dict
    ) -> None:
        """
        在更新当前记忆前，保存旧版本。

        例如：
        当前 skill_level：
            Python 初级

        更新为：
            Python 熟练

        更新前先保存一份历史记录，
        避免旧信息直接丢失。

        注意：
        历史版本与当前版本使用不同 Key。
        """

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%S%fZ")

        history_key = (
            f"__history__:{memory_key}:{timestamp}"
        )

        history_value = {
            "original_key": memory_key,
            "memory_type": old_value.get(
                "memory_type",
                "other"
            ),
            "content": old_value.get(
                "content",
                ""
            ),
            "archived_at": datetime.now(
                timezone.utc
            ).isoformat()
        }

        store.put(
            namespace,
            history_key,
            history_value
        )

    # =================================================
    # 4.8 执行最终写入
    # =================================================

    def apply_decision(
            self,
            store,
            user_id: str,
            candidate: MemoryCandidate,
            decision: MemoryDecision
    ) -> str:
        """
        根据 LLM 的建议执行实际数据库操作。

        关键：
        不能完全相信模型的决策。

        必须经过 Python 规则校验：
        1. 置信度是否足够。
        2. Key 是否合法。
        3. update 的目标是否存在。
        4. add 是否会覆盖已有 Key。
        5. 是否属于敏感信息。
        """

        # -------------------------------------------------
        # 1. 基础安全检查
        # -------------------------------------------------

        if decision.confidence < 0.75:
            return "skip：决策置信度不足"

        if not decision.final_content.strip():
            return "skip：最终内容为空"

        if self.contains_sensitive_data(
                decision.final_content
        ):
            return "skip：最终内容包含敏感信息"

        if self.contains_sensitive_data(
                candidate.content
        ):
            return "skip：候选记忆包含敏感信息"

        # -------------------------------------------------
        # 2. 准备用户 namespace
        # -------------------------------------------------

        namespace = self.build_namespace(user_id)

        # -------------------------------------------------
        # 3. 读取目标 Key
        # -------------------------------------------------

        if decision.action in ("update", "merge"):

            if not decision.target_key:
                return "skip：更新操作没有指定 target_key"

            target_key = self.normalize_key(
                decision.target_key
            )

            old_item = store.get(
                namespace,
                target_key
            )

            # 如果模型说要更新，但目标不存在，
            # 不能盲目写入一个不确定的记录。
            if old_item is None:
                return "skip：更新目标不存在"

            old_value = old_item.value

            # 如果内容没有任何变化，
            # 不必重复更新数据库。
            if old_value.get("content") == decision.final_content:
                return "skip：内容没有变化"

            # 更新前保存历史版本。
            self.save_history(
                store,
                namespace,
                target_key,
                old_value
            )

            # 保存新版本。
            store.put(
                namespace,
                target_key,
                {
                    "memory_type": candidate.memory_type,
                    "content": decision.final_content,
                    "updated_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "evidence": candidate.evidence,
                    "confidence": candidate.confidence
                }
            )

            if decision.action == "update":
                return f"update：更新了 {target_key}"

            return f"merge：合并更新了 {target_key}"

        # -------------------------------------------------
        # 4. 新增记忆
        # -------------------------------------------------

        if decision.action == "add":

            new_key = self.normalize_key(
                candidate.memory_key
            )

            # 不允许 add 覆盖已经存在的记忆。
            existing_item = store.get(
                namespace,
                new_key
            )

            if existing_item is not None:
                return (
                    f"skip：Key {new_key} 已存在，"
                    "不允许直接覆盖"
                )

            store.put(
                namespace,
                new_key,
                {
                    "memory_type": candidate.memory_type,
                    "content": decision.final_content,
                    "created_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "updated_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "evidence": candidate.evidence,
                    "confidence": candidate.confidence
                }
            )

            return f"add：新增了 {new_key}"

        # -------------------------------------------------
        # 5. 跳过
        # -------------------------------------------------

        return f"skip：{decision.reason}"

    # =================================================
    # 4.9 对一轮对话执行完整自动记忆流程
    # =================================================

    def process_conversation(
            self,
            store,
            user_id: str,
            conversation_text: str
    ) -> list[str]:
        """
        完整的自动记忆处理入口。

        执行流程：

        1. 提取候选记忆。
        2. 判断候选信息是否值得保存。
        3. 读取用户已有记忆。
        4. 判断新旧信息之间的关系。
        5. 执行新增、更新、合并或跳过。
        6. 返回处理结果。
        """

        print("\n========== 自动记忆处理开始 ==========")

        # -------------------------------------------------
        # 第一步：提取候选记忆
        # -------------------------------------------------

        candidates = self.extract_candidates(
            conversation_text
        )

        print(
            f"发现 {len(candidates)} 条候选记忆"
        )

        results = []

        # -------------------------------------------------
        # 第二步：逐条处理候选记忆
        # -------------------------------------------------

        for candidate in candidates:

            print("\n候选记忆：")
            print(candidate.model_dump())

            # -------------------------------------------------
            # 2.1 判断是否符合长期记忆保存条件
            # -------------------------------------------------

            if not candidate.is_stable:
                results.append(
                    "skip：不是稳定的长期信息"
                )
                continue

            if not candidate.is_explicit:
                results.append(
                    "skip：用户没有明确表达"
                )
                continue

            if candidate.is_sensitive:
                results.append(
                    "skip：模型识别为敏感信息"
                )
                continue

            if candidate.confidence < 0.75:
                results.append(
                    "skip：提取置信度不足"
                )
                continue

            if self.contains_sensitive_data(
                    candidate.content
            ):
                results.append(
                    "skip：规则检测到敏感信息"
                )
                continue

            # -------------------------------------------------
            # 2.2 读取用户当前的长期记忆
            # -------------------------------------------------

            existing_memories = self.get_existing_memories(
                store,
                user_id
            )

            # -------------------------------------------------
            # 2.3 让 LLM 判断如何处理
            # -------------------------------------------------

            decision = self.decide_memory_action(
                candidate,
                existing_memories
            )

            print("\n记忆处理建议：")
            print(decision.model_dump())

            # -------------------------------------------------
            # 2.4 Python 最终校验并执行
            # -------------------------------------------------

            result = self.apply_decision(
                store,
                user_id,
                candidate,
                decision
            )

            print("最终执行结果：", result)

            results.append(result)

        print("\n========== 自动记忆处理结束 ==========")

        return results
