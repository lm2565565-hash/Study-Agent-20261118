from dataclasses import dataclass
from typing import Literal

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()  # 负责跨会话保存员工重要信息
checkpointer = InMemorySaver()  # 负责保存同一个thread_id内的对话状态


@dataclass
class UserContext:
    '''
    保存本次运行的用户身份信息
    employee_id、name 由应用程序传入，不让模型自己决定当前操作哪个员工的记忆
    '''
    employee_id: str
    name: str
    pass


# employee_id = 'p001'
# namespace = ('users', employee_id, 'memories')
# store.put(namespace, 'profile', {'name': '小王', 'city': '北京', 'description': '负责区域城市'})
# store.put(namespace, 'business_interest',
#           {'interest': '报销政策', 'description': '安徽省科技等公司就看到过还是卡的韩国'})

# 读取
# profile=store.get(namespace,'profile')
# print(profile)
# memories = store.search(namespace, limit=1)
# print(memories)
#
# store.put(namespace, 'profile', {'name': '李当', 'city': '上海', 'description': '负责区域城市'})
# memories = store.search(namespace, limit=1)
# print(memories)
# 让agent自动的操作长期记忆  需要一个新的对象 ToolRuntime

from langchain.tools import tool, ToolRuntime  # 可以对store进行直接操作


# 理解业务
# 用户: 我目前常驻上海，我比较关注的是出差的补助报销政策，回答具体费用金额希望你展示计算给过程。
# agent分析用户意图---调用 save_memory--ToolRuntime 获取store对象----store.put----success
#
# 用户：我之前常驻的城市是哪里，我比较关注什么
@tool
def save_employee_memory(
        memory_type: Literal['profile', 'business_interest', 'answer_preference'],
        memory_content: str,
        runtime: ToolRuntime
):
    '''
    定义保存长期记忆的工具
    :return:
    '''
    # 从应用程序传入的 context 中获取员工编号
    employee_id = runtime.context.employee_id
    # 为当前员工构造独立的记忆空间
    namespace = ('users', employee_id, 'memories')
    runtime.store.put(namespace, memory_type, {"memory_type": memory_type, "content": memory_content})
    print(f'\n 【长期记忆】已保存员工 {employee_id} 的{memory_type}')
    return (
        f"{memory_content}"
    )
    pass


@tool
def get_employee_memory(runtime: ToolRuntime):
    '''
    查询当前员工已经 保存的长期记忆，当员工询问以前提供过的个人资料、业务关注点或回答偏好时使用
    :param runtime:
    :return:
    '''
    employee_id = runtime.context.employee_id
    namespace = ('users', employee_id, 'memories')
    memories = runtime.store.search(namespace, limit=20)
    if not memories:
        return '目前没有找到该员工的长期记忆。'
    result = []
    for memory in memories:
        memory_type = memory.value.get('memory_type', 'unknown')
        content = memory.value.get('content', '')
        result.append(f'记忆类型:{memory_type};内容{content}')
    print('长期 记忆读取完成')
    return '\n'.join(result)
    pass


api_key = 'sk-5ee5344602d1443afcbf9ea7bc68e93233'  # os.getenv("DEEPSEEK_API_KEY")

model = ChatOpenAI(
    model="deepseek-flash",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0,
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)

agent = create_agent(
    model=model,
    tools=[save_employee_memory, get_employee_memory],
    store=store,  # 注入长期记忆
    context_schema=UserContext,  # 注入运行时的context类型
    checkpointer=checkpointer,
    system_prompt="""
     你是企业的报销政策客服。
    你可以使用长期记忆工具保存和读取员工的重要信息。
    记忆管理规则：
    1.只有员工明确要求记住的信息，或者明确表达的稳定个人偏好，才可以保存
    2.当员工询问自已以前提供过的信息时，应先调用get_employee_memory查询。
    3.保存记忆时，选择正确的 memory_type;
     profile:员工基本资料
     business_interest:长期关注的业务；
     answer_perference:回答偏好
    4.不要将临时费用、单次报销申请 自动当作长期个人记忆
    5.如果没有找到记忆，应明确说明，不要自行编造。
    业务规则：
    1.北京和上海的住宿标准每晚500元,其他城市是300元。
    2.餐费每天补助150元.
    3.交通费用需要提供凭证。
    4.不确定的信息不能编造
    5.如果历史摘要中存在信息，可以结合最近的对话回答。
    6.如果缺少必要的信息，应主动询问员工
    请结合当前对话的历史信息回答问题。
    如果计算超标金额所需的信息不完整，需要主动询问用户，不要编造数据
    """
)


def chat(em_id, thread_id, user_input):
    config = {
        'configurable': {'thread_id': thread_id}
    }
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": user_input,
            }
        ]
    }, config=config,
        context=UserContext(employee_id=em_id, name='李当')
    )
    return result['messages'][-1].content
    pass

if __name__ == '__main__':
    employee_id = 'p001'
    print('=============会话A===================')
    answer=chat(employee_id,"session-A",user_input=(
        "请记住，我目前在深圳，我比较关注住宿的报销政策，希望你展示计算过程，请分别保存这些长期信息"
    ))
    print('AI:',answer)

    print('=============会话B===================')
    answer = chat(employee_id, "session-B", user_input=(
        "我之前告诉过你，我常驻哪个城市，我比较关注什么业务，回答问题有什么偏好"
    ))
    print('AI:', answer)

    print('\n================用户的隔离测试========================')
    employee_id = 'p003'
    print('=============会话A===================')
    answer = chat(employee_id, "session-C", user_input=(
        "你记得我常驻哪个城市吗"
    ))
    print('AI:', answer)
