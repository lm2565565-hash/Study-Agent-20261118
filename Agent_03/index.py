# 手动保存当前的对话状态
# history=[]
from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
# history.append(HumanMessages('''fsadgasdgasdgasdg'''))
# invoke(history)
# history=newdata
# Agent 自动状态管理
# Checkpointer
# state 保存agent运行状态
# checkpointer 保存和读取状态快照
# thread_id    对话的编号  独立的 不重复
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import trim_messages
# 创建快照对象
# checkpointer = InMemorySaver()
api_key = 'sk-5ee5344602d14afcbf9ea7bc68e93233'  # os.getenv("DEEPSEEK_API_KEY")

# 配置裁剪对象

trimmed_msg=trim_messages(api_key)

# if not api_key:
#     raise ValueError(
#         "请先配置环境变量 DEEPSEEK_API_KEY"
#     )


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
with SqliteSaver.from_conn_string('agent_memory.db') as checkpointer:
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt='''
        你是企业的报销政策客服。
        公司的报销制度：
        1.北京的住宿标准每晚500元.
        2.餐费每天补助150元.
        3.交通费用需要提供凭证。
        请结合当前对话的历史信息回答问题。
        如果计算超标金额所需的信息不完整，需要主动询问用户，不要编造数据
        ''',
        checkpointer=checkpointer  # agent 具有了持久化的能力
    )

    config = {
        'configurable': {
            "thread_id": "p001"
        }
    }
    while True:
        msg=input('input data')
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": msg}
            ]
        }, config=config)
        answer = result['messages'][-1].content
        print('input:', msg)
        print("AI", answer)

def chat(msg: str):
    '''
    连续对话
    :param msg:
    :return:
    '''
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": msg}
        ]
    }, config=config)
    answer = result['messages'][-1].content
    print('input:', msg)
    print("AI", answer)
    pass


if __name__ == '__main__':
    # chat('我是员工崔世界，这次去北京出差，住宿花了12000元')
    # chat('公司北京住宿标准是多少?')
    # chat('那我超了多少')
    # chat('如果我住3晚呢?')
    # #获取agent状态
    # snapshot=agent.get_state(config)
    # state_value=snapshot.values
    # message=state_value.get('messages')
    # print('状态数据...')
    # print(message)
    # print('state data')
    # print(state_value)


    # 消息才将
    pass

