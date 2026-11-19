'''
裁剪：保留最近内容 以便能够正常的控制上下文的大小
'''
from langchain.agents import create_agent
from langchain_core.messages import trim_messages  # 每次调用模型之前自动处理
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import wrap_model_call, wrap_tool_call

api_key = 'sk-5ee5344602d14afcbf9ea7bc68e93233'  # os.getenv("DEEPSEEK_API_KEY")

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


# 用户发送消息----agent 读取完整的state ------before_model-----对本次模型请求的messages进行裁剪------模型只接受裁剪后的消息-----生成回答
@wrap_model_call
def trim_history_middleware(request, handler):
    '''
    定义消息裁剪中间件，在每次模型调用之前执行
    :param request: request.messages agent 当前准备发送给模型的消息
    :param handler: 使用新的请求继续机型模型调用
    :return:
    '''
    original_messages = request.messages
    print('\n' + '=' * 50)
    print('消息裁剪中间件开始执行')
    print('=' * 50)
    print(f'裁剪前消息数量:{len(original_messages)}')
    trim_message = trim_messages(
        original_messages,
        max_tokens=100,
        token_counter="approximate",
        strategy="last",  # 优先保留最新的消息
        start_on='human',  # 从用户的消息开始
        include_system=True,  # 保留系统消息
        allow_partial=False
    )
    print(f'裁剪后消息数量:{len(trim_message)}')
    print('即将发送给模型的消息：')
    for index, msg in enumerate(trim_message, start=1):
        print(f"{index}. {msg.__class__.__name__}: {msg.content[:100]}")
        pass

    # 创建新的模型请求
    new_request = request.override(messages=trim_message)
    # 将新的请求交给模型执行
    return handler(new_request)
    pass


def main_agent():
    agent = create_agent(
        model=model,
        system_prompt='''
            你是企业的报销政策客服。
            公司的报销制度：
            1.北京的住宿标准每晚500元.
            2.餐费每天补助150元.
            3.交通费用需要提供凭证。
            请结合当前对话的历史信息回答问题。
            如果计算超标金额所需的信息不完整，需要主动询问用户，不要编造数据
            ''',
        middleware=[trim_history_middleware],  # 注册消息裁剪中间件
    )

    config = {
        'configurable': {
            "thread_id": "p00234"
        }
    }
    history = []  # 手动维护当前的会话历史
    while True:
        usinput = input('input msg:').strip()
        if usinput == 'quit':
            break
        if not usinput:
            continue
        history.append({"role": "user", "content": usinput})
        result = agent.invoke({"messages": history}, config=config)
        history = result['messages']  # 更新历史消息
        last_msg = history[-1]  # 最后一条ai消息
        print('AI:', last_msg.content)
        print(f'\n 当前完整历史消息数量：{len(history)}')
    pass


if __name__ == '__main__':
    main_agent()