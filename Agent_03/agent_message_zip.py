from langchain.agents import create_agent
from langchain_core.messages import trim_messages, HumanMessage, AIMessage, SystemMessage  # 每次调用模型之前自动处理
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
# 上下文管理的参数
summary_trigger_messages = 12
# 摘要之后 保留最近的多少条原始消息
keep_recent_messages = 6
max_token = 500


def summary_history(old_messages, old_summary):
    '''
    生成摘要的函数    较早的历史消息对话压缩为一段摘要
    :param old_messages:  本次准备压缩的旧消息
    :param old_summary:   之前已经生成的摘要
    :return: 摘要文本
    '''
    print('\n' + '=' * 50)
    print('开始生成历史摘要')
    print('=' * 50)
    histroy_text = []  # 将旧消息转换为文本
    for message in old_messages:
        if isinstance(message, HumanMessage):
            role = "员工"
        elif isinstance(message, AIMessage):
            role = "助手"
        else:
            continue
        histroy_text.append(f'{role}:{message.content}')
    histroy_text = '\n'.join(histroy_text)
    summary_prompt = f"""
    你是企业的报销业务信息整理助手，你的任务是根据历史对话生成一份准确、简洁的业务摘要。
    请你重点保留以下信息:
    
    一、员工基本信息
    1.员工编号
    2.常驻城市
    
    二、出差信息
    1.出差目的地
    2.出差日期
    3.住宿时间
        
    三、费用信息
    1.住宿金额
    2.交通金额
    3.餐饮金额
    4.其他费用
    
    四、报销政策
    1.已经明确的住宿标准
    2.已经明确的餐饮标准
    3.其他已经明确的政策
    
    五、当前业务进展
    1.已经确认的计算结果
    2.员工当前关注的问题
    3.尚未解决的问题
    
    六、信息的准确性要求
    1.不允许编造信息
    2.不允许把推测当作事实
    3.没有提供的信息标记为 "未提供"
    3.新信息与旧摘要冲突时，以员工最新明确确认的信息为准。
    
    之前的历史摘要：
    {old_summary if old_summary else '暂无历史摘要'}
    
    本次需要整理的历史对话：
    {histroy_text}
    请生成更新后的完整业务摘要。
    只输出摘要内容，不要输出其他的解释。    
    """
    response = model.invoke([
        SystemMessage(content="你负责准确整理企业报销对话中的业务信息"),
        HumanMessage(content=summary_prompt)
    ])
    # 获取摘要文本
    new_summary = response.content
    print('历史信息摘要生成完成....')
    print(new_summary)
    return new_summary


def update_summary_if_needed(history, summary):
    '''
    判断历史消息是否达到摘要的条件
    :param history:
    :param summary:
    :return:

    如果超过阈值
        找出较早的消息
        生成信息摘要
        保留最新的几条消息

    返回更新后的 histroy 和 summary
    '''
    message_count = len(history)
    print('当前历史消息的数量:', message_count)
    if message_count <= summary_trigger_messages:
        print('历史消息未达到摘要阈值，不执行摘要.')
        return history, summary
    print('历史消息已经超过摘要阈值，准备执行摘要...')
    # 找出需要压缩的旧消息
    old_messages = history[:-keep_recent_messages]
    # 找出需要保留的最近消息
    recent_messages = history[-keep_recent_messages:]

    print(f'准备摘要的旧消息数量:{len(old_messages)}')
    print(f'准备保留的近期消息数量:{len(recent_messages)}')

    # 删除已经被摘要覆盖的旧消息
    # 只在当前程序中保留摘要和最新消息
    summary = summary_history(old_messages, summary)
    history = recent_messages
    print('历史消息压缩完成')
    print('压缩后保留原始消息数量:', len(history))
    return history, summary
    pass


def build_model_context(history, summary):
    '''
    将系统提示词、历史摘要、近期消息组合起来
    也可以使用 trim_messages 控制token数量
    '''
    print('\n' + '=' * 50)
    print('开始构建本轮模型上下文')
    print('=' * 50)
    system_content = f"""
     你是企业的报销政策客服。
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
    if summary:
        system_content += f"""
以下是员工此前对话的重要业务摘要。
请将其作为历史参考看信息使用。
如果最近的对话对旧信息进行明确的修正，应优先采用最新确认的信息。
历史摘要：
{summary}

"""
    messages = [SystemMessage(content=system_content), *history]  # *history #最近的原始对话b
    print(f'裁剪前模型上下文的消息数量：{len(messages)}')

    trim_message = trim_messages(
        messages,
        max_tokens=max_token,
        token_counter="approximate",
        strategy="last",  # 优先保留最新的消息
        start_on='human',  # 从用户的消息开始
        include_system=True,  # 保留系统消息
        allow_partial=False
    )
    print(f'裁剪后消息数量:{len(trim_message)}')
    print('\n 本轮实际发送给模型的消息:')
    for index, msg in enumerate(trim_message, start=1):
        print(f"{index}. {msg.__class__.__name__}: {msg.content[:100]}")
        pass
    return trim_message

if __name__ == '__main__':
    history=[] #保存原始会话
    summary='' #保存压缩后的历史摘要
    while True:
        user_input=input('\n 员工:').strip()
        if user_input == 'exit':
            break
        if not user_input:
            continue

        history.append(HumanMessage(content=user_input))
        #判断是否需要摘要
        history, summary=update_summary_if_needed(history, summary)
        #构建上下文
        model_message=build_model_context(history, summary)
        print('正在调用模型......')
        response=model.invoke(model_message)
        answer=response.content
        history.append(AIMessage(content=answer))
        print('AI:',answer)
        print('本轮对话完成...')
    pass
