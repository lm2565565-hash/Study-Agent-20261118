# client--discover----calll---result
# langchain - mcp - adapters  将mcp工具适配为 langchain tool, 可以直接给create_agent使用
import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model='deepseek-flash',

    base_url='https://api.deepseek.com',
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}},
)


async def main():
    # 创建mcp客户端的配置
    client = MultiServerMCPClient(
        {
            'yuyue': {
                'transport': "http",
                'url': "http://127.0.0.1:8000/mcp"
            }
        }
    )
    # 从mcp server 获取工具  将mcp tool  转换为 langchain tool
    tools = await client.get_tools()
    # print('\n 发现的 langchain tools')
    # for tool in tools:
    #     print(tool.name, tool.description)
        # if tool.name == 'search_available_room':
        #     result = await tool.ainvoke(
        #         {
        #             'booking_date': '2026-12-18',
        #             'min_capacity': 15
        #         }
        #     )
        #     print('\n 会议场地查询结果:')
        #     print(result)
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt='你是一个校园预约智能助手, 你的工作是帮助学生查询校园活动和会议室信息'
    )

    question = input('input data：')
    result = await agent.ainvoke({
        'messages': [
            {"role": "user",
             "content": question}
        ]
    },
        config={"recursion_limit": 10}, )

    print('\n Agent 执行过程')
    for message in result['messages']:
        message_type = message.type
        if message_type == 'ai' and getattr(message, 'tool_calls', None):
            for tool_call in message.tool_calls:
                print('\n Agent 决策')
                print('选择工具：', tool_call['name'])
                print('调用参数：', tool_call['args'])
        elif message_type == 'tool':
            print('\n 工具返回')
            print('\n 工具名称：', message.name)
            print('\n 执行结果：', message.content)

    print('Agent:', result['messages'][-1].content)


if __name__ == '__main__':
    asyncio.run(main())
