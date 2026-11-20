import asyncio

from mcp.client import Client

mcp_server_url = "http://127.0.0.1:8000/mcp"


async def main():
    async with Client(mcp_server_url) as client:
        print('mcp server 连接成功...')
        tools_results = await client.list_tools()
        print('发现可提供的工具')
        for tool in tools_results.tools:
            print(tool.name)
            print(tool.description)
            print(tool.input_schema)
            print('==============调用mcp工具====================')
            if tool.name == 'search_available_room':
                result = await client.call_tool(  #调用指定的mcp工具  返回值 CallToolResult
                    name=tool.name,
                    arguments={
                        'booking_date': '2026-12-18',
                        'min_capacity': 15
                    }
                )
                for item in result.content:
                    print('\n会议室查询结果:' + item.text)
        pass


# async def test():

if __name__ == '__main__':
    asyncio.run(main())
