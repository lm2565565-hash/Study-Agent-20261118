# from zipapp import create_archive
#
#
# @tool
# def query_available_rooms(date:str,start_time:str,end_time:str):
#     '''
#     查询指定的时间会议室
#     :param date:
#     :param start_time:
#     :param end_time:
#     :return:
#     '''
#     pass
#
# agent=create_agent(
#     tools=[
#         query_available_rooms,
#         create_archive,
#         canel_booking
#     ]
# )
#
# 校园预约系统 ：查询会议室、创建预约、取消预约
# 校园活动系统：查询活动、活动报名情况，查询会议室
# 校园设备系统：机房、投影仪、路由器。查询会议室
# 校园通知系统：通知发放、发布、查询，查询会议室
#
# mcp: 是一种让AI应用以统一方式发现和调用 外部工具、资源和提示模板的开放协议，它主要解决是AI应用和外部能力之间标准化的连接问题。
#
# mcp server : 让mcp服务如何暴露工具
# mcp client :让客户端如何发现工具、获取工具的描述。传递参数、接收结果
# agent 调用mcp工具：只是工具的来源发生了变化