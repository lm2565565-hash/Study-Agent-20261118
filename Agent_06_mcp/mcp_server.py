from mcp.server import MCPServer

# mcp 服务对象  它负责注册工具、资源、按照mcp协议对外提供服务
mcp = MCPServer('yuyueServer')

# 生成模拟数据
envents_data = [
    {
        "event_id": "E001",
        "event_name": "模拟答辩",
        "date": "2026-12-01",
        "start_time": "14:00",
        "end_time": "16:00",
        "location": "学术报告厅",
        "status": "通过"
    },
    {
        "event_id": "E002",
        "event_name": "项目中期汇报会",
        "date": "2026-12-03",
        "start_time": "09:00",
        "end_time": "11:30",
        "location": "第一会议室",
        "status": "待审核"
    },
    {
        "event_id": "E003",
        "event_name": "学术前沿讲座",
        "date": "2026-12-05",
        "start_time": "15:00",
        "end_time": "17:00",
        "location": "多功能讲堂",
        "status": "通过"
    },
    {
        "event_id": "E004",
        "event_name": "科研团队讨论会",
        "date": "2026-12-08",
        "start_time": "10:00",
        "end_time": "12:00",
        "location": "第二研讨室",
        "status": "驳回"
    },
    {
        "event_id": "E005",
        "event_name": "毕业论文培训会",
        "date": "2026-12-10",
        "start_time": "13:30",
        "end_time": "15:30",
        "location": "学术报告厅",
        "status": "待审核"
    }
]

rooms = [
    {
        "room_id": "R101",
        "room_name": "第一会议室",
        "capacity": 50,
        "location": "行政楼 101",
        "available_dates": ["2026-12-18", "2026-12-19", "2026-12-24"]
    },
    {
        "room_id": "R102",
        "room_name": "第二会议室",
        "capacity": 30,
        "location": "行政楼 102",
        "available_dates": ["2026-12-17", "2026-12-20", "2026-12-25"]
    },
    {
        "room_id": "R103",
        "room_name": "小型研讨室",
        "capacity": 15,
        "location": "行政楼 103",
        "available_dates": ["2026-12-16", "2026-12-18", "2026-12-22"]
    },
    {
        "room_id": "R201",
        "room_name": "学术报告厅",
        "capacity": 120,
        "location": "教学楼2层",
        "available_dates": ["2026-12-19", "2026-12-21", "2026-12-26"]
    },
    {
        "room_id": "R202",
        "room_name": "多功能讲堂",
        "capacity": 80,
        "location": "教学楼202",
        "available_dates": ["2026-12-18", "2026-12-20", "2026-12-23"]
    },
    {
        "room_id": "R301",
        "room_name": "小组讨论室A",
        "capacity": 10,
        "location": "图书馆301",
        "available_dates": ["2026-12-17", "2026-12-21", "2026-12-24"]
    },
    {
        "room_id": "R302",
        "room_name": "小组讨论室B",
        "capacity": 12,
        "location": "图书馆302",
        "available_dates": ["2026-12-16", "2026-12-19", "2026-12-25"]
    },
    {
        "room_id": "R401",
        "room_name": "大型演播会议室",
        "capacity": 200,
        "location": "综合楼4层",
        "available_dates": ["2026-12-20", "2026-12-22", "2026-12-26"]
    }
]


# 定义mcp 工具

@mcp.tool()
def search_events(keyword: str):
    '''
    根据关键词查询校园活动
    :param keyword:
    :return:
    匹配的活动列表
    '''
    keyword = keyword.strip().lower()
    results = [envent for envent in envents_data if keyword in envent['event_name'].lower()]
    return results
    pass

@mcp.tool()   #暴露可执行的业务函数
def search_available_room(booking_date,min_capacity:int):
    '''
    查询指定日期可预约的会议室
    :param booking_date:
    :param min_capacity:
    :return:
    '''
    results = []
    for room in rooms:
        date_available=(booking_date in room.get('available_dates'))
        capacity_available =(room.get('capacity')>=min_capacity)
        if date_available and capacity_available:
            results.append(room)

    return results
    pass

if __name__ == '__main__':
    mcp.run(transport='streamable-http')

