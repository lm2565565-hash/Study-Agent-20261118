'''
多工具调用与动态决策

我准备开一一家户外用品店，目前预算2万元，帮我分析适合新店铺销售的商品，选出3款商品，
要求有一定的市场需求，竞争不要太激烈，预计毛利率不低于30%，最后给我一份选品报告
根据市场需求、竞争情况、毛利率、预算筛选候选商品 、生成选品报告


主要职责：
1. 接收用户任务
2. 调用 Agent
3. 展示工具调用轨迹
4. 输出最终选品报告

'''

# 目标理解----工具选择----多工具调用----连续执行----工具结果分析----动态决策----最终结果生成 【Agent核心知识点】
from langchain.messages import AIMessage, ToolMessage

from agent import selection_agent
from structModel import SelectionResult

# =====================================================
# 一、用户提出复杂任务
# =====================================================

USER_TASK = """
我准备开一家户外用品店，目前预算100元。
请帮我完成以下工作：
1. 分析适合新店销售的户外用品。
2. 从商品数据库中选择2款商品。
3. 商品必须有一定的市场需求。
4. 商品竞争不要太激烈。
5. 每款商品的预估贡献毛利率不低于70%。
6. 计算每款商品的采购数量。
7. 计算总采购投入和预算剩余金额。
8. 最后生成一份完整的选品报告。

请根据实际工具返回的数据进行分析。

不要直接编造商品或利润数据。
"""


# =====================================================
# 二、执行 Agent
# =====================================================

def run_selection_agent():
    print("=" * 60)
    print("跨境电商 AI 智能选品助理启动")
    print("=" * 60)

    print("\n用户任务：")
    print(USER_TASK)

    # 调用 Agent
    result = selection_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": USER_TASK
                }
            ]
        },
        config={
            "recursion_limit": 50  # 限制Agent在一次任务执行过程中，最多允许进行多次步图执行，防止Agent死循环 或者执行时间过长
        }
    )
    # =================================================
    # 三、打印 Agent 执行轨迹
    # =================================================
    print("\n")
    print("=" * 60)
    print("Agent 执行过程")
    print("=" * 60)

    messages = result["messages"]

    for index, message in enumerate(messages):

        # ---------------------------------------------
        # 1. 大模型决策
        # ---------------------------------------------

        if isinstance(message, AIMessage):

            if message.tool_calls:

                print(
                    f"\n[第{index + 1}条消息] "
                    "AI 决策：准备调用工具"
                )

                for tool_call in message.tool_calls:
                    print(
                        "工具名称：",
                        tool_call["name"]
                    )

                    print(
                        "工具参数：",
                        tool_call["args"]
                    )

            elif message.content:

                print(
                    f"\n[第{index + 1}条消息] "
                    "AI 输出："
                )

                print(message.content)

        # ---------------------------------------------
        # 2. 工具执行结果
        # ---------------------------------------------

        elif isinstance(message, ToolMessage):

            print(
                f"\n[第{index + 1}条消息] "
                "工具执行完成"
            )

            print(
                "工具名称：",
                message.name
            )

            print(
                "工具返回结果："
            )

            print(message.content)

    # =================================================
    # 四、提取最终回答
    # =================================================

    print("\n")
    print("=" * 60)
    print("最终选品报告------------------------------哈哈哈哈哈哈哈哈哈哈哈哈")
    print("=" * 60)
    struct_result = result.get('structured_response')
    if struct_result is None:
        print('Agent 没有返回结构化的结果.')
    else:
        if not isinstance(struct_result, SelectionResult):
            raise TypeError('结果化的结果不是 SelectionResult类型', type(struct_result))
        print('结构化数据获取成功...')
        print('\n 整体选品信息')
        print('是否成功:', struct_result.success)
        print('总采购投入:', struct_result.total_investment)
        print('用户预算:', struct_result.budget)
        print('初步审核状态:', struct_result.passed)
        print('选品摘要:', struct_result.report_summary)

        for index, item in enumerate(struct_result.selected_products):
            print(f'第 {index + 1}款商品')
            print('商品编号', item.product_id)
            print('商品名称', item.product_name)
            print('商品贡献毛利率', item.profit_margin)
            print('商品推荐理由', item.recommend_reason)

    print('结构化的json数据')

    json_result=struct_result.model_dump_json(indent=2)
    print(json_result)




# =====================================================
# 五、程序入口
# =====================================================

if __name__ == "__main__":
    run_selection_agent()
