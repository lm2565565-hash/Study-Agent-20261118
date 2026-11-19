"""
业务工具层

主要职责：
1. 查询市场趋势
2. 搜索商品
3. 分析竞品
4. 计算利润
5. 生成选品报告

注意：
大模型只负责决定调用哪个工具。
实际业务数据处理和数学计算由 Python 完成。
"""

import json
import math

from langchain.tools import tool

from data import (
    products,
    MARKET_TRENDS,
    jingping,
    get_competitor_info, CATEGORY_NAMES,
)


# =====================================================
# 一、工具1：查询市场趋势
# =====================================================

@tool
def search_market_trends(category: str = "all") -> str:
    """
    查询户外用品市场趋势。

    category:
        商品分类英文代码。
        可以是 camping、hiking、lighting、
        cooking、apparel、safety 等。
        如果不指定，则查询全部分类。

    返回：
        各个品类的市场需求评分、增长率、
        竞争评分以及市场趋势。
    """

    print("\n[工具执行] search_market_trends")

    if category == "all":
        result = MARKET_TRENDS
    else:
        if category not in MARKET_TRENDS:
            return json.dumps(
                {"error": f"不存在的商品分类：{category}"},
                ensure_ascii=False
            )

        result = {
            category: MARKET_TRENDS[category]
        }

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2
    )


# =====================================================
# 二、工具2：搜索候选商品
# =====================================================

@tool
def search_products(
    category: str = "all",
    max_cost: float = 1000,
    min_stock: int = 1,
    limit: int = 40,
) -> str:
    """
    根据商品分类、采购成本和库存搜索商品。

    category:
        商品分类。all表示全部分类。

    max_cost:
        单件商品最高采购成本。

    min_stock:
        最低库存要求。

    limit:
        最多返回多少个商品。

    返回：
        符合条件的商品列表。
    """

    print("\n[工具执行] search_products")

    results = []

    for product in products.values():

        # 商品分类过滤
        if category != "all":
            if product["category"] != category:
                continue

        # 采购成本过滤
        if product["cost"] > max_cost:
            continue

        # 库存过滤
        if product["stock"] < min_stock:
            continue

        results.append(product)

    # 按采购成本从低到高排序
    results.sort(key=lambda x: x["cost"])

    # 控制返回数量
    results = results[:limit]

    return json.dumps(
        {
            "total": len(results),
            "products": results,
        },
        ensure_ascii=False,
        indent=2
    )


# =====================================================
# 三、工具3：查询竞品价格和竞争情况
# =====================================================

@tool
def get_competitor_prices(product_ids: list[str]) -> str:
    """
    查询指定商品的竞品价格和竞争情况。

    product_ids:
        商品编号列表，例如：
        ["p013", "p021", "p034"]

    返回：
        竞品数量、竞品平均售价、模拟月销量、
        竞争评分。

    竞争评分越低，代表竞争越弱。
    """

    print("\n[工具执行] get_competitor_prices")

    results = []

    for product_id in product_ids:

        if product_id not in products:
            continue

        product = products[product_id]

        competitor = get_competitor_info(
            product_id
        )

        # 竞争程度分类
        score = competitor["competition_score"]

        if score < 40:
            level = "低竞争"
        elif score < 65:
            level = "中等竞争"
        else:
            level = "高竞争"

        results.append({
            "product_id": product_id,
            "product_name": product["name"],
            "competitor_count": competitor["competitor_count"],
            "average_price": competitor["average_price"],
            "monthly_sales": competitor["monthly_sales"],
            "competition_score": score,
            "competition_level": level,
        })

    return json.dumps(
        results,
        ensure_ascii=False,
        indent=2
    )


# =====================================================
# 四、内部函数：计算单个商品利润
# =====================================================

def calculate_product_profit(
    product: dict,
    quantity: int,
    commission_rate: float,
) -> dict:
    """
    计算单个商品的利润。

    注意：
    这里计算的是扣除采购、物流和平台佣金后的
    预估销售贡献利润，不是扣除全部经营费用后的净利润。
    """

    cost = product["cost"]
    price = product["price"]
    logistics = product["logistics"]

    # 平台佣金
    commission = round(
        price * commission_rate,
        2
    )

    # 单件商品的预估贡献利润
    unit_profit = round(
        price - cost - logistics - commission,
        2
    )

    # 预估销售贡献毛利率
    margin_rate = round(
        unit_profit / price * 100,
        2
    )

    # 单件采购及物流投入
    unit_investment = cost + logistics

    # 总采购投入
    total_investment = round(
        unit_investment * quantity,
        2
    )

    # 预计销售额
    total_revenue = round(
        price * quantity,
        2
    )

    # 预计贡献利润
    total_profit = round(
        unit_profit * quantity,
        2
    )

    return {
        "product_id": product["id"],
        "product_name": product["name"],
        "quantity": quantity,
        "purchase_cost": cost,
        "selling_price": price,
        "logistics_cost": logistics,
        "commission_per_unit": commission,
        "unit_profit": unit_profit,
        "margin_rate": margin_rate,
        "total_investment": total_investment,
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "stock": product["stock"],
    }


# =====================================================
# 五、内部函数：统一执行利润测算
# =====================================================

def build_profit_analysis(
    product_ids: list[str],
    budget: float,
    commission_rate: float = 0.05,
) -> dict:
    """
    根据选中的商品和总预算，分配采购数量。

    分配规则：
    1. 每款商品平均分配采购预算。
    2. 采购数量不能超过库存。
    3. 单件贡献毛利率必须达到30%。
    4. 总投入不能超过总预算。
    """

    if not product_ids:
        return {"error": "没有提供商品编号"}

    if len(set(product_ids)) != len(product_ids):
        return {"error": "商品编号不能重复"}

    if len(product_ids) != 3:
        return {"error": "本项目要求选择恰好3款商品"}

    if budget <= 0:
        return {"error": "预算必须大于0"}

    if not 0 <= commission_rate < 1:
        return {"error": "佣金比例必须在0到1之间"}

    selected = [] #已经选中的商品列表里面

    for product_id in product_ids:

        if product_id not in products:
            return {
                "error": f"商品不存在：{product_id}"
            }

        selected.append(products[product_id])

    # 三款商品平均分配预算
    per_product_budget = budget / len(selected)

    results = []

    for product in selected:

        # 先计算单件利润
        single = calculate_product_profit(
            product,
            quantity=1,
            commission_rate=commission_rate,
        )

        # 毛利率必须达到30%
        if single["margin_rate"] < 70:
            return {
                "error": (
                    f"{product['name']}贡献毛利率"
                    f"{single['margin_rate']}%，"
                    "低于70%，不符合选品要求"
                )
            }

        # 单件采购及物流投入
        unit_investment = (
            product["cost"] + product["logistics"]
        )

        # 根据单款预算计算可采购数量
        quantity = math.floor(
            per_product_budget / unit_investment
        )

        # 不能超过库存
        quantity = min(
            quantity,
            product["stock"]
        )

        if quantity < 1:
            return {
                "error": f"{product['name']}预算不足，无法采购"
            }

        item_result = calculate_product_profit(
            product,
            quantity,
            commission_rate,
        )

        results.append(item_result)

    # 汇总投入
    total_investment = round(
        sum(x["total_investment"] for x in results),
        2
    )

    # 汇总销售额
    total_revenue = round(
        sum(x["total_revenue"] for x in results),
        2
    )

    # 汇总贡献利润
    total_profit = round(
        sum(x["total_profit"] for x in results),
        2
    )

    # 综合贡献毛利率
    overall_margin = round(
        total_profit / total_revenue * 100,
        2
    ) if total_revenue else 0

    return {
        "budget": budget,
        "commission_rate": commission_rate,
        "total_investment": total_investment,
        "remaining_budget": round(
            budget - total_investment,
            2
        ),
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "overall_margin": overall_margin,
        "products": results,
    }


# =====================================================
# 六、工具4：计算商品利润
# =====================================================

@tool
def calculate_profit(
    product_ids: list[str],
    budget: float,
    commission_rate: float = 0.05,
) -> str:
    """
    计算选中商品的采购数量、采购预算、
    预估销售额、预估贡献利润和贡献毛利率。

    product_ids:
        必须是3个商品编号。

    budget:
        用户总预算。

    commission_rate:
        平台佣金率，默认5%。

    注意：
        本工具会按照三款商品平均分配预算。
        计算结果必须满足每款商品贡献毛利率不低于30%。
    """

    print("\n[工具执行] calculate_profit")

    result = build_profit_analysis(
        product_ids=product_ids,
        budget=budget,
        commission_rate=commission_rate,
    )

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2
    )


# =====================================================
# 七、工具5：生成选品运营报告
# =====================================================

@tool
def generate_operation_report(
    product_ids: list[str],
    budget: float,
    commission_rate: float = 0.05,
) -> str:
    """
    根据最终选中的3款商品生成选品报告。

    product_ids:
        最终推荐的3个商品编号。

    budget:
        用户经营预算。

    commission_rate:
        平台佣金率。

    返回：
        包含市场分析、竞品分析、利润测算、
        采购预算和经营建议的报告数据。

    本工具使用模拟市场数据和模拟竞品数据。
    """

    print("\n[工具执行] generate_operation_report")

    # 重新计算利润，避免报告使用未经验证的数字
    profit = build_profit_analysis(
        product_ids,
        budget,
        commission_rate,
    )

    if "error" in profit:
        return json.dumps(
            profit,
            ensure_ascii=False
        )

    report_products = []

    for item in profit["products"]:

        product_id = item["product_id"]

        product = products[product_id]

        # 查询商品所属市场
        market = MARKET_TRENDS[
            product["category"]
        ]

        # 查询竞品
        competitor = get_competitor_info(
            product_id
        )

        report_products.append({
            **item,
            "category": CATEGORY_NAMES[
                product["category"]
            ],
            "market_demand_score": market[
                "demand_score"
            ],
            "market_growth_rate": market[
                "growth_rate"
            ],
            "competitor_count": competitor[
                "competitor_count"
            ],
            "competitor_average_price": competitor[
                "average_price"
            ],
            "monthly_sales_estimate": competitor[
                "monthly_sales"
            ],
            "competition_score": competitor[
                "competition_score"
            ],
        })

    report = {
        "report_title": "户外用品新店智能选品分析报告",
        "data_source": "模拟数据，非真实市场调查结果",
        "business_plan": {
            "budget": budget,
            "product_count": len(product_ids),
            "minimum_margin": 30,
            "commission_rate": commission_rate,
        },
        "financial_summary": {
            "total_investment": profit[
                "total_investment"
            ],
            "remaining_budget": profit[
                "remaining_budget"
            ],
            "expected_revenue": profit[
                "total_revenue"
            ],
            "expected_profit": profit[
                "total_profit"
            ],
            "overall_margin": profit[
                "overall_margin"
            ],
        },
        "recommended_products": report_products,
        "business_notes": [
            "优先进行小批量试销，不建议一次性耗尽全部预算",
            "关注实际转化率和退货率，及时调整采购数量",
            "平台佣金率需要根据实际销售平台重新核算",
            "物流成本未覆盖仓储、包装和退货等全部经营费用",
            "正式开店前需要通过真实平台数据验证市场需求",
        ],
    }

    return json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    )


# =====================================================
# 八、统一导出 Agent 可使用的工具
# =====================================================

selection_tools = [
    search_market_trends,
    search_products,
    get_competitor_prices,
    calculate_profit,
    generate_operation_report,
]
