'''
只负责定义数据结构
'''
from pydantic import BaseModel, Field

'''
定义单个商品的选品结果
'''
class ProductSelectionItem(BaseModel):
    product_id: str = Field(description='商品的唯一编号, 如：p013')
    product_name: str = Field(description='商品名称')
    quantity: int = Field(ge=1, description='建议采购的数量,必须大于等于1')
    profit_margin: float = Field(ge=0, le=1, description='预计贡献的毛利率 0-1')
    recommend_reason: str = Field(description='推荐该商品的业务理由')
    pass


class SelectionResult(BaseModel):
    '''
    完成的选品结果  Agent 完成选品后
    '''
    success: bool = Field(description='是否成功完成选品')
    selected_products: list[ProductSelectionItem] = Field(default_factory=list)
    total_investment: float = Field(ge=0, description='推荐商品总投入,单位是元')
    budget: float = Field(gt=0, description='用户提供的采购预算')
    passed: bool = Field(description='选品结果是否通过初步审核')
    report_summary: str = Field(description='选品结论和整体分析摘要')
    failure_reason:str=Field(default=None,description="失败原因; 成功时候为空")
