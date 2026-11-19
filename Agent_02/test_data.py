from pydantic import ValidationError
from structModel import SelectionResult

selection_data={
    "success":True,
    "selected_products":[
        {"product_id":"p013","product_name":"手机","quantity":1,"profit_margin":0.415,
         "recommend_reason":"市场需求好"
         }

    ],
    "total_investment":15200,
    'budget':20000,
    'passed':True,
    'report_summary':"满足要求"
}

try:
    result=SelectionResult.model_validate(selection_data)
    print('ook')
except ValidationError as e:
    print(e)