import json
from openai import OpenAI

# 填入你的 API Key
client = OpenAI(api_key="sk-f9ff5eeea9ca4a3aa4ab825775c14ce8", base_url="https://api.deepseek.com")

# ==========================================
# 1. 写一个真实的 Python 工具（这是真干活的）
# ==========================================
def calculator(expression):
    """这是一个极其简单的计算器函数"""
    try:
        # eval 能把字符串 "1234 * 5678" 当作数学公式算出来
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

# ==========================================
# 2. 给 AI 写一份“工具说明书”（极其重要）
# ==========================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "当需要进行复杂的数学计算时，调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "需要计算的数学表达式，例如 '1234 * 5678'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# ==========================================
# 3. 开始对话
# ==========================================
user_query = "请帮我算一下 345 乘以 789 等于多少？"
print(f"用户提问：{user_query}")

messages = [{"role": "user", "content": user_query}]

# 第一次呼叫大模型，把工具说明书（tools）传给它
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools
)

response_message = response.choices[0].message

# ==========================================
# 4. 检查 AI 是否想要使用工具
# ==========================================
if response_message.tool_calls:
    print("\n【后台日志】AI 觉得脑子算不准，决定呼叫计算器！")
    
    # 拿到 AI 发来的指令
    tool_call = response_message.tool_calls[0]
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    
    print(f"【后台日志】AI 要求调用的工具是：{function_name}")
    print(f"【后台日志】AI 传过来的参数是：{function_args}")
    
    # 我们（程序员）真正去执行本地的 Python 函数
    if function_name == "calculator":
        expression = function_args.get("expression")
        result = calculator(expression) # 算出真结果
        print(f"【后台日志】本地计算器算出的真实结果是：{result}")
        
        # 把 AI 之前的回答，以及工具算出的结果，再次打包发给 AI
        messages.append(response_message)
        messages.append({
            "role": "tool", # 告诉 AI 这是工具的结果
            "tool_call_id": tool_call.id,
            "content": result
        })
        
        # 第二次呼叫大模型，让它看着结果写最终回复
        final_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        
        print("\n--- AI 最终回答 ---")
        print(final_response.choices[0].message.content)
else:
    # 如果 AI 没用工具，直接输出了结果
    print("\n--- AI 最终回答 ---")
    print(response_message.content)