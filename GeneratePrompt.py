# -*- coding: utf-8 -*-
import os
import json
from openai import OpenAI

class DynamicCodeGenerator:
    def __init__(self, api_key):
        self.client = self._init_client(api_key)
        self.model = "deepseek/deepseek-chat"
        
    def _init_client(self, api_key):
            
        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        
    def generate_code(self, requirement):
        system_prompt = """
        Please strictly follow the template to generate the code, it must include:
        1. The complete structure of the AdvancedCatalysisParser class, and only these structures, do not add any other content:
           - The __init__ method to initialize the client and required_columns
           - The generate_prompt method (including dynamic field descriptions)
        2. The required_columns must be dynamically generated based on the user's needs

        Complete template:
        class AdvancedCatalysisParser:
            def __init__(self, api_key):
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                )
                self.model_name = "deepseek/deepseek-chat"
                self.required_columns = [dynamic_columns]  # Generated based on user input
                提取以下信息：催化反应温度（数值+单位）、催化反应pH值、材料名称、催化剂投加量、活性中心浓度E（单位M）、Km(mM) of H2O2、Km(mM) of TMB、Vmax of H2O2、Vmax of TMB、Kcat of H2O2、Kcat of TMB、文献DOI等共12项信息。
            def generate_prompt(self, text):
                \"\"\"重构后的精准提示词\"\"\"
                return f\"\"\"
                【结构化数据提取指令】
                请从文本中提取以下信息，并以单层JSON格式返回：
                请严格遵循以下要求：
                1. 输出必须是可直接解析的标准JSON
                2. 使用英文双引号
                3. 缺失字段用null表示：
                请从文本中提取以下信息，并以单层JSON格式返回    
                {instruction_content}
                
                【示例JSON结构】
                {example_json}
                
                【待解析文本】
                {{text}}
                \"\"\"
        
        生成规则：
        要求所有生成字段使用英文表达
        1. instruction_content必须明确要求单层JSON结构
        2. example_json使用单层结构（无metadata/pollutants嵌套）
        3. single_row_fields直接映射所有字段
        4. 字段名称必须与用户需求完全一致
        5. 保留所有异常处理逻辑
        """
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": requirement}
                ],
                temperature=0.1,
                max_tokens=5000
            )
            return self._clean_output(response.choices[0].message.content)
            
        except Exception as e:
            raise RuntimeError(f"API请求失败: {str(e)}")

    def _clean_output(self, raw_code):
        """增强的代码清洗逻辑"""
        # 修复不完整的方法
        if 'def parse_response' in raw_code and 'return pd.DataFrame' not in raw_code:
            raw_code += """
                    except Exception as e:
                        print(f"解析异常: {str(e)}")
                        return pd.DataFrame()"""
        
        # 确保代码块闭合
        code = raw_code.replace('```python', '').replace('```', '')
        return code.strip()

def main():
    config = {
        "api_key": "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",
        "output_file": "generated_code.txt"
    }
    
    try:
        # 获取用户输入
        user_input = input("请输入数据提取需求：\n").strip()
        if not user_input:
            raise ValueError("输入不能为空")
            
        # 生成代码
        generator = DynamicCodeGenerator(config["api_key"])
        generated = generator.generate_code(user_input)
        
        # 保存结果
        with open(config["output_file"], "w", encoding="utf-8") as f:
            f.write(generated)
            
        print(f"生成成功！文件已保存至：{os.path.abspath(config['output_file'])}")
        
    except Exception as e:
        print(f"错误发生：{str(e)}")
        print("故障排除：")
        print("1. 确认API密钥有效且格式正确")
        print("2. 检查网络连接是否正常")
        print("3. 简化需求描述后重试")

if __name__ == "__main__":
    main()
