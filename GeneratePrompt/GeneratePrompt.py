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
        system_prompt = """请严格按模板生成代码，必须包含：
        1. 完整的AdvancedCatalysisParser类结构，且只需要这些结构，不要增加其他内容：
           - __init__方法初始化client和required_columns
           - generate_prompt方法（含动态字段说明）
        2. required_columns必须根据用户需求动态生成

        完整模板：
        class AdvancedCatalysisParser:
            def __init__(self, api_key):
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                )
                self.model_name = "deepseek/deepseek-chat"
                self.required_columns = [dynamic_columns]  # 根据用户输入生成
                
            def generate_prompt(self, text):
                \"\"\"Generated prompt\"\"\"
                return f\"\"\"
                ***Structured Data Extraction Instructions***
                Please extract the following information from the text and return it in a single-layer JSON format:
                Please strictly follow the following requirements:
                1. The output must be a valid, directly parseable standard JSON
                2. Use English double quotes
                3. Missing fields should be represented as null: 
                {instruction_content}
                
                ***Example JSON Structure***
                {example_json}
                
                ***Text to be Parsed***
                {{text}}
                \"\"\"
        
            All generated fields must be expressed in English
            1. instruction_content must explicitly request a single-layer JSON structure
            2. example_json uses a single-layer structure (no metadata/pollutants nesting)
            3. single_row_fields directly maps all fields
            4. Field names must match the user's requirements exactly
            5. Keep all exception handling logic
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
        "api_key": "------------------------------------------------------------",
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
