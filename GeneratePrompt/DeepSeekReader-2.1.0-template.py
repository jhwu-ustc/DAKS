# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 17:27:43 2025

@author: pc
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:46:18 2025
@author: JINGHANG
"""

import fitz
import pandas as pd
import time
import os
import json
from openai import OpenAI
import glob

class AdvancedCatalysisParser:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model_name = "deepseek/deepseek-chat"
        self.required_columns = [
            "literature_title", "literature_DOI", "catalyst_material_name", "is_single_atom_catalyst", 
            "metal_elements_with_valency", "non_metal_elements", "element_percentages", 
            "Km_H2O2", "Km_TMB", "Vmax_H2O2", "Vmax_TMB", "Kcat_H2O2", "Kcat_TMB", 
            "catalyst_concentration_or_dosage", "test_temperature", "test_pH"
        ]

    def generate_prompt(self, text):
        """重构后的精准提示词"""
        return f"""
        【结构化数据提取指令】
        请从文本中提取以下信息，并以单层JSON格式返回，并要求所有输出内容符合JSON的格式要求：
        请严格遵循以下要求：
        1. 必须使用标准JSON格式
        2. 所有字段名和字符串值必须使用英文双引号
        3. 缺失字段用null表示
        
        【关键约束】
        1. 仅当文献明确进行过氧化物酶（POD）活性测试（即使用H₂O₂作为氧化剂）时，才填写Km_TMB/Vmax_TMB
        2. 若文献使用溶解氧（O₂）而非H₂O₂催化TMB氧化（即oxidase活性），则所有POD相关字段设为null
        
        请从文本中提取以下信息，并以单层JSON格式返回    
        {{
            "literature_title": "title of the article, please remove punctuations",
            "literature_DOI": "DOI of the article",
            "catalyst_material_name": "Read the paragraphs about the POD-mimicking activity test, what is the name or designation of the used catalyst?",
            "is_single_atom_catalyst": "The used catalyst is single-atom catalyst（Yes/No）",
            "metal_elements_with_valency": "Read the results of EDS, EDX, XPS, or othher elemental analysis. Read the chemical formular of the catalyst (if provided). Metallic elements contained in the catalyst developed in this article (separated by commas), and their valence or oxidation states represented by Roman numerals in ()",
            "non_metal_elements": "Read the results of EDS, EDX, XPS, or othher elemental analysis. Read the chemical formular of the catalyst (if provided). Non-metallic elements contained in the catalyst developed in this article (separated by commas)",
            "element_percentages": "各元素的占比（包含单位，如有多种则用逗号隔开）",
            "Km_H2O2": "Km value obtained from the POD-mimicking activity test when using H2O2 as the substrate.）",
            "Km_TMB": "If the TMB oxidation test used H₂O₂ as oxidizer, provide Km value. Otherwise return null. Explicitly confirm H₂O₂ usage from Methods section.",
            "Vmax_H2O2": "Vmax value obtained from the POD-mimicking activity test when using H2O2 as the substrate.）",
            "Vmax_TMB": "If the TMB oxidation test used H₂O₂ as oxidizer, provide Vmax. Otherwise return null. Explicitly confirm H₂O₂ usage from Methods section.",
            "Kcat_H2O2": "Kcat value obtained from the POD-mimicking activity test when using H2O2 as the substrate.）",
            "Kcat_TMB": "If the TMB oxidation test used H₂O₂ as oxidizer, provide Kcat_TMB. Otherwise return null. Explicitly confirm H₂O₂ usage from Methods section.",
            "catalyst_concentration_or_dosage": "进行上述动力学测试时的催化剂浓度或投加量（mg/L）",
            "test_temperature": "the temperature condition of POD-mimicking activity test",
            "test_pH": "the pH condition of the POD-mimicking activity test"
        }}

        【示例JSON结构】
        {{
            "literature_title": "Example Title",
            "literature_DOI": "10.1234/example",
            "catalyst_material_name": "Example Material",
            "is_single_atom_catalyst": "Yes",
            "metal_elements_with_valency": "Fe(III), Cu(II)",
            "non_metal_elements": "C, O",
            "element_percentages": "Fe: 30%, Cu: 20%, C: 25%, O: 25%",
            "Km_H2O2": "0.5 mM",
            "Km_TMB": "0.3 mM",
            "Vmax_H2O2": "0.02 M/S",
            "Vmax_TMB": "0.015 M/S",
            "Kcat_H2O2": "10 S-1",
            "Kcat_TMB": "8 S-1",
            "catalyst_concentration_or_dosage": "5 mg/L",
            "test_temperature": "25°C",
            "test_pH": "7.0"
        }}

        【待解析文本】
        {text}
        """

    
    def extract_text(self, pdf_path):
        """优化文本提取逻辑"""
        full_text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text()
                full_text += text.replace('\n', ' ') + "\n"
            print(f"提取到有效文本段落: {len(full_text.split('.'))} 个")
            return full_text
        except Exception as e:
            print(f"PDF解析异常: {str(e)}")
            return ""
    
    def parse_response(self, response):
        """增强版JSON解析器"""
        try:
            cleaned = response.strip()
            
            # 改进Markdown代码块处理
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:-3].strip()  # 正确去除前后标记
            elif cleaned.startswith('```'):
                cleaned = cleaned[3:-3].strip()
            
            # 增强字符过滤
            cleaned = (
                cleaned.replace('“', '"')
                       .replace('”', '"')
                       .replace("'", '"')  # 处理单引号
                       #.replace("\\n", "")  # 去除转义字符
                       #.replace('"', '"')  # 转义双引号

            )
            
            # 新增空内容检查
            if not cleaned:
                raise ValueError("Empty JSON content")
                
            # 验证并解析
            data = json.loads(cleaned)
            
            # 字段类型校验
            return pd.DataFrame([{
                col: data.get(col, None) for col in self.required_columns  # 使用None保持null
            }])
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失败 | 清理后内容:\n{cleaned}\n错误详情: {str(e)}")  # 显示清理后的内容
            return pd.DataFrame()
        except Exception as e:
            print(f"预处理错误: {str(e)}")
            return pd.DataFrame()

    def process_pdf(self, pdf_path):
        """优化后的PDF处理函数（含完整错误处理）"""
        text = self.extract_text(pdf_path)
        if not text:
            print(f"警告: {pdf_path} 无有效文本")
            return pd.DataFrame()
    
        # 配置参数
        max_retries = 3
        base_delay = 30  # 基础等待时间(秒)
        timeout = 300  # 单次请求超时(秒)
        parsed_data = pd.DataFrame()
    
        for attempt in range(max_retries + 1):
            try:
                # 带超时的API请求
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": self.generate_prompt(text[:60000])  # 限制输入长度
                    }],
                    temperature=0.05,
                    timeout=timeout  # 新增超时参数
                )
    
                # 响应验证
                if not response.choices:
                    raise ValueError("空API响应")
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("无有效响应内容")
    
                # 解析响应
                parsed_data = self.parse_response(content)
                if not parsed_data.empty:
                    return parsed_data
    
            except Exception as e:
                error_type = self.classify_error(e)
                print(f"尝试 {attempt+1}/{max_retries} 失败 | 错误类型: {error_type}")
    
                # 错误处理策略
                if error_type == "RATE_LIMIT":
                    delay = base_delay * (2 ** attempt)  # 指数退避
                    print(f"速率限制触发，等待 {delay} 秒后重试...")
                    time.sleep(delay)
                elif error_type == "TIMEOUT":
                    timeout = min(timeout * 1.5, 600)  # 动态调整超时
                    print(f"超时设置调整为 {timeout} 秒")
                else:
                    break  # 非重试性错误立即终止
    
        # 最终失败处理
        print(f"无法处理 {os.path.basename(pdf_path)}")
        return pd.DataFrame()
    
    def classify_error(self, error):
        """错误分类器"""
        error_str = str(error).lower()
        
        if "rate limit" in error_str:
            return "RATE_LIMIT"
        elif "timeout" in error_str:
            return "TIMEOUT"
        elif "connection" in error_str:
            return "CONNECTION_ERROR"
        elif "invalid request" in error_str:
            return "INVALID_REQUEST"
        else:
            return "UNKNOWN_ERROR"

class SerialCatalysisParser(AdvancedCatalysisParser):
    def __init__(self, api_key):
        super().__init__(api_key)
        self.header_written = False  # 新增状态跟踪
        
    def process_files(self, pdf_files, output_path):
        """顺序处理所有文件"""
        total_files = len(pdf_files)
        success_count = 0
        
        # 初始化文件时直接写入表头
        if not os.path.exists(output_path):
            pd.DataFrame(columns=self.required_columns).to_csv(
                output_path, 
                index=False, 
                encoding='utf-8-sig'
            )
            self.header_written = True
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            try:
                print(f"进度: {idx}/{total_files} | 处理: {os.path.basename(pdf_path)}")
                start = time.perf_counter()
                
                df = self.process_pdf(pdf_path)
                if not df.empty:
                    # 追加模式写入，仅第一次需要header
                    df.to_csv(
                        output_path,
                        mode='a',
                        header=not self.header_written,
                        index=False,
                        encoding='utf-8-sig'
                    )
                    self.header_written = True  # 更新状态
                    success_count += 1
                
                elapsed = time.perf_counter() - start
                print(f"完成: {os.path.basename(pdf_path)} [耗时: {elapsed:.2f}s]")
            except Exception as e:
                print(f"处理失败: {pdf_path} - {str(e)}")
        
        return success_count



def main(config):
    start_time = time.perf_counter()
    
    # 路径验证
    print(f"\n当前工作目录: {os.getcwd()}")
    input_folder = os.path.abspath(config["input_folder"])
    print(f"配置输入目录: {input_folder}")
    
    if not os.path.exists(input_folder):
        print(f"错误：输入目录不存在 - {input_folder}")
        return

    # 获取PDF文件列表
    pdf_files = glob.glob(os.path.join(input_folder, "*.[pP][dD][fF]"))
    if not pdf_files:
        print("未找到PDF文件")
        return
    
    # 处理文件前先创建带表头的空文件
    output_path = os.path.abspath(config["output_csv"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 如果文件已存在则删除重建（可选）
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # 初始化解析器
    parser = SerialCatalysisParser(config["api_key"])
    
    # 处理文件
    success_count = parser.process_files(pdf_files, output_path)
    
    # 统计信息
    total_time = time.perf_counter() - start_time
    print(f"\n处理完成：成功 {success_count}/{len(pdf_files)} 篇文献")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"平均处理速度: {len(pdf_files)/total_time:.2f} 文件/秒")

if __name__ == "__main__":
    config = {
        "api_key": "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",
        "input_folder": "./pdf_files",
        "output_csv": "results_RShi.csv"
    }
    main(config)
