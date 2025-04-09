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
        self.model_name = "deepseek/deepseek-r1"
        self.required_columns = [
            "catalyst", "catalyst_substrate", "SA_element", "SA_valence", "co_elements", 
            "oxidant", "pollutants", "pollutant_constant", "dose_catalyst", "dose_oxidant", 
            "pH", "catalyst_cycles", "pilot_pollutant", "pilot_time", "ORS", "EPR_signals", 
            "quencher", "DOI"
        ]
    
    def generate_prompt(self, text):
        """Generated prompt"""
        return f"""
        ***Structured Data Extraction Instructions***
        Please extract the following information from the text and return it in a single-layer JSON format:
        Please strictly follow the following requirements:
        1. The output must be a valid, directly parseable standard JSON
        2. Use English double quotes
        3. Missing fields should be represented as null: 
        - catalyst: name or designation of the catalyst used in the kinetic studies with the best performance, remove any symbol not suitable for JSON
        - catalyst_substrate: substrate or support of the single-atom catalyst in the catalyst used in the kinetic studies with the best performance (with all doping detailes explicitly stated)
        - SA_element: element type of the single atom in the catalyst used in the kinetic studies with the best performance 
        - SA_valence: the valence or oxidation state of the single atom used in the kinetic studies with the best performance
        - co_elements: type of coordinating atoms
        - oxidant: oxidant used in the catalytic oxidations (such as H2O2, PMS, PDS, PAA)
        - pollutants: types and concentrations of target pollutants in the kinetic studies, formatted as concentration followed by pollutant in parentheses (if multiple pollutants are present, separate them with commas)
        - pollutant_constant: maximum degradation rate constant of the target pollutant(s) (if multiple pollutants are involved, separate them with commas, and specify the corresponding pollutants in parentheses)
        - dose_catalyst: catalyst dosage for the catalytic reaction (in experimental group with the highest kinetic constant)
        - dose_oxidant: oxidant dosage for the catalytic reaction (in experimental group with the highest kinetic constant)
        - pH: pH condition of the catalytic reaction (in experimental group with the highest kinetic constant)
        - catalyst_cycles: number of catalytic cycles for the catalyst (reusability)
        - pilot_pollutant: type and concentration of target pollutant(s) in the pilot-scale experiment, formatted as concentration followed by pollutant in parentheses (if multiple pollutants are present, separate them with commas)
        - pilot_time: duration or reaction time of the pilot-scale experiment
        - ORS: dominant oxidative reactive species identified in mechanism studies (ORS; if multiple types, separate them with commas)
        - EPR_signals: signals observed in the spin-trapping EPR tests (such as DMPO-OH, DMPO-R; if multiple signals are observed, separate them with commas)
        - quencher: quenchers, inhibitors, or scavengers used in the mechanism studies, formatted as inhibitory efficiency (percentage or qualitative effect) followed by quencher type in parentheses, e.g., '90% (ethanol), significant inhibition (tert-butanol)' (if multiple pollutants are present, separate them with commas)
        - DOI: DOI of the literature
        
        ***Example JSON Structure***
        {{
            "catalyst": "Pt1/Fe2O3",
            "catalyst_substrate": "Fe2O3",
            "SA_element": "Pt",
            "SA_valence": "II",
            "co_elements": "O, Fe",
            "oxidant": "H2O2",
            "pollutants": "10 mg/L (phenol), 5 mg/L (benzene)",
            "pollutant_constant": "0.05 min-1 (phenol), 0.03 min-1 (benzene)",
            "dose_catalyst": "0.1 g/L",
            "dose_oxidant": "10 mM",
            "pH": "7.0",
            "catalyst_cycles": "5",
            "pilot_pollutant": "20 mg/L (phenol)",
            "pilot_time": "2 hours",
            "ORS": "OH, SO4-",
            "EPR_signals": "DMPO-OH, DMPO-SO4",
            "quencher": "slight inhibition (ethanol), 85% (100 mM tert-butanol), ",
            "DOI": "10.1039/d0ee00000a"
        }}
        
        ***Text to be Parsed***
        {text}
        """

    
    def extract_text(self, pdf_path):
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
            
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:-3].strip()
            elif cleaned.startswith('```'):
                cleaned = cleaned[3:-3].strip()
            
            cleaned = (
                cleaned.replace('“', '"')
                       .replace('”', '"')
                       .replace("'", '"') 
                       #.replace("\\n", "") 
                       #.replace('"', '"') 

            )
            
            if not cleaned:
                raise ValueError("Empty JSON content")
                
            data = json.loads(cleaned)
            
            return pd.DataFrame([{
                col: data.get(col, None) for col in self.required_columns 
            }])
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失败 | 清理后内容:\n{cleaned}\n错误详情: {str(e)}") 
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
        base_delay = 30
        timeout = 300
        parsed_data = pd.DataFrame()
    
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": self.generate_prompt(text[:60000])  # 限制输入长度
                    }],
                    temperature=0.05,
                    timeout=timeout  # 新增超时参数
                )
    
                if not response.choices:
                    raise ValueError("空API响应")
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("无有效响应内容")
    
                parsed_data = self.parse_response(content)
                if not parsed_data.empty:
                    return parsed_data
    
            except Exception as e:
                error_type = self.classify_error(e)
                print(f"尝试 {attempt+1}/{max_retries} 失败 | 错误类型: {error_type}")
    
                if error_type == "RATE_LIMIT":
                    delay = base_delay * (2 ** attempt)
                    print(f"速率限制触发，等待 {delay} 秒后重试...")
                    time.sleep(delay)
                elif error_type == "TIMEOUT":
                    timeout = min(timeout * 1.5, 600) 
                    print(f"超时设置调整为 {timeout} 秒")
                else:
                    break
    
        print(f"无法处理 {os.path.basename(pdf_path)}")
        return pd.DataFrame()
    
    def classify_error(self, error):
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
        self.header_written = False
        
    def process_files(self, pdf_files, output_path):
        total_files = len(pdf_files)
        success_count = 0
        
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
                    df.to_csv(
                        output_path,
                        mode='a',
                        header=not self.header_written,
                        index=False,
                        encoding='utf-8-sig'
                    )
                    self.header_written = True
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

    pdf_files = glob.glob(os.path.join(input_folder, "*.[pP][dD][fF]"))
    if not pdf_files:
        print("未找到PDF文件")
        return
    
    output_path = os.path.abspath(config["output_csv"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        os.remove(output_path)
    
    parser = SerialCatalysisParser(config["api_key"])
    
    success_count = parser.process_files(pdf_files, output_path)
    
    total_time = time.perf_counter() - start_time
    print(f"\n处理完成：成功 {success_count}/{len(pdf_files)} 篇文献")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"平均处理速度: {len(pdf_files)/total_time:.2f} 文件/秒")

if __name__ == "__main__":
    config = {
        "api_key": "----------------------------------------------",
        "input_folder": "./pdf_files",
        "output_csv": "results-deepseek-r1.csv"
    }
    main(config)
