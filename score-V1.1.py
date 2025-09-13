# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:57:21 2025

@author: pc
"""
# -*- coding: utf-8 -*-
import pandas as pd
import csv
import re
import time
from difflib import SequenceMatcher
from openai import OpenAI
import chardet

class DeepSeekValidator:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = "deepseek/deepseek-chat"
        self.delay = 0.1  # 安全间隔
        self.encoding_cache = {}
        self.special_columns = {'ORS', 'EPR_signals'}

    # region 文件处理模块
    def get_encoding(self, path):
        """智能编码检测"""
        if path not in self.encoding_cache:
            with open(path, 'rb') as f:
                rawdata = f.read(10000)
                result = chardet.detect(rawdata)
                self.encoding_cache[path] = result['encoding'] if result['confidence'] > 0.7 else 'gb18030'
        return self.encoding_cache[path]

    def clean_content(self, text):
        """统一字符清洗"""
        if not isinstance(text, str):
            return text
        return (
            text.replace('渭', 'μ')
            .replace('', '=')
            .replace('�', '')
            .replace('˙', '')
            .replace('·', '')
            .strip()
        )

    def safe_read_csv(self, path):
        """安全读取CSV文件"""
        for _ in range(3):  # 重试机制
            try:
                encoding = self.get_encoding(path)
                df = pd.read_csv(
                    path,
                    encoding=encoding,
                    keep_default_na=False,
                    on_bad_lines='warn'
                )
                return df.map(self.clean_content) 
            except Exception as e:
                print(f"读取{path}失败: {e}")
                continue
        raise ValueError(f"无法读取文件: {path}")

    def init_output(self, output_path, columns):
        """初始化输出文件"""
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([f"{col}_Score" for col in columns])
    # endregion

    # region 核心评分模块
    def preprocess_special_column(self, text, col_name):
        """特殊字段预处理（增强版）"""
        if col_name in self.special_columns:
            # 移除所有非字母数字字符（保留数字）
            cleaned = re.sub(r'[^a-zA-Z0-9]', '', str(text))
            # 统一化学符号格式
            replacements = {
                'O2': 'O2', 'OH': 'OH', 
                'SO4': 'SO4', 'SO4radical': 'SO4',
                'DMPOOH': 'DMPOOH', 'TEMP1O2': 'TEMP1O2'
            }
            return replacements.get(cleaned, cleaned).upper()
        return str(text)

    def generate_scoring_prompt(self, text1, text2, col_name):
        """生成动态提示词（优化版）"""
        base_rules = """评分规则（基准分100分）：
        1. 完全一致 → 100分
        2. 数值错误 → 扣30分
        3. 单位等效 → 不扣分（如mM与mmol/L）
        4. 格式差异 → 扣1-5分"""

        special_rules = ""
        if col_name in self.special_columns:
            special_rules = """
            █ 特殊字段处理规则：
            - 忽略所有非字母数字符号（如?、.、·等）
            - 示例：?OH ≡ OH ≡ .OH → 不扣分
            - 示例：SO4·- ≡ SO4- → 不扣分
            """

        return f"""请根据以下规则进行专业评分：
        【评估字段】{col_name}
        【标准文本】{text2}
        【待测文本】{text1}

        {base_rules}
        {special_rules}

        请直接返回0-100的整数，不要包含任何文字说明。"""
    
    def get_cell_score(self, text1, text2, col_name):
        """增强型评分解析"""
        try:
            # 预处理特殊字段
            text1_clean = self.preprocess_special_column(text1, col_name)
            text2_clean = self.preprocess_special_column(text2, col_name)
            
            # 快速匹配检查
            if text1_clean == text2_clean:
                return 100

            # 构造提示词
            prompt = self.generate_scoring_prompt(text1, text2, col_name)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4
            )
            raw_text = response.choices[0].message.content.strip()
            
            # 增强解析逻辑
            if match := re.search(r'\b(\d{1,3})\b', raw_text):
                score = int(match.group(1))
                return max(0, min(100, score))
            return self.get_fallback_score(text1_clean, text2_clean)
            
        except Exception as e:
            print(f"API请求失败: {e}")
            return self.get_fallback_score(text1_clean, text2_clean)

    def get_fallback_score(self, text1, text2):
        """降级评分策略（本地相似度计算）"""
        if text1 == text2:
            return 100
        similarity = SequenceMatcher(None, text1, text2).ratio()
        return int(similarity * 100)
    # endregion

    # region 主流程
    def process_files(self, v3_path, std_path, output_path):
        """主处理流程（增强版）"""
        # 读取数据
        df_v3 = self.safe_read_csv(v3_path)
        df_std = self.safe_read_csv(std_path)
        
        # 校验结构
        assert list(df_v3.columns) == list(df_std.columns), "列结构不一致"
        columns = df_v3.columns.tolist()
        
        # 初始化输出文件
        self.init_output(output_path, columns)
        
        # 逐行处理
        with open(output_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            for idx in range(len(df_v3)):
                row_scores = []
                for col in columns:
                    try:
                        v3_val = str(df_v3.iloc[idx][col])
                        std_val = str(df_std.iloc[idx][col])
                        
                        # 空值处理
                        if not std_val.strip():
                            score = 100 if not v3_val.strip() else 0
                        elif not v3_val.strip():
                            score = 0
                        else:
                            score = self.get_cell_score(v3_val, std_val, col)
                            time.sleep(self.delay)
                        
                        row_scores.append(score)
                        print(f"处理进度：行 {idx+1}/{len(df_v3)} | 列 {col} | 得分 {score}")
                        
                    except Exception as e:
                        print(f"处理失败：行 {idx+1} 列 {col} - {str(e)}")
                        row_scores.append(50)  # 错误中间值
                
                # 写入当前行
                writer.writerow(row_scores)
                f.flush()  # 强制写入

        print(f"处理完成，结果保存至: {output_path}")
    # endregion

if __name__ == "__main__":
    validator = DeepSeekValidator(
        api_key=""
    )
    validator.process_files(
        "results-gpt-4o.csv",
        "results-standard.csv",
        "final_scores-GPT-4o.csv"
    )

