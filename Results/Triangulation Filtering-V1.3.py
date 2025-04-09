# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 17:36:56 2025
@author: pc
Optimized version with enhanced encoding handling
"""

import csv
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 0=DeepSeek, 1=Gemini, 2=GPT-4o）
priority_map = {
    'catalyst': 1,
    'catalyst_substrate': 0,
    'SA_element': 0,
    'SA_valence': 0,
    'co_elements': 1,
    'oxidant': 1,
    'pollutants': 0,
    'pollutant_constant': 0,
    'dose_catalyst': 2,
    'dose_oxidant': 2,
    'pH': 2,
    'catalyst_cycles': 0,
    'pilot_pollutant': 0,
    'pilot_time': 1,
    'ORS': 2,
    'EPR_signals': 0,
    'quencher': 2,
    'DOI': 0
}

def validate_encoding(filename):
    try:
        with open(filename, 'rb') as f:
            f.read().decode('utf-8-sig')
        return True
    except UnicodeDecodeError as e:
        logging.error(f"编码验证失败: {filename} - {str(e)}")
        return False

def read_csv(filename):
    """读取CSV文件并返回字典列表（增强编码处理）"""
    if not validate_encoding(filename):
        raise ValueError(f"文件编码不兼容: {filename}")
    
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader((line.replace('\ufeff', '') for line in f))
        return [{
            k: str(v).strip() if v is not None else ''
            for k, v in row.items()
        } for row in reader]

def compare_values(values):
    """检查三个值是否一致，返回布尔值和统一值（增强空值处理）"""
    cleaned = [v for v in values if v.strip()]
    if len(cleaned) == 0:
        return False, ''
    
    unique_values = set(cleaned)
    return len(unique_values) == 1, next(iter(unique_values))

def process_row(row1, row2, row3, fieldnames):
    new_row = OrderedDict()
    for field in fieldnames:
        values = [
            row1.get(field, ''),
            row2.get(field, ''),
            row3.get(field, '')
        ]
        
        encoded_values = [
            v.encode('utf-8').decode('utf-8-sig').strip()
            for v in values
        ]
        
        is_consistent, val = compare_values(encoded_values)
        if is_consistent:
            new_row[field] = val
        else:
            selected_idx = priority_map.get(field, 0)
            selected_val = encoded_values[selected_idx]
            if selected_val != encoded_values[0]:
                logging.debug(f"字段 {field} 使用优先级 {selected_idx} 的值: {selected_val}")
            new_row[field] = selected_val
    return new_row

def main():
    try:
        files = [
            read_csv('results-DeepSeek.csv'),
            read_csv('results-Gemini.csv'),
            read_csv('results-GPT.csv')
        ]

        line_counts = [len(data) for data in files]
        if len(set(line_counts)) > 1:
            raise ValueError(f"输入文件行数不一致: DeepSeek={line_counts[0]}, Gemini={line_counts[1]}, GPT-4o={line_counts[2]}")

        fieldnames = list(files[0][0].keys())

        with open('results-consensus-5.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(len(files[0])):
                try:
                    row1 = files[0][i]
                    row2 = files[1][i]
                    row3 = files[2][i]
                except IndexError as e:
                    logging.error(f"行索引越界: {i} - {str(e)}")
                    break

                new_row = process_row(row1, row2, row3, fieldnames)
                writer.writerow(new_row)
                
        logging.info("文件合并完成，输出保存至 results-consensus-5.csv")

    except Exception as e:
        logging.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
