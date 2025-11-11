import re
import json

def parse_report(report_content):
    """
    Parses the FRAGSTATS_분석결과_종합보고서.txt to extract class-level metrics
    for the 8 relevant data points.
    """
    data = {}
    current_category = None
    current_region = None

    # Regex to find category, region, and type blocks
    category_re = re.compile(r"카테고리: (.*)")
    region_re = re.compile(r"지역: (.*)")
    type_re = re.compile(r"  타입: (giban_benefited|cls_1)")
    metric_re = re.compile(r"    ([A-Z_]+)\s*: \s* ([\d\.]+)")

    lines = report_content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        
        cat_match = category_re.search(line)
        if cat_match:
            current_category = cat_match.group(1).strip().lower()
            i += 1
            continue

        reg_match = region_re.search(line)
        if reg_match:
            current_region = reg_match.group(1).strip().lower()
            i += 1
            continue
        
        type_match = type_re.search(line)
        if type_match and current_category and current_region:
            type_name = type_match.group(1)
            
            # This is one of the 8 data points we care about
            data_point_name = f"{current_region}_{current_category}".replace(f"_{current_category}", "")
            
            if "hwasun" in data_point_name:
                data_point_name = f"hwasun_{current_category}"
            elif "naju" in data_point_name:
                data_point_name = f"naju_{current_category}"

            metrics = {}
            i += 1
            while i < len(lines) and not re.search(r"={3,}", lines[i]) and not re.search(r"  타입:", lines[i]) and not re.search(r"지역:", lines[i]):
                metric_line = lines[i]
                met_match = metric_re.search(metric_line)
                if met_match:
                    metric_name = met_match.group(1)
                    try:
                        metric_value = float(met_match.group(2))
                    except ValueError:
                        metric_value = None # For "N/A" cases
                    metrics[metric_name] = metric_value
                i += 1
            
            data[data_point_name] = metrics
            continue

        i += 1
        
    return data

if __name__ == "__main__":
    report_path = "/Users/yunhyungchang/Documents/FRAGSTATS/FRAGSTATS_분석결과_종합보고서.txt"
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    extracted_data = parse_report(content)
    
    # Save to a file for inspection
    output_path = "/Users/yunhyungchang/Documents/FRAGSTATS/extracted_metrics.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        
    print(f"Data extracted and saved to {output_path}")
    # Also print to stdout to see the result immediately
    print(json.dumps(extracted_data, indent=2))
