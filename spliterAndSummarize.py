from dataclasses import dataclass, asdict
import json
from os import walk
import os
import re
import pprint
from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter
import requests
import env

@dataclass
class MarkdownSplit:
    include_image: bool
    content_with_overview: str
    content_with_description: str
    path: str
    head: str

@dataclass
class SplitSummary:
    key_phrases: List[str]
    summary: str
    example_questions: List[str]
    path: str
    head: str

MD_HEADER_LEVEL = {
    'FIRST_LEVEL': 'FIRST_LEVEL',
    'SECOND_LEVEL': 'SECOND_LEVEL',
    'THIRD_LEVEL': 'THIRD_LEVEL',
    'FOURTH_LEVEL': 'FOURTH_LEVEL',
    'FIFTH_LEVEL': 'FIFTH_LEVEL'
}

def load_image_dict():
    with open('./markdowns/imgs/info.json', 'r') as f:
        return json.load(f)

def load_all_md_files():
    filepaths = []
    for (dirpath, dirnames, filenames) in walk('./markdowns'):
        filenames = list(filenames)
        filenames = list(filter(lambda it: it.endswith('.md'), filenames))
        filenames = list(map(lambda it: dirpath + '/' + it, filenames))
        filepaths.extend(filenames)
    ret = {}
    for path in filepaths:
        key = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            ret[key] = f.read()
    return ret

def split_md_files(file_dict: dict, image_dict: dict):
    headers_to_split_on = [
        ("#", MD_HEADER_LEVEL['FIRST_LEVEL']),
        ("##", MD_HEADER_LEVEL['SECOND_LEVEL']),
        ("###", MD_HEADER_LEVEL['THIRD_LEVEL']),
        ("####", MD_HEADER_LEVEL['FOURTH_LEVEL']),
        ("#####", MD_HEADER_LEVEL['FIFTH_LEVEL']),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    ret = {}
    for it in file_dict.items():
        file_name = it[0]
        raw_content = it[1]
        spliteds = markdown_splitter.split_text(raw_content)
        arr = []
        for it in spliteds:
            content = it.page_content
            content_with_overview = content
            content_with_description = content
            metadata = it.metadata
            include_image = False
            image_defines = re.findall(r"!\[.*?\]\(\./imgs/.*?\.(?:png|jpg|jpeg|gif)\)", content)
            for image_define in image_defines:
                image_name = re.search(r"imgs/(.*)\)", image_define).group(1)
                if isinstance(image_name, str) and image_name in image_dict:
                    include_image = True
                    img_obj =  image_dict[image_name]
                    keywords = img_obj['keywords']
                    overview = img_obj['overview']
                    description = img_obj['description']
                    keywords_join = '/'.join(keywords)
                    content_with_overview = content.replace(image_define, f"{image_define} 图片关键字: {keywords_join} 图片总结: {overview}")
                    content_with_description = content.replace(image_define, f"{image_define} 图片关键字: {keywords_join} 图片描述: {description}")
            path = '€'.join(list(metadata.values()))
            head = list(metadata.values())[-1]
            arr.append(MarkdownSplit(head=head, include_image=include_image, path=path, content_with_overview=content_with_overview, content_with_description=content_with_description))
        ret[file_name] = arr
    return ret

def extract_content(text: str, tag: str):
    pattern = rf'<{tag}>\s?(.*?)\s?</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else ''

def remove_empty (arr: List[str]):
    arr = list(map(lambda it: it.strip(), arr))
    return list(filter(lambda it: len(it) > 0, arr))

def call_llm(input: str, upper_context: str, lower_context: str = ''):
    content = f"""
# 要求
请根据<供应链金融基础知识>和<上下文>里面的内容，对<内容>里面的内容进行关键短语提取，总结和生成示例问题。示例问题是为了方便后续检索。
其中提取到的关键短语请放在<extracted_key_phrases></extracted_key_phrases>标签里面，关键短语可以是一个词语也可以是一个句子，以方便后续进行关键词和向量检索。
总结请放在<summary></summary>标签里面。
示例问题请放在<example_questions></example_questions>标签里面，生成的示例问题一部分需要对内容进行概括性提问，一部分需要对细节进行提问。

<供应链金融基础知识>
供应链金融是一种集物流运作、商业运作和金融管理为一体的管理行为和过程，它将贸易中的买方、卖方、第三方物流以及金融机构紧密地联系在了一起，实现了用供应链物流盘活资金、同时用资金拉动供应链物流的作用。
供应链融资形式：包括贷款、票据贴现、银行承兑汇票、保函、保理融资、融资租赁、信用证（含国际和国内）、押汇等。
供应链核心企业：在供应链上拥有该供应链关键资源和能力，具有相对较强实力，对其供应链上下游企业群体有较大影响和支配能力的企业
</供应链金融基础知识>

# 输出示例
<extracted_key_phrases>
风险调查的关键点
有实力关联方的责任和资产捆绑
</extracted_key_phrases>
<summary>
这段内容重点在讲述供应链金融的风险来自于法规的不透明和信用缺失，可以分成内部和外部两个方面，我们需要严格遵循四流合一和银行进行合作鉴别和防范。
</summary>
<example_questions>
供应链金融的风险来自于何处？
如何鉴别供应链金融的风险？
供应链金融的风险大吗？普通人适合做吗？
企业如何防范供应链金融的风险？
</example_questions>

# 输入
<上下文>
{upper_context}
<!-- input所在的位置 -->
{lower_context}
</上下文>
<内容>
{input}
</内容>

好的，请开始你的回答：
    """
    body = {
        "stream": False,
        "model": "Vendor-A/Qwen/Qwen2.5-72B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "请你扮演一个专业的供应链金融行业专家来回答问题。请带着专业和谨慎的态度。"
            },
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 4096,
        "temperature": .2
    }
    SILICON_FLOW_API_KEY = os.environ["SILICON_FLOW_API_KEY"]
    response = requests.post(
        "https://api.siliconflow.cn/v1/chat/completions",
        headers={"Authorization": f"Bearer {SILICON_FLOW_API_KEY}","accept": "application/json", "content-type": "application/json",},
        json=body,
        timeout=300,
    )
    json_response = response.json()
    # print(json_response)
    content = json_response["choices"][0]["message"]["content"]
    if isinstance(content, str):
        key_phrases = extract_content(content, "extracted_key_phrases")
        summary = extract_content(content, "summary")
        example_questions = extract_content(content, "example_questions")
        return SplitSummary(
            key_phrases=remove_empty(key_phrases.split('\n')),
            summary=summary,
            example_questions=remove_empty(example_questions.split('\n')),
            path='',
            head=''
        )
    else:
        return None

def summarize_by_llm(splits: List[MarkdownSplit]):
    first_levels = []
    second_levels = []
    third_levels = []
    fourth_levels = []
    fifth_levels = []
    second_level_summaries = []
    third_level_summaries = []
    fourth_level_summaries = []
    fifth_level_summaries = []
    def create_context_from_path(path: List[str]):
        context = ''
        for i in range(0, len(path)):
            head = path[i]
            content = ''
            levels = []
            if i == 0:
                levels = first_levels
            elif i == 1:
                levels = second_levels
            elif i == 2:
                levels = third_levels
            elif i == 3:
                levels = fourth_levels
            elif i == 4:
                levels = fifth_levels
            if len(levels) > 0:
                for it in levels:
                    if it.head == head:
                        content = it.content_with_description
                        break
            i = i + 1
            left = '#' * i
            context = context + f"{left} {head}\n{content}\n"
        return context
    def create_lower_context(level: int, path: str):
        levels = []
        if level == 0: levels = second_level_summaries
        elif level == 1: levels = third_level_summaries
        elif level == 2: levels = fourth_level_summaries
        elif level == 3: levels = fifth_level_summaries
        if len(levels) == 0:
            return ''
        sub_splits = list(filter(lambda it: it.path.startswith(path), levels))
        ret = ''
        for split in sub_splits:
            ret = ret + f"{'#' * (level + 2)} {split.head}\n关键短语：\n{('\n'.join(split.key_phrases))}\n总结：{split.summary}\n"
        return ret
    for split in splits:
        path = split.path
        len_spliter = len(re.findall('€', path))
        if len_spliter == 0: first_levels.append(split)
        elif len_spliter == 1: second_levels.append(split)
        elif len_spliter == 2: third_levels.append(split)
        elif len_spliter == 3: fourth_levels.append(split)
        elif len_spliter == 4: fifth_levels.append(split)
    sorted_splits = []
    sorted_splits.extend(fifth_levels)
    sorted_splits.extend(fourth_levels)
    sorted_splits.extend(third_levels)
    sorted_splits.extend(second_levels)
    sorted_splits.extend(first_levels)
    ret = []
    for split in sorted_splits:
        path = split.path
        len_spliter = len(re.findall('€', path))
        summary = None
        lower_context = create_lower_context(len_spliter, path)
        if len_spliter == 0:
            summary = call_llm(input=split.content_with_description, upper_context=f"# {split.head}", lower_context=lower_context)
        else:
            upper_context = create_context_from_path(path.split('€'))
            summary = call_llm(input=split.content_with_description, upper_context=upper_context, lower_context=lower_context)
        if summary is None:
            print(f"{path} is none")
        else:
            summary.head = split.head
            summary.path = path
            if len_spliter > 0:
                if len_spliter == 1: second_level_summaries.append(summary)
                if len_spliter == 2: third_level_summaries.append(summary)
                if len_spliter == 3: fourth_level_summaries.append(summary)
                if len_spliter == 4: fifth_level_summaries.append(summary)
            ret.append(summary)
    return ret

def write_json(file_name: str, arr: List[SplitSummary]):
    arr = list(map(lambda it: asdict(it), arr))
    with open(f'./markdown_summaries/{file_name}.json', 'w') as file:
        json.dump(arr, file, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    image_dict = load_image_dict()
    md_files = load_all_md_files()
    splited = split_md_files(md_files, image_dict)
    # test_str = 'test'
    # if test_str in splited: write_json(test_str, summarize_by_llm(splited[test_str]))
    # 这里建议一个一个模块进行处理 方便出现问题解决
    targets = [
        #"不良资产",
        #"供应链金融",
        #"国企",
        #"数据资产",
        #"方案",
        #"暴雷案例",
        #"票据",
        #"聚力盟",
        #"融资租赁",
        # "问答",
        # "风险",
    ]
    for target in targets:
        if target in splited:
            write_json(target, summarize_by_llm(splited[target]))


