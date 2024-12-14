import requests
import base64

def read_image (path: str):
    with open(path, 'rb') as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode('utf-8')
    return None

def get_image_description(image_path: str, image_title: str, extra: str):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    encoded_image = read_image(image_path)
    if encoded_image is None:
        print('fail to encode image')
        return
    body = {
        "stream": False,
        "model": "OpenGVLab/InternVL2-26B",
        "messages": [
            { "role": "system", "content": "请专业和谨慎地回答问题" },
            {
                "role": "user",
                "content": [
                    { "type": "image_url", "image_url": { "url": encoded_image } },
                    { "type": "text", "text": f"{extra if extra is not None else ''}\n图片标题：{image_title}\n请根据以上消息，简要描述图片" }
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": .2
    }
    response = requests.post(
        url,
        headers={"Authorization": "Bearer sk-gcezbcticyektqbdrimjxrinkhfsxnsmedelzmfukfvdqzqv","accept": "application/json", "content-type": "application/json",},
        json=body,
        timeout=300,
    )
    json_response = response.json()
    print(json_response)
    content = json_response["choices"][0]["message"]["content"]
    print(content)


#print(read_image('./markdowns/imgs/上游代采.png'))
get_image_description('./markdowns/imgs/供应链金融贸易.png', "如何搭建供应链金融贸易供应链体系", """ 
供应链金融: 金融机构根据行业特点,围绕供应链中的核心企业,在实际交易过程的基础上,向核心企业其上下游相关企业提供的综合金融服务
核心企业: 3个亿销售额+3个亿净资产
夹层融资: 通过夹层机构提供资金实施杠政策
保理: (比如国企)利用应收账款转让给保理商 (银行)保理商提供合同额80%融资(指定账户), 缩短回款周期并且获得夹层融资;
价值: 在真实的贸易额里利用核心企业的价值提高杠杆能力缩短帐期;增加资金流通;获得更高收益;
""")