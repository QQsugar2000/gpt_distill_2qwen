import os
import json
import time
from PIL import Image
from generate_response import image_to_response
from config.prompt import prompt, system_prompt

# 读取文件夹中的所有图片
folder_path = 'data/image'
image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# 存储生成的所有响应
responses = []

# 设置最大重试次数
max_retries = 2

# 遍历图片文件并生成响应
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    
    
    # 尝试调用 image_to_response 并捕获异常
    retries = 0
    success = False
    while retries <= max_retries:
        try:
            # 生成图片响应
            temp = image_to_response(image_path)
            
            # 组装消息条目
            message_entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt  # system_prompt 是系统指令
                    },
                    {
                        "role": "user",
                        "content": prompt  # prompt 是用户传入的大模型指令
                    },
                    {
                        "role": "assistant",
                        "content": temp  # 这里是返回的response
                    }
                ]
            }
            
            # 将当前条目添加到响应列表
            responses.append(message_entry)
            success = True
            break  # 如果成功，跳出重试循环
        
        except Exception as e:
            # 捕获异常并增加重试次数
            retries += 1
            print(f"处理 {image_file} 时出现错误: {e}. 正在进行第 {retries} 次重试...")
            if retries <= max_retries:
                time.sleep(2)  # 延迟 2 秒后再试
            else:
                print(f"处理 {image_file} 失败，已达到最大重试次数。")
                # 如果达到最大重试次数，保存已处理的部分数据并中断
                with open('data/breakpoint_responses.json', 'w', encoding='utf-8') as f:
                    json.dump(responses, f, ensure_ascii=False, indent=4)
                print(f"生成的响应已保存到断点文件 'breakpoint_responses.json'，处理已中断。")
                exit(1)

# 如果所有图片处理成功，则将响应保存到 JSON 文件
if success:
    output_file = 'data/responses.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)

    print(f"生成的响应已保存到 {output_file}")
