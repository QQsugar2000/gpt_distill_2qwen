import os
import json
import time
import argparse
from PIL import Image
from script.generate_response import image_to_response
from script.config.prompt import prompt, system_prompt

def generate_responses(image_folder_path, output_json_path, max_retries=2):
    """
    处理指定文件夹中的所有图片，生成响应并保存到指定的 JSON 文件。

    :param image_folder_path: 图片文件夹路径
    :param output_json_path: 输出 JSON 文件路径
    :param max_retries: 最大重试次数
    """
    
    # 读取文件夹中的所有图片
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.png')]

    # 存储生成的所有响应
    responses = []

    # 遍历图片文件并生成响应
    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        
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
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(responses, f, ensure_ascii=False, indent=4)
                    print(f"生成的响应已保存到断点文件 '{output_json_path}'，处理已中断。")
                    exit(1)

    # 如果所有图片处理成功，则将响应保存到 JSON 文件
    if success:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=4)

        print(f"生成的响应已保存到 {output_json_path}")

def parse_args():
    """ 解析命令行参数 """
    parser = argparse.ArgumentParser(description="处理图片并生成响应")
    
    # 添加命令行参数
    parser.add_argument('--image_folder', type=str, required=True, help="图片文件夹路径")
    parser.add_argument('--output_json', type=str, required=True, help="输出 JSON 文件路径")
    parser.add_argument('--max_retries', type=int, default=2, help="最大重试次数（默认2）")
    
    # 返回解析后的参数
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 调用生成响应的函数
    generate_responses(
        image_folder_path=args.image_folder,
        output_json_path=args.output_json,
        max_retries=args.max_retries
    )
