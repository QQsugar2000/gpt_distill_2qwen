import os
import argparse
from script.config.infer_requests import infer_requests
from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift

def main(args):
    # 设置 CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    # 加载模型和 checkpoint
    model_path = args.model_path
    checkpoint_path = args.checkpoint_path
    lora_checkpoint = safe_snapshot_download(checkpoint_path)
    
    # 使用传入的模板类型和系统默认提示，如果未指定则使用模型自带的模板
    template_type = args.template_type or None
    default_system = args.default_system

    # 加载模型和 tokenizer
    model, tokenizer = get_model_tokenizer(model_path)
    model = Swift.from_pretrained(model, lora_checkpoint)
    
    # 如果 template_type 仍为 None，则取模型元信息中的默认模板
    template_type = template_type or model.model_meta.template
    template = get_template(template_type, tokenizer, default_system=default_system)
    
    # 构造推理引擎
    engine = PtEngine.from_model_template(model, template, max_batch_size=args.max_batch_size)
    request_config = RequestConfig(max_tokens=args.max_tokens, temperature=args.temperature)

    # 调用推理，输出结果
    resp_list = engine.infer(infer_requests, request_config)
    for i, resp in enumerate(resp_list):
        print(f'response{i}: {resp.choices[0].message.content}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='使用参数化方式运行 LLM 推理')
    parser.add_argument('--cuda_device', type=str, default='4', help='CUDA_VISIBLE_DEVICES 的值')
    
    # 必填参数，没有默认值
    parser.add_argument('--model_path', type=str, required=True, help='模型所在路径')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='checkpoint 所在路径')
    
    parser.add_argument('--template_type', type=str, default=None, help='对话模板类型，默认为 None 使用模型默认')
    parser.add_argument('--default_system', type=str, default="You are a helpful assistant.", help='系统默认提示语')
    parser.add_argument('--max_batch_size', type=int, default=2, help='推理时的最大批次大小')
    parser.add_argument('--max_tokens', type=int, default=512, help='生成回答时的最大 token 数')
    parser.add_argument('--temperature', type=float, default=0.0, help='推理温度')

    args = parser.parse_args()
    main(args)
