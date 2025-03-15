# 介绍
采用gpt api生成数据并微调qwen模型实战。采用闭源模型生成多模态数据（主要是VQA任务），并将生成的数据转化为swift框架可用的json格式，然后采用qwen模型微调。
目前仅提供lora微调方法，后续可能配置更多微调方法。
# 使用方法
clone以后，cd到文件夹,先配置prompt数据和api，然后运行generate_data生成数据，再运行process_all_image处理数据，最后运行training2微调模型。
### 配置prompt数据和api
+ 先配置prompt路径，路径'script/config/prompt.py'中有两个prompt，一个是系统prompt(system_prompt，用于配置大模型生成数据时使用的系统prompt，如“你是一个xx助手”)，一个是直接发给大模型的prompt，默认只需要配置后者。
+ 再配置api，路径'script/config/api.py'中，需要配置大模型api的url和key，url如果你使用了gpt的官方api不需要修改，如果使用了转发api，按转发接口填写。
### generate_data生成数据
'python script_name.py --image_folder data/image --output_json data/responses.json --max_retries 3'