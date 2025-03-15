
# 介绍
本项目通过使用GPT API生成数据并微调Qwen模型。采用闭源模型生成多模态数据（主要是VQA任务），并将生成的数据转化为Swift框架可用的JSON格式，然后使用Qwen模型进行微调。
你需要自己提供图片数据集，以及对应的指令prompt，并配置API。整体思路是用给定的prompt和图片生成问题，然后使用GPT API生成答案，拿这个答案去微调qwen。注意，这种方法并不保证生成的数据质量，因为GPT API生成的数据可能并不准确。

目前只提供LoRA微调方法，后续可能会配置更多微调方法。

# 使用方法

克隆本仓库后，进入项目文件夹并按照以下步骤操作：

1. 配置Prompt数据和API。
2. 运行 `generate_data` 生成数据。
3. 运行 `process_all_image` 处理数据。
4. 最后运行 `training2` 进行模型微调。

### 配置Prompt数据和API

1. **配置Prompt**  
   - 在 `script/config/prompt.py` 中配置两个Prompt：
     - **系统Prompt (`system_prompt`，选填)**：用于配置大模型生成数据时使用的系统Prompt，例如：“你是一个XX助手”。
     - **用户Prompt**：直接发送给大模型的Prompt，默认只需配置此项。
   
2. **配置API**  
   - 在 `script/config/api.py` 中配置大模型API的URL和Key：
     - 如果使用官方的GPT API，URL无需修改。
     - 如果使用转发API，请根据转发接口填写相关信息。

### 生成数据

运行以下命令来生成数据：

```bash
python process_all_image.py --image_folder_path data/image --output_json_path data/responses.json --max_retries 3
```
说明:
可修改参数如下
- `--image_folder_path`：指定图像数据的文件夹路径。你可以在里面放入图像文件，只要能够配合你上一步配置的prompt生成数据即可。
- `--output_json_path`：生成的JSON数据保存路径。
- `max_retries`: （选填）由于gpt api不一定稳定，生成数据过程中如果一旦网络波动就会失败，所以需要指定每条数据生成重试次数，默认为3次。

### 数据处理

运行以下命令来处理数据：（样例）

```bash
python script_name.py --process_all_image --input_json data/responses.json --output_json data/processed_responses.json
```

### 微调模型

使用以下命令来微调模型：

```bash
python script_name.py --training2 --train_data data/processed_responses.json --output_dir checkpoints --num_epochs 5 --batch_size 1
```

---


- `--max_retries`：请求失败时最大重试次数。
- `--input_json`：输入的未处理JSON数据路径。
- `--output_json`：处理后的JSON数据保存路径。
- `--train_data`：用于微调的训练数据路径。
- `--output_dir`：微调模型输出路径。
- `--num_epochs`：微调训练的周期数。
- `--batch_size`：训练时的批量大小。
# prompt编写指南
### 1. prompt编写原则
- **简洁明了**：prompt应该简洁明了，避免冗余信息。
- **明确意图**：prompt应该明确表达用户的意图，避免歧义。
- **具体描述**：prompt应该具体描述用户希望得到的信息，避免模糊不清。
- **避免主观判断**：prompt应该避免主观判断，避免引导用户产生偏见。
- **避免负面情绪**：prompt应该避免负面情绪，避免引导用户产生负面情绪。

### 2. prompt编写技巧
- **使用引导词**：在prompt中使用引导词，如“请”，“告诉我”，“帮我”等，可以引导用户按照你的期望进行回答。
- **使用具体描述**：在prompt中使用具体描述，如“请告诉我这张图片的内容”，“请帮我解释这个概念”等，可以引导用户提供具体的信息。
- **使用示例**：在prompt中使用示例，如“请告诉我这张图片的内容，例如：这张图片是一个女孩在玩滑板。”，可以引导用户按照你的期望进行回答。
# 效果展示