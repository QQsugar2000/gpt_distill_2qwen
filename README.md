
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
python process_all_image.py --image_folder_path data/image 
                            --output_json_path data/responses.json 
                            --max_retries 3
```
说明:
可修改参数如下
- `--image_folder_path`：指定图像数据的文件夹路径。你可以在里面放入图像文件，只要能够配合你上一步配置的prompt生成数据即可。
- `--output_json_path`：生成的JSON数据保存路径。
- `max_retries`: （选填）由于gpt api不一定稳定，生成数据过程中如果一旦网络波动就会失败，所以需要指定每条数据生成重试次数，默认为3次。

### 微调模型

使用以下命令来微调模型：

```bash
python train.py  --train_data data/processed_responses.json --output_dir checkpoints --num_epochs 5 --batch_size 1
```

---
输入参数说明
--cuda_devices：CUDA设备ID，默认值为 0。用于指定要使用的CUDA设备。
--model_id_or_path：模型的路径或ID，必填项。指定你要使用的模型路径或ID。
--system：系统Prompt，默认值为 You are a helpful assistant.。用于大模型生成数据时的系统Prompt。
--output_dir：输出目录，默认值为 checkpoint。模型训练过程中的输出将保存在此目录。
数据集相关配置
--dataset：数据集路径，必填项。指定训练使用的数据集路径。
--data_seed：数据划分的随机种子，默认值为 42。用于控制数据集划分时的随机性。
--max_length：最大token长度，默认值为 2048。指定模型输入的最大token长度。
--split_dataset_ratio：验证集的划分比例，默认值为 0.01。用于划分训练集与验证集的比例。
--num_proc：数据加载时的进程数，默认值为 4。指定数据加载时使用的并行进程数。
模型名称和作者
--model_name：模型的中文和英文名称，默认值为 ['小黄', 'Xiao Huang']。指定模型的名称。
--model_author：模型的中文和英文作者，默认值为 ['魔搭', 'ModelScope']。指定模型的作者名称。
LoRA配置
--lora_rank：LoRA的秩，默认值为 8。指定LoRA的秩。
--lora_alpha：LoRA的alpha值，默认值为 32。指定LoRA的alpha值。
训练相关配置
--learning_rate：训练时的学习率，默认值为 1e-4。指定优化器的学习率。
--per_device_train_batch_size：每个设备的训练批量大小，默认值为 1。设置每个设备的训练批量大小。
--per_device_eval_batch_size：每个设备的评估批量大小，默认值为 1。设置每个设备的评估批量大小。
--gradient_accumulation_steps：梯度累积的步数，默认值为 16。指定梯度累积的步数。
--num_train_epochs：训练的总周期数，默认值为 5。设置训练的总周期数。
# prompt编写指南
### 1. prompt编写原则
- **简洁明了**：prompt应该简洁明了，避免冗余信息。
- **明确意图**：明确表达用户的意图，避免歧义。
- **具体描述**：具体描述用户希望得到的信息，避免模糊不清。

### 2. prompt编写技巧
- **使用引导词**：在prompt中使用引导词，如“请”，“告诉我”，“帮我”等，可以引导用户按照你的期望进行回答。
- **使用具体描述**：在prompt中使用具体描述，如“请告诉我这张图片的内容”，“请帮我解释这个概念”等，可以引导用户提供具体的信息。
- **使用示例**：在prompt中使用示例，如“请告诉我这张图片的内容，例如：这张图片是一个女孩在玩滑板。”，可以引导用户按照你的期望进行回答。
# 效果展示