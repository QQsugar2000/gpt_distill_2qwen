import os
import argparse
from swift.llm import get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from functools import partial
from IPython.display import display
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning the model with specified configurations.")
    
    # CUDA devices
    parser.add_argument('--cuda_devices', type=str, default='0', help="CUDA device ID to use")
    
    # Model configurations
    parser.add_argument('--model_id_or_path', type=str, required=True, help="Path or ID of the model")
    parser.add_argument('--system', type=str, default='You are a helpful assistant.', help="System prompt for the model")
    parser.add_argument('--output_dir', type=str, default='checkpoint', help="Directory to save the output")
    
    # Dataset configurations
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--data_seed', type=int, default=42, help="Seed for data splitting")
    parser.add_argument('--max_length', type=int, default=2048, help="Maximum token length")
    parser.add_argument('--split_dataset_ratio', type=float, default=0.01, help="Ratio for validation split")
    parser.add_argument('--num_proc', type=int, default=4, help="Number of processes for data loading")
    
    # Model name and author
    parser.add_argument('--model_name', type=str, nargs=2, default=['小黄', 'Xiao Huang'], help="Model names")
    parser.add_argument('--model_author', type=str, nargs=2, default=['魔搭', 'ModelScope'], help="Model authors")
    
    # LoRA configurations
    parser.add_argument('--lora_rank', type=int, default=8, help="Rank for LoRA")
    parser.add_argument('--lora_alpha', type=int, default=32, help="Alpha for LoRA")
    
    # Training configurations
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help="Batch size per device for training")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help="Batch size per device for evaluation")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument('--num_train_epochs', type=int, default=5, help="Number of training epochs")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Set the CUDA device environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

    logger = get_logger()
    seed_everything(args.data_seed)

    # Hyperparameters for training
    model_id_or_path = args.model_id_or_path
    system = args.system
    output_dir = args.output_dir

    dataset = args.dataset
    data_seed = args.data_seed
    max_length = args.max_length
    split_dataset_ratio = args.split_dataset_ratio
    num_proc = args.num_proc
    model_name = args.model_name
    model_author = args.model_author

    # LoRA configurations
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha

    # training_args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_checkpointing=True,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        report_to=['tensorboard'],
        logging_first_step=True,
        save_strategy='steps',
        save_steps=50,
        eval_strategy='steps',
        eval_steps=50,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        metric_for_best_model='loss',
        save_total_limit=5,
        logging_steps=5,
        dataloader_num_workers=1,
        data_seed=data_seed,
    )

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    logger.info(f'output_dir: {output_dir}')

    # Obtain the model and template, and add a trainable Lora layer on the model.
    model, tokenizer = get_model_tokenizer(model_id_or_path)
    logger.info(f'model_info: {model.model_info}')
    template = get_template(model.model_meta.template, tokenizer, default_system=system, max_length=max_length)
    template.set_mode('train')

    target_modules = find_all_linears(model)
    lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                             target_modules=target_modules)
    model = Swift.prepare_model(model, lora_config)
    logger.info(f'lora_config: {lora_config}')

    # Print model structure and trainable parameters.
    logger.info(f'model: {model}')
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')

    # Download and load the dataset, split it into a training set and a validation set,
    # and encode the text data into tokens.
    train_dataset, val_dataset = load_dataset(dataset, split_dataset_ratio=split_dataset_ratio, num_proc=num_proc,
            model_name=model_name, model_author=model_author, seed=data_seed)

    logger.info(f'train_dataset: {train_dataset}')
    logger.info(f'val_dataset: {val_dataset}')
    logger.info(f'train_dataset[0]: {train_dataset[0]}')

    train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
    val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)
    logger.info(f'encoded_train_dataset[0]: {train_dataset[0]}')

    # Print a sample
    template.print_inputs(train_dataset[0])

    # Get the trainer and start the training.
    model.enable_input_require_grads()  # Compatible with gradient checkpointing
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
    )
    trainer.train()

    last_model_checkpoint = trainer.state.last_model_checkpoint
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')

    # Visualize the training loss.
    images_dir = os.path.join(output_dir, 'images')
    logger.info(f'images_dir: {images_dir}')
    plot_images(images_dir, training_args.logging_dir, ['train/loss'], 0.9)  # save images

    # Read and display the image.
    image = Image.open(os.path.join(images_dir, 'train_loss.png'))
    display(image)

if __name__ == "__main__":
    main()
