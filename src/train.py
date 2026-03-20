import argparse
from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, prepare_model_for_kbit_training
from utils import (
    clear_gpu_memory,
    format_example,
    VLDataCollator,
    get_quant_config,
    get_lora_config,
    load_base_model,
    load_processor,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL for LaTeX OCR")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Base model name on HuggingFace")
    parser.add_argument("--output_dir", type=str, default="./qwen3vl-latex-ocr",
                        help="Directory to save model checkpoints")
    parser.add_argument("--dataset", type=str, choices=["linxy", "combined"], default="linxy",
                        help="Which dataset to use: 'linxy' only or 'combined' (linxy + MathWriting)")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_samples_mathwriting", type=int, default=2500,
                        help="Number of samples to take from MathWriting (if combined)")
    return parser.parse_args()

def main():
    args = parse_args()

    print("Loading datasets...")
    latex_train = load_dataset("linxy/LaTeX_OCR", "human_handwrite", split='train')
    latex_test = load_dataset("linxy/LaTeX_OCR", "human_handwrite", split='test')
    latex_train = latex_train.rename_column("text", "latex")
    latex_test = latex_test.rename_column("text", "latex")

    if args.dataset == "combined":
        mathwriting = load_dataset("deepcopy/MathWriting-human", split='train', columns=['image', 'latex'])
        mathwriting = mathwriting.select(range(min(args.max_samples_mathwriting, len(mathwriting))))
        mathwriting = mathwriting.rename_column("text", "latex")
        combined_train = concatenate_datasets([latex_train, mathwriting])
        train_dataset = combined_train
    else:
        train_dataset = latex_train

    print("Formatting dataset...")
    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)

    print("Loading base model with quantization...")
    quant_config = get_quant_config()
    model = load_base_model(args.model_name, quant_config)

    processor = load_processor(args.model_name)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    print("Preparing model for k-bit training and adding LoRA...")
    model = prepare_model_for_kbit_training(model)
    lora_config = get_lora_config(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        push_to_hub=False,
        report_to="none",
    )

    data_collator = VLDataCollator(processor)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    clear_gpu_memory()
    print("Starting training...")
    trainer.train()

    model.save_pretrained(f"{args.output_dir}/lora_adapter")
    processor.save_pretrained(f"{args.output_dir}/lora_adapter")
    print(f"Model saved to {args.output_dir}/lora_adapter")

if __name__ == "__main__":
    main()