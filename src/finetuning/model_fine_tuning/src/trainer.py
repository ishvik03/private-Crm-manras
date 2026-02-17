from peft import get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer
from .config_loader import load_training_config
from .prepare_datasets import prepare_datasets,load_tokenizer
from .modelling.fine_tune_modelling import load_base_model, get_lora_config
from .logging_utils import setup_logger
import inspect




def run_training():
    # sig = inspect.signature(TrainingArguments.__init__)
    #
    # with open("training_arguments_signature.txt", "w") as f:
    #     f.write(str(sig))

    # print("Saved to training_arguments_signature.txt")

    cfg = load_training_config()
    logger = setup_logger()
    logger.info("Loaded config: %s", cfg)

    model_name = cfg["model_name"]
    max_seq_len = cfg["max_seq_len"]

    logger.info("Preparing datasets...")

    #I want to know that every model type or company has their own tokeniser,
    #So do we just return that base level tokeniser or do we return that base level activated or used for this specific case
    train_ds, eval_ds = prepare_datasets(
        model_name=model_name,
        train_file=cfg["train_file"],
        eval_file=cfg["eval_file"],
        max_seq_len=max_seq_len
    )

    logger.info("Loading base model...")

    model_m = load_base_model(model_name)
    logger.info("Base model is loaded !!!")

    peft_config = get_lora_config()
    logger.info("LORA Configs are loaded  !!!")


    #get_peft_model :---  It takes a pretrained model and injects LoRA adapters
    #into specific layers, without modifying the original weights.
    model = get_peft_model(model_m, peft_config) #doubt
    model.config.use_cache = False

    logger.info("Get peft model is also done   !!!")
    model.print_trainable_parameters()

    


    training_args = TrainingArguments(
        output_dir=cfg["output_dir"], ####
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        logging_steps=cfg["logging_steps"],
        eval_strategy= cfg["eval_strategy"],
        save_strategy= cfg["save_strategy"],
        warmup_ratio=cfg["warmup_ratio"],
        bf16=cfg.get("bf16", False),
        report_to="none"
    )

    logger.info("Initializing SFTTrainer...")

    # sig =  inspect.signature(SFTTrainer.__init__)

    # with open("SFTT_arguments_signature.txt", "w") as f:
    #     f.write(str(sig))

    # print("Saved to SFTT_arguments_signature.txt")

    my_tokenizer = load_tokenizer(model_name)

    def format_example(example):
      return example["text"]

    trainer = SFTTrainer(
        model=model,
        # tokenizer = my_tokenizer,
        processing_class=my_tokenizer,
        args=training_args,
        train_dataset=train_ds,
        formatting_func=format_example,
        eval_dataset=eval_ds if eval_ds else None,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed :) ")

    logger.info("Saving final model...")
    trainer.save_model(cfg["output_dir"])
    my_tokenizer.save_pretrained(cfg["output_dir"])
    logger.info("Training complete.")

if __name__ == "__main__":
    run_training()
