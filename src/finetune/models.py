"""
Functions for running pretrained/finetuned language models
- set up model
- finetune on a dataset
- sample outputs given input text
"""
from typing import Tuple, Set, Dict, List, Optional
import sys
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, DatasetInfo, DatasetDict
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model


def load_model(model_name: str, k: Optional[int] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    if not k:
        args = {}
    elif k == 4:
        args = {"load_in_4bit": True}
    elif k == 8:
        args = {"load_in_8bit": True}
    else:
        raise ValueError(f"Invalid k for kbit model: k={k}")

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **args)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name.startswith("codellama"):
        tokenizer.padding_side = "right"
        tokenizer.truncation = True
        tokenizer.padding = "max_length"

        if not tokenizer.pad_token:
            print("WARNING: no pad token defined by tokenizer, setting to default", file=sys.stderr)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def format_prompt(q: str, a: str) -> str:
    return f"# Question: {q}\n# Answer: {a}\n#DONE#"


def format_question(q: str) -> str:
    return f"# Question: {q}\n# Answer: "


def format_prompts(x) -> List[str]:
    outputs = []
    problems, solutions = x["problem"], x["solution"]
    assert len(problems) == len(solutions), \
        (f"Expected to have the same number of problems and solutions, but got "
         f"|problems|={len(problems)}, |solutions|={len(solutions)}")
    for problem, solution in zip(problems, solutions):
        outputs.append(format_prompt(problem, solution))
    return outputs


def finetune_model(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dict,
        max_seq_length: int,
        batch_size: int,
        epochs: int,
        lr_init: float,
        lr_scheduler_type: str,
        logging_steps: int,
        output_dir: str,
):
    # setup data collator
    # compensate for lack of context in response template
    response_template_w_context = "\n# Answer:"
    response_template_ids = tokenizer.encode(response_template_w_context, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids,
        tokenizer=tokenizer
    )

    def encode_len(x: Dict):
        return len(tokenizer.encode(format_prompt(x['problem'], x['solution'])))

    # filter out data that is too long
    orig_train_len = len(dataset['train'])
    train = dataset['train'].filter(lambda x: encode_len(x) < max_seq_length)
    orig_validation_len = len(dataset['validation'])
    validation = dataset['validation'].filter(lambda x: encode_len(x) < max_seq_length)
    print(f"Train after filtering to max_seq_len={max_seq_length}: "
          f"{orig_train_len} => {len(train)}")
    print(f"Validation after filtering to max_seq_len={max_seq_length}: "
          f"{orig_validation_len} => {len(validation)}")

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        bf16=True,
        evaluation_strategy="steps",
        eval_steps=5000,
        num_train_epochs=epochs,
        learning_rate=lr_init,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
    )

    # setup lora
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model,
        train_dataset=train,
        eval_dataset=validation,
        tokenizer=tokenizer,
        data_collator=collator,
        formatting_func=format_prompts,
        args=args,
        max_seq_length=max_seq_length,
    )
    trainer.train()
    trainer.save_model(output_dir)


def sample_model(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt_batch: List[str],
        max_length: int,
        max_new_tokens: int,
        do_sample: bool,
        num_beams: int,
) -> List[str]:
    tokenizer.padding_side = "left"
    input_ids = tokenizer(prompt_batch,
                          return_tensors="pt",
                          padding="max_length",
                          truncation=True,
                          max_length=max_length).to("cuda")
    print(f"Input ids: {input_ids.shape}")
    output_ids = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
    )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return outputs
