"""
Functions for running pretrained/finetuned language models
- set up model
- finetune on a dataset
- sample outputs given input text
"""
from typing import Tuple, Set, Dict, List, Optional
import sys
import pdb
import torch
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    EvalPrediction,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, DatasetInfo, DatasetDict
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model

from finetune.root import evaluate


def load_model_and_tokenizer(
        model_name: str, 
        k: Optional[int] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model = load_model(model_name, k if k is not None else 32)
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


def load_model(model_name: str, k: int = 32) -> PreTrainedModel:
    if not k or k == 32:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif k == 4:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        )
    elif k == 8:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            load_in_8bit=True,
        )
    elif k == 16:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    else:
        raise ValueError(f"Invalid k for kbit model: k={k}")
    return model


def load_peft_model_and_tokenizer(
        model_name: str, 
        k: int, 
        adapter_name: str
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model, tokenizer = load_model_and_tokenizer(model_name, k)
    peft_model = PeftModel.from_pretrained(model, adapter_name)
    return peft_model, tokenizer


def format_prompt(q: str, a: str) -> str:
    return f"# Question:\n{q}\n# Answer:\n{a}\n#DONE#"


def format_question(q: str) -> str:
    return f"# Question:\n{q}\n# Answer:\n"


def format_prompts(dataset) -> List[str]:
    outputs = []
    problems, solutions = dataset["problem"], dataset["solution"]
    assert len(problems) == len(solutions), \
        (f"Expected to have the same number of problems and solutions, but "
         f"|problems|={len(problems)}, |solutions|={len(solutions)}")
    for problem, solution in zip(problems, solutions):
        outputs.append(format_prompt(problem, solution))
    return outputs


def encode_len(tokenizer: PreTrainedTokenizer, x: Dict):
    return len(tokenizer.encode(format_prompt(x['problem'], x['solution'])))


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
        eval_steps: int,
        output_dir: str,
):
    # setup data collator
    # compensate for lack of context in response template
    response_template_w_context = "\n# Answer:"
    response_template_ids = tokenizer.encode(response_template_w_context, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids,
        tokenizer=tokenizer,
    )

    # filter out data that is too long
    orig_train_len = len(dataset['train'])
    train = dataset['train'].filter(lambda x: encode_len(tokenizer, x) < max_seq_length)
    orig_validation_len = len(dataset['validation'])
    validation = dataset['validation'].filter(lambda x: encode_len(tokenizer, x) < max_seq_length)
    print(f"Train after filtering to max_seq_len={max_seq_length}: "
          f"{orig_train_len} => {len(train)}")
    print(f"Validation after filtering to max_seq_len={max_seq_length}: "
          f"{orig_validation_len} => {len(validation)}")
    if len(validation) > 100:
        print("Clamping validation dataset to 100 instances...")
        validation = validation.select(range(100))

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
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
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=train,
        eval_dataset=validation,
        tokenizer=tokenizer,
        data_collator=collator,
        formatting_func=format_prompts,
        args=args,
        max_seq_length=max_seq_length,
        # timeout=5,
    )
    trainer.evaluate()
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
        end_stamp="#DONE#",
) -> List[str]:
    tokenizer.padding_side = "left"
    model = model
    input_ids = tokenizer(prompt_batch,
                          return_tensors="pt",
                          padding="max_length",
                          truncation=True,
                          max_length=max_length).to("cuda")
    # print(f"Input ids: {input_ids.shape}")
    output_ids = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
    )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return outputs

