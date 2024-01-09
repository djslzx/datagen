# Dataset processing steps
Running LLM solution generation:
```bash
TODO
```

## Preparation for fine-tuning
Solution dataset => cleaned huggingface dataset:
```bash
data.py -m pre-ft \
        -d $proj_dir/datasets/wiz/all-solved/all-solved-20k:30k.jsonl \
        -o <output dir>
```

## Evaluation
Preparing a dataset for soln/test evaluation:
```bash
data.py -m pre-eval \
        -d $proj_dir/datasets/wiz/all-solved/all-solved-20k:30k.jsonl?? \
        -o $proj_dir/datasets/wiz/pre-eval...
```

Splitting:
```bash
TODO
```

Running evaluation on a prepared dataset:
```bash
data.py -m eval
        -d $proj_dir/datasets/wiz/pre-eval...
```

## Post-eval filtering
Evaluation data => filtered huggingface dataset
```bash
data.py -m eval-filter-ft \
        -d $proj_dir/datasets/wiz/eval-n100/all.jsonl
        -o <output dir>
```

