# Summarization Task

This python script is based on adapters [example](https://github.com/adapter-hub/adapters/tree/master/examples/pytorch/token-classification) of usage that's modified for this usage.

## How to run

There's `run_summarization.sh` that's available for usage, the script will run all the script in `./script` directory. It will train the model that's specified on baseline fine-tuning, LoRA, Prefix Tuning, Sequential Bottleneck, and UniPELT. By using this, it will push the result to wandb and the trained models to huggingface. There are two datasets for summarization which is indosum and liputan6, set the `$DATASET` to either indosum or liputan6. Indosum database already provided in the data dir, for the liputan6 please refer to the original [repo](https://github.com/fajri91/sum_liputan6) to get the dataset, then put the dataset in the liputan6 dir.

For modifed usage, you can use `run_summarization.py` straight away. Check for the arguments that can be used.

```
python run_summarization.py --help
```

Or you can modified based on each individual script that's in the `./script` directory.
