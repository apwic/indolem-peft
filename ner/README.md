# NER Task

This python script is based on adapters [example](https://github.com/adapter-hub/adapters/tree/master/examples/pytorch/token-classification) of usage that's modified for this usage.

## How to run

There's `run_ner.sh` that's available for usage, the script will run all the script in `./script` directory. It will train the model that's specified on baseline fine-tuning, LoRA, Prefix Tuning, Sequential Bottleneck, and UniPELT. By using this, it will push the result to wandb and the trained models to huggingface.

For modifed usage, you can use `run_ner.py` straight away. Check for the arguments that can be used.

```
python run_ner.py --help
```

Or you can modified based on each individual script that's in the `./script` directory.
