# memes-revised
Overhauled version of memes-lol

One of the main changes is that I'm tryign to do everything with config-defined behavior

## Dependencies
All the staples are good: `torch`, `numpy`, `pandas`, `scipy`,  `matplotlib`, `yaml`

More specific libraries: `wandb`, `progressbar`

## Config-Defined Structure
The point of having everything about the models and datasets defined in config files is that hopefully, it lets us do much less mucking around in the code when we want to change something like model architecture or dataset structure. It also leaves a much more complete description of what we're doing on any given model training cycle, which can then be logged to W&B.

## Datasets
The datasets work by taking in raw data (possibly in some more easily editable or human-readable format), processing it, and saving it in a new location in a processed, more quickly loadable form. 

Essentially, each time you initialize the dataset class, it checks if there is an existing processed of the processed dataset. If so, it just loads it directly. If not, it goes to the raw dataset, does all the processing, and then saves the processed dataset. This means you only have to do the processing once for a dataset, and then it's always there.

This has the additional advantage of easily allowing you to have multiple copies of the dataset, possibly with different features, sizes, or really any other parameter of the dataset.

It also supports splitting the dataset into chunks, in case it's too large to be in memory all at once. An instance of the Dataset class can contain one, multiple, or all of the chunks of a chunked dataset.

## Training Script

### Options
`--project` (string): The W&B project the run is, or will be, in

`--config` (string) : The path to the config file

`--runid` (string) : The id of the run to resume, if resuming a run

`--from_sweep` : If the run is being controlled by a sweep. Should only be passed by a W&B agent

`--epochs_override` (int) : If you want to continue training a model that has already been trained for the number of epochs specified in the config and saved, this lets you train it for more.

`--save_epoch` (int) : How frequently to save the model. Saves every N epochs. If not specified, only saves at the end of training.

`--no_wandb` : If passed, cuts out all W&B logging and loading. For debugging. May save in a slightly unexpected place. Also, right now, metrics aren't printed or anything because they interfere with the progress bar (lol I'll fix this sometime), so you won't have access to model metrics without W&B.

`--restart` : If passed, restart the model's training from epoch 0. Will overwrite any existing saves.


### Sample Commands
You will need to change these to suit your needs.

#### Train new model from config:

```bash
python train_script.py --config testcfg.yaml --project meme-dummy-test  
```

#### Continue training on existing run, extending to 30 epochs (including previous ones):
```bash
python train_script.py --runid i8dh403g --project meme-dummy-test --epochs_override 30
```

#### Train model without W&B's meddling:
```bash
python train_script.py --config testcfg.yaml --no_wandb
```
