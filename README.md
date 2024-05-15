# memes-revised
Overhauled version of memes-lol

One of the main changes is that I'm tryign to do everything with config-defined behavior

## Config-Defined Structure
The point of having everything about the models and datasets defined in config files is that hopefully, it lets us do much less mucking around in the code when we want to change something like model architecture or dataset structure. It also leaves a much more complete description of what we're doing on any given model training cycle, which can then be logged to W&B.

## Datasets
The datasets work by taking in raw data (possibly in some more easily editable or human-readable format), processing it, and saving it in a new location in a processed, more quickly loadable form. 

Essentially, each time you initialize the dataset class, it checks if there is an existing processed of the processed dataset. If so, it just loads it directly. If not, it goes to the raw dataset, does all the processing, and then saves the processed dataset. This means you only have to do the processing once for a dataset, and then it's always there.

This has the additional advantage of easily allowing you to have multiple copies of the dataset, possibly with different features, sizes, or really any other parameter of the dataset.

It also supports splitting the dataset into chunks, in case it's too large to be in memory all at once. An instance of the Dataset class can contain one, multiple, or all of the chunks of a chunked dataset.

