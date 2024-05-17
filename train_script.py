import torch
from torch.utils.data import random_split, DataLoader
from torch.nn import MSELoss

from memes_base import losses, utils, models
import progressbar
import wandb
import argparse
import os

def train(model, dataloader, optimizer, loss_fn, device):

    size = len(dataloader.dataset)
    model.train()

    net_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        #* Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        net_loss += loss.item() * len(X)

    net_loss /= size

    return net_loss

def test(model, dataloader, loss_fn, device):
    size = len(dataloader.dataset)
    # num_batches = len(dataloader)
    model.eval()

    trues, preds = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            trues.append(y)
            preds.append(pred)

        #* Combine all predictions and trues and calculate loss all at once
        trues = torch.concatenate(trues, dim=0)
        preds = torch.concatenate(preds, dim=0)
        
        loss = loss_fn(preds, trues)
        #? Should calculate other metrics here

    return loss.item()

def main(pargs):
    runid = pargs.runid
    project = pargs.project
    config_path = pargs.config
    from_sweep = pargs.from_sweep
    epochs_override = pargs.epochs_override
    save_epoch = pargs.save_epoch
    use_wandb = not pargs.no_wandb

    if use_wandb:
        if runid is not None:
            #* Resuming an existing run
            run = wandb.init(project=project, id=runid, resume='must')

            if 'full_config' in wandb.config:
                config = wandb.config["full_config"]
            else:
                print("Full config for this run was not logged in W&B config. Loading from existing config. This may cause problems.")
                if config_path is None:
                    print("Loading from base_config logged by W&B")
                    config = utils.load_config(wandb.config['base_config'])
                else:
                    print("Loading from config path argument")
                    config = utils.load_config(config_path)
                utils.build_agent_config(config, wandb.config, run)
        elif from_sweep:
            #* Making a new run as determined by a sweep
            run = wandb.init()

            config = utils.load_config(wandb.config['base_config'])
            utils.build_agent_config(config, wandb.config, run)
        else:
            #* This should mean we're manually starting a run directly from a specified config file
            assert config_path is not None
            config = utils.load_config(config_path)
            run = wandb.init(project=project, config=utils.build_wandb_config(config, config_path))
            config["Training_Directory"] += f"run_{run.id}/"

        wandb.config.update({'full_config' : config})
    else:
        #* Not involving W&B at all
        assert config_path is not None
        config = utils.load_config(config_path)


    #* Unpacking the config parameters
    batch_size = config["Training"]['batch_size']
    learning_rate = config["Training"]['learning_rate']
    epochs = config['Training']['epochs'] if epochs_override is None else epochs_override
    
    #* DATASET CREATION
    full_dataset = utils.buildFromConfig(config["Dataset"])

    #* Perform the train-test splitting
    train_ds, test_ds = random_split(full_dataset, (0.9, 0.1))

    #* Make the dataloaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False) #* Whether to drop the final incomplete batch.
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=False) #* False means batches will vary in length

    #* MODEL CREATION
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = utils.buildFromConfig(config["Model"]).to(device)

    #* LOSS AND OPTIMIZER
    loss = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #* Load latest model and optimizer checkpoint
    save_dir = config["Training_Directory"]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    latest_epoch, checkpoint = utils.latest_save(save_dir, device)

    if checkpoint is not None:

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f'Loaded epoch {latest_epoch} from checkpoint')

    
    epoch_widgets = [
        '[Epoch: ', progressbar.Counter(format='%3d'), f'/{epochs}, ',
        progressbar.Timer(), '] ',
        progressbar.Bar('█', left='|', right='|'),
        ' (', progressbar.AdaptiveETA(), ') '
        ]
    
    epoch_progress = progressbar.ProgressBar(maxval=epochs, widgets=epoch_widgets).start()

    #* TRAINING LOOP
    for epoch in range(latest_epoch+1, epochs):
        train_loss = train(model, train_dl, optimizer, loss, device)
        test_loss = test(model, test_dl, loss, device)

        if use_wandb:
            wandb.log({
                'Train' : {'Loss' : train_loss},
                'Test' : {'Loss' : test_loss},
                'Other' : {}
            }, step=epoch)

        
        #* Save every save_epoch epochs if it is specified. If it isn't, will only save at the end
        if save_epoch is not None and epoch % save_epoch == 0:
            torch.save({
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict()
            }, os.path.join(save_dir, f'model_epoch_{epoch}.pt'))

        epoch_progress.update(epoch+1)
    epoch_progress.finish()

    #* Save final epoch
    torch.save({
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict()
    }, os.path.join(save_dir, f'model_epoch_{epoch}.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg("--config", type=str, help="Path to config file if manually starting a new run")
    add_arg("--runid", type=str, help="The id of the wandb run to resume")
    add_arg("--from_sweep", action="store_true", help="Whether to expect to recieve the config from the wandb sweep")
    add_arg("--project", type=str, help="The W&B project that the run is in")
    add_arg("--epochs_override", type=int, help="Epochs to run for. Overrides the epochs parameter in the config.")
    add_arg("--save_epoch", type=int, default=None)
    add_arg("--no_wandb", action="store_true", help="Don't use or log to W&B")

    pargs = parser.parse_args()
    
    main(pargs)