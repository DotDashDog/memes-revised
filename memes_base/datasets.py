import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np

def save_file_name(dir, name, chunk=None):
    if chunk is None:
        return os.path.join(dir, name + ".npz") #! May want to save as other filetype???
    else:
        return os.path.join(dir, name + f"_{chunk}.npz") #! See above
    
def get_chunk_indices(length, chunks):
    if chunks > 1:
        split_borders = length//chunks * np.arange(1, chunks)
        chunk_indices = np.split(np.arange(length), split_borders)
    else:
        chunk_indices = [np.arange(length)]
    return chunk_indices

class DummyDataset(Dataset):
    #! Currently does not allow prebatching, which may be needed for faster loading
    def __init__(self, name, raw_file, save_dir=None, save=True, chunks=1, process_chunks=None, **kwargs):
        self.name = name
        self.raw_file = raw_file
        self.save_dir = save_dir
        self.save_to_disk = save
        self.chunks = chunks
        self.process_chunks = process_chunks

        print("Unused arguments upon creation of dataset:", kwargs)

        if isinstance(self.process_chunks, int):
            self.process_chunks = [self.process_chunks]
        elif self.process_chunks is None:
            self.process_chunks = list(range(self.chunks))

        if self.save_to_disk: 
            if not self.hasCache():
                self.process()
                self.save()
            else:
                self.loadFromCache()
        else:
            self.process()

    
    def hasCache(self):
        """Checks if the dataset already has a processed copy saved to disk

        Returns:
            boolean: Whether a cache of the dataset already exists
        """
        if self.chunks == 1:
            #* Simple case if there's only one chunk (aka not splitting the dataset into chunks)
            file_path = save_file_name(self.save_dir, self.name)
            if os.path.exists(file_path):
                print(f"Cache of {self.name} found at {file_path}")
                return True
            else:
                print(f"Cache of {self.name} not found at {file_path}")
                return False
        else:
            #* Check chunk-by-chunk. If any of the chunks that it is supposed to be processing don't exist, return False.
            for chunk in self.process_chunks:
                file_path = save_file_name(self.save_dir, self.name, chunk)
                if os.path.exists(file_path):
                    print(f"Cache of {self.name}, chunk {chunk} found at {file_path}")
                else:
                    print(f"Cache of {self.name}, chunk {chunk} not found at {file_path}")
                    return False
            return True

    #! WIP Functions. Really depend on how exactly our dataset is set up.
    def process(self):
        df = pd.read_csv(self.raw_file)

        chunk_indices = get_chunk_indices(len(df), self.chunks)
        self.chunk_Xs = {}
        self.chunk_ys = {}

        for chunk in self.process_chunks:
            indices = chunk_indices[chunk]
            print(f"Processing chunk {chunk} from index {indices[0]} to {indices[-1]}")

            chunk_df = df.loc[indices]
            #? I don't think I need to add a mask function for train-test splitting. torch.utils.data.random_split will do that for me

            self.chunk_Xs[chunk] = np.array(chunk_df[['x0', 'x1', 'x2', 'x3', 'x4']])
            self.chunk_ys[chunk] = np.array(chunk_df['y'])

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.chunks == 1:
            np.savez(save_file_name(self.save_dir, self.name), X=self.chunk_Xs[0], y=self.chunk_ys[0])
        else:
            for chunk in self.process_chunks:
                np.savez(save_file_name(self.save_dir, self.name, chunk), X=self.chunk_Xs[chunk], y=self.chunk_ys[chunk])

    def loadFromCache(self):
        self.chunk_Xs = {}
        self.chunk_ys = {}
        if self.chunks == 1:
            npzfile = np.load(save_file_name(self.save_dir, self.name))
            self.chunk_Xs[0] = npzfile["X"]
            self.chunk_ys[0] = npzfile["y"]
        else:
            for chunk in self.process_chunks:
                npzfile = np.load(save_file_name(self.save_dir, self.name, chunk))
                self.chunk_Xs[chunk] = npzfile["X"]
                self.chunk_ys[chunk] = npzfile["y"]

    #* Functions needed for torch Dataset class
    def __len__(self):
        return np.sum([self.chunk_ys[chunk].shape[0] for chunk in self.process_chunks])
    
    def __getitem__(self, idx):
        if self.chunks == 1:
            return self.chunk_Xs[0][idx], self.chunk_ys[0][idx]
        else:
            current_idx = idx
            for chunk in self.process_chunks:
                chunk_length = self.chunk_ys[chunk].shape[0]
                if current_idx < chunk_length:
                    return self.chunk_Xs[chunk][current_idx], self.chunk_ys[chunk][current_idx]
                
                current_idx -= chunk_length
            
            raise IndexError("Index out of bounds")