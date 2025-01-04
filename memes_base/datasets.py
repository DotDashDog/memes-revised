import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import progressbar
    
def get_chunk_indices(length, chunks):
    """Get an array of arrays of indices. The ith element of the outer array is the set of indices that are in the ith chunk of the dataset

    Args:
        length (int): the length of the dataset
        chunks (int): the number of chunks to split the dataset into

    Returns:
        list(np.array): The indices of each chunk. The last chunk may be smaller
    """
    if chunks > 1:
        split_borders = length//chunks * np.arange(1, chunks)
        chunk_indices = np.split(np.arange(length), split_borders)
    else:
        chunk_indices = [np.arange(length)]
    return chunk_indices

class CachedDataset(Dataset):
    #* A template class, with the common functions that all of our datasets should need.
    #! Currently does not allow prebatching, which may be needed for faster loading
    def __init__(self, name, raw_files, save_dir=None, save=True, chunks=1, process_chunks=None, **kwargs):
        """Intitializes the dataset, including processing, chunking, and creating save files

        Args:
            name (string): the name of the dataset (to be used in save files)
            raw_files (string): the paths to the files the raw data is coming from
            save_dir (strinng, optional): the directory to save the processed data in. Defaults to None.
            save (bool, optional): whether to save the dataset. Defaults to True.
            chunks (int, optional): how many chunks to split the dataset into. Defaults to 1.
            process_chunks (list, optional): which chunks should be included in this instance. Defaults to None.
        """
        self.name = name
        self.raw_files = raw_files
        self.save_dir = os.path.join(save_dir, self.name)
        self.save_to_disk = save
        self.chunks = chunks
        self.process_chunks = process_chunks

        self.processing_widgets = [
            '[Preprocessing: ', progressbar.Percentage(), ', ',
            progressbar.Timer(), '] ',
            progressbar.Bar('█', left='|', right='|'),
            ' (', progressbar.AdaptiveETA(), ') '
        ]

        print("Unused arguments upon creation of dataset:", kwargs)
        if isinstance(self.raw_files, str):
            self.raw_files = [self.raw_files]
        
        if isinstance(self.process_chunks, int):
            self.process_chunks = [self.process_chunks]
        elif self.process_chunks is None:
            self.process_chunks = list(range(self.chunks))

        if self.save_to_disk: 
            chunks_to_process = [chunk for chunk in self.process_chunks if not self.hasCache(chunk)]
            
            if len(chunks_to_process) != 0:
                print("Processing input for chunks:", chunks_to_process)
                self.process(chunks_to_process)
                self.save()
            else:
                self.loadFromCache()
        else:
            self.process()

    def save_file_name(self, chunk=None, file_ext=".bin"):
        """Creates the path to a dataset save file

        Args:
            dir (string): the directory the file is to be stored in
            name (string): the base name of the save file
            chunk (int, optional): The chunk of the dataset to be stored in the file. Defaults to None, omitting the chunk number.

        Returns:
            string: the path to the file
        """
        if chunk is None:
            return os.path.join(self.save_dir, self.name + file_ext)
        else:
            return os.path.join(self.save_dir, self.name + f"_{chunk}{file_ext}")

    def hasCache(self, chunk=None):
        """Checks if the dataset already has a processed copy saved to disk

        Args:
            chunk (int, optional): The chunk of the dataset to check for. Defaults to None, checking for the entire dataset.

        Returns:
            boolean: Whether a cache of the dataset already exists
        """
        if chunk is not None:
            file_path = self.save_file_name(chunk)
            if os.path.exists(file_path):
                print(f"Cache of {self.name}, chunk {chunk} found at {file_path}")
                return True
            else:
                print(f"Cache of {self.name}, chunk {chunk} not found at {file_path}")
                return False
            
        #* Otherwise, check all chunks and return False if any are missing
            
        if self.chunks == 1:
            #* Simple case if there's only one chunk (aka not splitting the dataset into chunks)
            file_path = self.save_file_name()
            if os.path.exists(file_path):
                print(f"Cache of {self.name} found at {file_path}")
                return True
            else:
                print(f"Cache of {self.name} not found at {file_path}")
                return False
        else:
            #* Check chunk-by-chunk. If any of the chunks that it is supposed to be processing don't exist, return False.
            for chunk in self.process_chunks:
                file_path = self.save_file_name(chunk)
                if os.path.exists(file_path):
                    print(f"Cache of {self.name}, chunk {chunk} found at {file_path}")
                else:
                    print(f"Cache of {self.name}, chunk {chunk} not found at {file_path}")
                    return False
            return True
    
    #* Currently undefined functions
    def process(self, chunks=None):
        pass

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def loadFromCache(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass

class DummyDataset(CachedDataset):
    #* This is a dummy dataset class that loads randomly generated data from a .csv file
    #* But it does most of the stuff our actual dataset class will need to do, so hopefully we won't need to make a ton of changes

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dtype = torch.float32
        
        if len(self.raw_files) != 1:
            raise ValueError("Only one raw file is allowed for DummyDataset")

    def save_file_name(self, chunk=None):
        return super().save_file_name(chunk, ".npz")

    #! WIP Functions. Really depend on how exactly our dataset is set up.
    def process(self):
        """Loads and processes the raw data, keeps it in instance variables
        """
        df = pd.read_csv(self.raw_files[0])

        chunk_indices = get_chunk_indices(len(df), self.chunks)
        self.chunk_Xs = {}
        self.chunk_ys = {}

        for chunk in self.process_chunks:
            indices = chunk_indices[chunk]
            print(f"Processing chunk {chunk} from index {indices[0]} to {indices[-1]}")

            chunk_df = df.loc[indices]
            #? I don't think I need to add a mask function for train-test splitting. torch.utils.data.random_split will do that for me

            self.chunk_Xs[chunk] = torch.tensor(np.array(chunk_df[['x0', 'x1', 'x2', 'x3', 'x4']]), dtype=self.dtype)
            self.chunk_ys[chunk] = torch.tensor(np.array(chunk_df[['y',]]), dtype=self.dtype)

    def save(self):
        """Saves the processed data stored in the dataset's instance variables
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.chunks == 1:
            np.savez(self.save_file_name(), X=self.chunk_Xs[0].numpy(), y=self.chunk_ys[0].numpy())
        else:
            for chunk in self.process_chunks:
                np.savez(self.save_file_name(chunk), X=self.chunk_Xs[chunk].numpy(), y=self.chunk_ys[chunk].numpy())

    def loadFromCache(self):
        """Loads saved processed data from the dataset's save files
        """
        self.chunk_Xs = {}
        self.chunk_ys = {}
        if self.chunks == 1:
            npzfile = np.load(self.save_file_name())
            self.chunk_Xs[0] = torch.tensor(npzfile["X"], dtype=self.dtype)
            self.chunk_ys[0] = torch.tensor(npzfile["y"], dtype=self.dtype)
        else:
            for chunk in self.process_chunks:
                npzfile = np.load(self.save_file_name(chunk))
                self.chunk_Xs[chunk] = torch.tensor(npzfile["X"], dtype=self.dtype)
                self.chunk_ys[chunk] = torch.tensor(npzfile["y"], dtype=self.dtype)

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
        
