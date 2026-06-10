#%%
import zstandard
import os
import json
import csv
from datetime import datetime
import numpy as np
import pandas as pd
from memes_base.datasets import CachedDataset
import progressbar
from transformers import CLIPProcessor, CLIPModel
import torch
from memes_base.image_processing import load_from_url, image_to_tensor, extract_text_from_image
from memes_base.utils import buildFromConfig
import requests
import time
from alive_progress import alive_bar, alive_it
# from safetensors.torch import save_file, safe_open
import lmdb
import pickle


def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
	chunk = reader.read(chunk_size)
	bytes_read += chunk_size
	if previous_chunk is not None:
		chunk = previous_chunk + chunk
	try:
		return chunk.decode()
	except UnicodeDecodeError:
		if bytes_read > max_window_size:
			raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
		# log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
		return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)

def chunk_bounds(l, n):
    #* l : length of object to split into chunks
    #* n : number of chunks to split it into
    bounds = [0]
    for chunk in range(n):
        if chunk < l%n:
            bounds.append(bounds[-1] + l//n + 1)
        else:
            bounds.append(bounds[-1] + l//n)

    return bounds

def in_bounds(i, process_chunks, chunk_bounds):
    #* Returns true if i is in the bounds for one of the chunks in process_chunks
    for chunk in process_chunks:
        if i > chunk_bounds[chunk] and i < chunk_bounds[chunk+1]:
            return True
        
    return False

def embed_text_and_images(texts, imgs, model, processor):

    inputs = processor(text=texts, images=imgs, return_tensors="pt", padding=True, truncation=True)

    outputs = model(**inputs)

    return torch.cat((outputs.text_embeds, outputs.image_embeds), dim=1)


def append_to_lmdb(lmdb_path, new_data):
    # Open the existing LMDB file in write mode
    env = lmdb.open(lmdb_path, map_size=1099511627776, create=True)
    with env.begin(write=True) as txn:
        for key in new_data:
            tensor = new_data[key]
            pickled = pickle.dumps(tensor)

            if isinstance(key, str):
                key = key.encode('utf-8')
            elif isinstance(key, int):
                key = str(key).encode('utf-8')
            
            txn.put(key, pickled)
    env.close()


def read_from_lmdb(lmdb_path, keys):
    # Open the existing LMDB file in read mode
    if not os.path.exists(lmdb_path):
        raise FileNotFoundError(f"LMDB file {lmdb_path} not found")

    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key in keys:
            if isinstance(key, str):
                key_binary = key.encode('utf-8')
            elif isinstance(key, int):
                key_binary = str(key).encode('utf-8')
            elif isinstance(key, bytes):
                key_binary = key

            value = cursor.get(key_binary)
            yield pickle.loads(value)

def get_keys_from_lmdb(lmdb_path):
    if not os.path.exists(lmdb_path):
        raise FileNotFoundError(f"LMDB file {lmdb_path} not found")
    
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
    return keys


class ZSTReader:
    def __init__(self, file_name):
        self.file_name = file_name
        
    def __iter__(self):
        self.file_handle = open(self.file_name, 'rb')
        self.reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(self.file_handle)
        self.buffer = ''
        return self

    def __next__(self):
        chunk = read_and_decode(self.reader, 2**27, (2**29)*2)
        if not chunk:
            self.reader.close()
            self.file_handle.close()
            raise StopIteration

        lines = (self.buffer + chunk).split("\n")
        
        self.buffer = lines[-1]

        return lines[:-1], self.file_handle.tell()
    
class RedditRawDataset(CachedDataset):
    def __init__(self, name, raw_files, save_dir=None, save=True, chunks=1, process_chunks=None, **kwargs):

        super().__init__(name, raw_files, save_dir, save, chunks, process_chunks, **kwargs)

    def save_file_name(self, chunk=None):
        return super().save_file_name(chunk, ".parquet")
        
    def process(self, chunks_to_process=None):
        # for raw_file in self.raw_files:
        #     fh = open(raw_file, 'rb')
        #     reader = zstandard.ZstdDecompressor().decompress(raw_file)
        #     fh.close()
        if len(self.raw_files) > 1:
            raise NotImplementedError("Cannot handle multiple files yet")
        
        reader = ZSTReader(self.raw_files[0])

        is_comment = 'comments' in self.raw_files[0]

        # ! THIS MAY BE VERY SLOW
        #* Counting blocks, since there's no easy way to find length
        for i, _ in enumerate(iter(reader)):
            if i % 40 == 0:
                print(i)
        n_blocks = i+1


        process_progress = progressbar.ProgressBar(maxval=n_blocks, widgets=self.processing_widgets).start()

        bounds = chunk_bounds(n_blocks, self.chunks)

        self.df = []

        bad_lines = 0

        line_num = 0

        for i, (block, bytes_read) in enumerate(iter(reader)):
            

            if not in_bounds(i, chunks_to_process, bounds):
                continue
            
            processed_lines = []
            
            for line in block:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                line_info = {}

                #* Store if it is a comment
                line_info['is_comment'] = is_comment

                #* Image url
                #! If there isn't an image, I think the url field is just a link to the post
                line_info['img_url'] = obj['url']

                #* Creation time: UTC
                line_info['created_utc'] = int(obj["created_utc"])

                #* Author
                line_info['author'] = obj['author']
                
                #* Upvotes (I think)
                line_info['score'] = int(obj['score'])

                #! May want to add more fields

                if is_comment:
                    #* Not the full link, omitting "https://www.reddit.com from the beginning"
                    line_info['link'] = f"/r/{obj['subreddit']}/comments/{obj['link_id'][3:]}/_/{obj['id']}/"

                    #* Leave title blank for comment
                    line_info['title'] = ''

                    #* Text of comment
                    line_info['text'] = obj['body']
                else:
                    #* Same as above link, but for a comment
                    line_info['link'] = obj["permalink"]

                    #* Title
                    line_info['title'] = obj['title']

                    #* Body text
                    line_info['text'] = obj['selftext']
                
                processed_lines.append(line_info)
            
            index = np.arange(line_num, line_num+len(block))

            line_num += len(block)

            processed_lines = pd.DataFrame(processed_lines, index=index)

            self.df.append(processed_lines)

            process_progress.update(i+1)

        process_progress.finish()

        self.df = pd.concat(self.df)
        print(self.df)
        print(f"Finished Reading File. Bad Lines: {bad_lines}")

    def save(self):
        super().save()
        #* Saves the chunks to parquet files  

        bounds = chunk_bounds(len(self.df), self.chunks)
        for chunk in self.process_chunks:
            chunk_min = bounds[chunk]
            chunk_max = bounds[chunk+1]
            print(f"Saving chunk {chunk} to file:", self.save_file_name(chunk))
            self.df.loc[chunk_min:chunk_max].to_parquet(self.save_file_name(chunk))
    
    def loadFromCache(self):
        self.df = pd.DataFrame()

        for chunk in self.process_chunks:
            chunk_df = pd.read_parquet(self.save_file_name(chunk))
            self.df = pd.concat([self.df, chunk_df], ignore_index=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
         return self.df.loc[idx]

class RedditEmbedDataset(CachedDataset):
    def __init__(self, raw_dset_cfg, embedding_model="openai/clip-vit-base-patch32", embedding_batch_size=64, **kwargs):
        #* Takes in a RedditRawDataset configuration, and extracts embeddings from it
        self.raw_dset_cfg = raw_dset_cfg

        self.embed_model = CLIPModel.from_pretrained(embedding_model)
        self.embed_processor = CLIPProcessor.from_pretrained(embedding_model)

        self.embedding_batch_size = embedding_batch_size

        args = raw_dset_cfg["args"]
        #* Run initialization to set instance variables right, but want to override typical cache checking, chunking
        super().__init__(args['name'] + "_embeds", args['raw_files'], save_dir=args['save_dir'], custom_save=True, **kwargs)

        #* Do custom save and loading process, built into the process function
        super().process()
        self.process()
        
    
    def save_file_name(self):
        return super().save_file_name(file_ext=".lmbd")

    def process(self):
        raw_dset = buildFromConfig(self.raw_dset_cfg)

        dset_index = raw_dset.df.index

        super().save()

        try:
            existing_keys = [int(key) for key in get_keys_from_lmdb(self.save_file_name())]
        except FileNotFoundError:
            existing_keys = []

        process_keys = [i for i in dset_index if i not in existing_keys]

        text_batch = []
        img_batch = []
        tracking_batch = []

        times = {"download": 0, "embed": 0, "OCR": 0, "save": 0}
        download_failures = 0
        HTTPErrors = 0
        ConnectionErrors = 0

        process_start = time.time()

        with alive_bar(len(process_keys), title="Processing Reddit Posts", force_tty=True, spinner="dots_waves") as bar:
            for i, key in enumerate(process_keys):
                post = raw_dset[key]
                # print(f"Processing post {i}, url: {post['img_url']}")

                process_post = True
                #! Check what happens if there's no image, or image is inaccesible
                try:
                    startTime = time.time()
                    # print(f"\nAttempting to download image from URL: {post['img_url']}")
                    img = load_from_url(post['img_url'])
                    times["download"] += time.time() - startTime
                except (requests.HTTPError) as e:
                    print(f"HTTP Error for URL {post['img_url']}: {str(e)}")
                    process_post = False
                    HTTPErrors += 1
                except requests.exceptions.ConnectionError as e:
                    print(f"Connection Error for URL {post['img_url']}: {str(e)}")
                    process_post = False
                    ConnectionErrors += 1
                except Exception as e:
                    print(f"Unexpected error for URL {post['img_url']}: {str(e)}")
                    process_post = False
                    HTTPErrors += 1

                if img is None:
                    print(f"Failed to load image from URL: {post['img_url']}. Post Link: reddit.com/{post['link']}")
                    process_post = False

                if process_post:
                    img_tensor = image_to_tensor(img)

                    body_text = post['text']
                    title_text = post['title']

                    startTime = time.time()
                    image_text = extract_text_from_image(img)
                    times["OCR"] += time.time() - startTime

                    #! Figure out best way to put text together so that the embedding model understands it
                    whole_text = f"{title_text}:\n{body_text}\nImage Caption: {image_text}"

                    text_batch.append(whole_text)
                    img_batch.append(img_tensor)
                    tracking_batch.append(key)
                else:
                    download_failures += 1
                
                bar()
                bar.text(f"Successes: {i+1-download_failures}, Download Failures: {download_failures}/{i+1}, HTTP Errors: {HTTPErrors}, Connection Errors: {ConnectionErrors}")

                #* Running embedding every batch_size posts, or at the end of the process
                if len(text_batch) != 0 and (len(text_batch) % self.embedding_batch_size == 0 or i == len(process_keys)-1):
                    #* Run embedding once every batch_size posts for speed
                    #* Then save to the LMDB file

                    startTime = time.time()
                    #! Eventually, what if a post doesn't have an image? Will need to change embed_text_and_images, prob not concatenating but taking mean??
                    embeds = embed_text_and_images(text_batch, img_batch, self.embed_model, self.embed_processor)
                    times["embed"] += time.time() - startTime

                    startTime = time.time()

                    save_dict = {tracking_batch[j]: embeds[j] for j in range(len(tracking_batch))}
                    append_to_lmdb(self.save_file_name(), save_dict)

                    times["save"] += time.time() - startTime

                    text_batch = []
                    img_batch = []
                    tracking_batch = []
                

        print(f"Finished Processing in {time.time() - process_start}s. {download_failures}/{len(process_keys)} posts failed to download images.",
              f"\nDownload Time: {times['download']}s, OCR Time: {times['OCR']}s, Embedding Time: {times['embed']}s, Save Time: {times['save']}s")

    def __getitem__(self, indices):
        #! MUCH FASTER IF ALL INDICES ARE PASSED AT ONCE

        #* Check if multiple indices have been passed
        if hasattr(indices, "__getitem__"):
            return read_from_lmdb(self.save_file_name(), indices)
        else:
            return read_from_lmdb(self.save_file_name(), [indices])[0]
            




#%%

# file_name = "reddit_data/reddit/subreddits23/memes_submissions.zst"

# ds = RedditDataset("test_reddit_dataset", file_name, save_dir='dataset_saves/reddit/', chunks=10)#, process_chunks=[0]) 



# %%