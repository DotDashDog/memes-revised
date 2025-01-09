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
         return self.df.iloc[idx]

class RedditEmbedDataset(CachedDataset):
    def __init__(self, raw_dset_cfg, embedding_model="openai/clip-vit-base-patch32", embedding_batch_size=64, **kwargs):
        #* Takes in a RedditRawDataset configuration, and extracts embeddings from it
        self.raw_dset_cfg = raw_dset_cfg

        self.embed_model = CLIPModel.from_pretrained(embedding_model)
        self.embed_processor = CLIPProcessor.from_pretrained(embedding_model)

        self.embedding_batch_size = embedding_batch_size

        args = raw_dset_cfg["args"]
        super().__init__(args['name'] + "_embeds", args['raw_files'], args['save_dir'], args['save'], **kwargs) #chunks, process_chunks,
        #* 
    
    def save_file_name(self, chunk=None):
        return super().save_file_name(chunk, ".pt")

    def process(self, chunks_to_process=None):
        raw_dset = buildFromConfig(self.raw_dset_cfg)

        if chunks_to_process is None:
            chunks_to_process = self.process_chunks

        #! Maybe switch raw dataset to being an iterable??
        #! NEED TO MAKE WORK WITH CHUNKS FOR LARGER DATASETS
        text_batch = []
        img_batch = []

        tracking = []

        embed_batches = []
        max_length = 100 #! IMPORTANT: SWITCH BACK TO len(raw_dset)
        for i in range(max_length): 
            post = raw_dset[i]
            print(f"Processing post {i}, url: {post['img_url']}")

            process_post = True
            #! Check what happens if there's no image, or image is inaccesible
            try:
                img = load_from_url(post['img_url'])
            except requests.HTTPError as e:
                print("HTTP Error:", e)
                process_post = False
            if img is None:
                print("Invalid Image URL")
                process_post = False

            if process_post:
                img_tensor = image_to_tensor(img)

                body_text = post['text']
                title_text = post['title']
                image_text = extract_text_from_image(img)

                #! Figure out best way to put text together
                whole_text = f"{title_text}:\n{body_text}\nImage Caption: {image_text}"

                text_batch.append(whole_text)
                img_batch.append(img_tensor)
                tracking.append(i)

            if len(text_batch) != 0 and (len(text_batch) % self.embedding_batch_size == 0 or i == max_length-1):
                #* Run embedding once every batch_size posts for speed
                print(len(text_batch), "posts processed. Running Embedding")

                # for w in text_batch:
                #     print(w.replace("\n", " "))

                embeds = embed_text_and_images(text_batch, img_batch, self.embed_model, self.embed_processor)

                embed_batches.append(embeds)

                text_batch = []
                img_batch = []
        
        
        self.tracking = torch.tensor(tracking)
        self.embeddings = torch.cat(embed_batches, dim=0)


    def save(self):
        super().save()
        torch.save({"embeddings": self.embeddings, "tracking": self.tracking}, self.save_file_name(0)) #! Change when chunking is added

    def loadFromCache(self):
        data = torch.load(self.save_file_name(0)) #! Change when chunking is added
        self.embeddings = data["embeddings"]
        self.tracking = data["tracking"]
    
    





#%%

# file_name = "reddit_data/reddit/subreddits23/memes_submissions.zst"

# ds = RedditDataset("test_reddit_dataset", file_name, save_dir='dataset_saves/reddit/', chunks=10)#, process_chunks=[0]) 



# %%