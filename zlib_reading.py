#%%
import zstandard
import os
import json
import csv
from datetime import datetime
import numpy as np
import pandas as pd
from memes_base.datasets import CachedDataset

# def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
# 	chunk = reader.read(chunk_size)
# 	bytes_read += chunk_size
# 	if previous_chunk is not None:
# 		chunk = previous_chunk + chunk
# 	try:
# 		return chunk.decode()
# 	except UnicodeDecodeError:
# 		if bytes_read > max_window_size :
# 			raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
# 		return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)

# def in_bounds(pos, split_borders, chunks):
#     n_chunks = len(split_borders)+1
    
#     if 0 in chunks:
#         if pos < split_borders[0]:
#             return True
#         else:
#             return False
#     elif n_chunks-1 in chunks:
#         if pos >= split_borders[-1]:
#             return True
#         else:
#             return False
    
#     if n_chunks <= 2:
#         return False
    
#     for chunk in chunks:
#         # print(split_borders[chunk-1], split_borders[chunk])
#         if pos >= split_borders[chunk-1] and pos < split_borders[chunk]:
#             return True
    
#     return False

def chunk_bounds(l, n):
    bounds = [0]
    for chunk in range(n):
        if chunk < l%n:
            bounds.append(bounds[-1] + l//n + 1)
        else:
            bounds.append(bounds[-1] + l//n)

    return bounds

class ZSTReader:
    def __init__(self, file_name):
        self.file_name = file_name
        
    def __iter__(self):
        self.file_handle = open(self.file_name, 'rb')
        self.reader = zstandard.ZstdDecompressor().read_to_iter(self.file_handle) #zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(self.file_handle)
        self.buffer = ''
        return self

    def __next__(self):
        # chunk = read_and_decode(self.reader, 2**27, (2**29)*2)
        chunk = next(self.reader).decode()
        if not chunk:
            self.reader.close()
            self.file_handle.close()
            raise StopIteration

        lines = (self.buffer + chunk).split("\n")
        
        self.buffer = lines[-1]

        return lines[:-1], self.file_handle.tell()
    
class RedditDataset(CachedDataset):
    def __init__(self, name, raw_files, save_dir=None, save=True, chunks=1, process_chunks=None, **kwargs):
        # self.file_sizes = [os.stat(file_path).st_size for file_path in self.raw_files]
        # self.total_size = sum(self.file_sizes)

        # split_lows = self.total_size//chunks * np.arange(0, chunks)
        # split_highs = self.total_size//chunks * (np.arange(0, chunks)+1)
        # self.chunk_boundaries = np.stack(split_lows, split_highs)

        super().__init__(name, raw_files, save_dir, save, chunks, process_chunks, **kwargs)

    def save_file_name(self, chunk=None):
        return super().save_file_name(chunk, ".parquet")
        
    def process(self):
        # for raw_file in self.raw_files:
        #     fh = open(raw_file, 'rb')
        #     reader = zstandard.ZstdDecompressor().decompress(raw_file)
        #     fh.close()

        reader = ZSTReader(self.raw_files[0])

        is_comment = 'comments' in self.raw_files[0]

        # ! THIS MAY BE VERY SLOW
        # n_blocks = sum(1 for _ in iter(reader))

        self.df = pd.DataFrame()

        bad_lines = 0

        line_num = 0

        for (block, bytes_read) in iter(reader):
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
        
        print(f"Finished Reading File. Bad Lines: {bad_lines}")

    def save(self):
        #* Saves the chunks to parquet files
        bounds = chunk_bounds(len(self.df), self.chunks)
        for chunk in self.process_chunks:
            chunk_min = bounds[chunk]
            chunk_max = bounds[chunk+1]

            self.df.loc[chunk_min:chunk_max].to_parquet(self.save_file_name(chunk))
    
    def loadFromCache(self):
        self.df = pd.DataFrame()

        for chunk in self.process_chunks:
            self.df.append(pd.read_parquet(self.save_file_name(chunk)))

#%%

file_name = "/Users/linusupson/Downloads/reddit/subreddits23/memes_submissions.zst"

ds = RedditDataset("test_reddit_dataset", file_name, save_dir='dataset_saves/reddit', chunks=10, process_chunks=[0]) 

# %%

# is_comment = 'comments' in file_name

# reader = ZSTReader(file_name)

# # reader_iter = iter(reader)

# # chunk = next(iter(reader))

# processed_lines = []
# bad_lines = 0

# blocks_per_file = 5

# df = pd.DataFrame()

# for i, (block, bytes_read)in enumerate(iter(reader)):
#     for line in block:
#         try:
#             obj = json.loads(line)
#         except json.JSONDecodeError:
#             bad_lines += 1
#             continue
#         line_info = {}

#         #* Store if it is a comment
#         line_info['is_comment'] = is_comment

#         #* Image url
#         #! If there isn't an image, I think the url field is just a link to the post
#         line_info['img_url'] = obj['url']

#         #* Creation time: UTC
#         line_info['created_utc'] = int(obj["created_utc"])

#         #* Author
#         line_info['author'] = obj['author']
        
#         #* Upvotes (I think)
#         line_info['score'] = int(obj['score'])

#         #! May want to add more fields

#         if is_comment:
#             #* Not the full link, omitting "https://www.reddit.com from the beginning"
#             line_info['link'] = f"/r/{obj['subreddit']}/comments/{obj['link_id'][3:]}/_/{obj['id']}/"

#             #* Leave title blank for comment
#             line_info['title'] = ''

#             #* Text of comment
#             line_info['text'] = obj['body']
#         else:
#             #* Same as above link, but for a comment
#             line_info['link'] = obj["permalink"]

#             #* Title
#             line_info['title'] = obj['title']

#             #* Body text
#             line_info['text'] = obj['selftext']
        
#         processed_lines.append(line_info)
    
#     df.append(processed_lines)

#     if (i+1)%blocks_per_file == 0:
#         #* Save the current chunk of data
#         pass





# %%