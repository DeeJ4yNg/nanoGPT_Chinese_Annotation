import os
import pickle
import numpy as np

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

# 打开dataset，内容存入data
with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# 所有字符去重当成vocab size，实作中会用BPE之类的算法来分token
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

stoi = { ch:i for i,ch in enumerate(chars) } # 字符：索引
itos = { i:ch for i,ch in enumerate(chars) } # 索引：字符
def encode(s):
    return [stoi[c] for c in s] # encoder: 字符变索引
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: 索引变字符

# train test splits 9：1
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# training data 和 validation data 均 encode 成索引
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# 保存成bin文件
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# 保存下列信息到pkl文件，方便后续训练读取vocab size
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

#Python中的 pickle 模块实现了基本的数据序列与反序列化。
#序列化对象可以在磁盘上保存对象，并在需要的时候读取出来。
#任何对象都可以执行序列化操作。