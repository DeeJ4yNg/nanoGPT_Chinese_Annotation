######################################################################################################################################################
#                                               Name: Model_GPT                                                                                      #
#                                               Comments Author: Devin Wu                                                                            #
#                                               Date: Jul, 6, 2023 11:05AM                                                                           #
#                                               Ref: http://pointborn.com/article/2022/2/18/1820.html                                                #
#                                                    https://blog.csdn.net/Mikeyboi/article/details/119522689                                        #
#                                                                                                                                                    #
######################################################################################################################################################


import math
import inspect
from dataclasses import dataclass
from pyclbr import Class
import torch
import torch.nn as nn
from torch.nn import functional as F

def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def relu(x):
    return x if x > 0 else 0

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        # 初始化权重，标准化公式x_hat = x-u/根号下方差a^2-epsilon加入可学习参数，变成 output = alpha * x_hat + beta, 其中alpha相当于权重，beta相当于偏置。
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        # F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape)--自定义的weight, self.bias.expand(normalized_shape))--自定义的bias
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        #确保词嵌入的总维度数可以整除head的数量，不然每个head的维度不一样
        assert config.n_embd % config.n_head == 0
        #输入维度投影到三倍自身的维度，后续分别为q,k,v操作
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        #定义输出的线性层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        #定义dropout和其他属性
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # B，T，C 分别表示 batch_size, sequence_length, embedding_dimension
        B, T, C = x.size() 
        # 定义q,k,v
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        # 更改q,k,v的维度，变成(batch_size, n_head, sequence_length, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 根据公式计算self attention，维度变化：(B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # qk^T/根号d(k的最后一维即head_size也就是论文中的d)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # 添加decoder的mask矩阵，右上三角为负无穷防止训练过程中后面的token信息泄露
        att = F.softmax(att, dim=-1) # Softmax(qk^T/根号d)
        att = self.attn_dropout(att)
        y = att @ v # Softmax(qk^T/根号d)V, 维度变化：(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 输出y维度重组，y.transpose(1, 2) -> (B, T, nh, hs), .view(B, T, C)-> (B, T, nh X hs) = (B,T,C)

        # 输出y过线性层再dropout，最后return出去
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，从输入维度投影到4倍输入维度的隐藏层，再输出与输入相同的维度
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = gelu(x) # 高斯误差线性单元
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    # 把上面定义的模块组合成一个block，以便于之后堆叠，这里就实现了论文里的GPT结构图
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x)) # 加上Residual
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # Token size
    vocab_size: int = 50304 # 词汇表大小
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # LayerNorm 和线性层的偏置，为了快选择了false

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 断言一定得有Vocab_size 和 Block_size
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # 构造一个transformer的字典用于方便后续堆叠
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Word Embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # Position Embedding, 这里直接用nn.Embedding来构造可学习的位置嵌入，而没有使用更新潮的RoPE等POS
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Block 根据超参layer数来堆叠
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # TODO 这是什么鬼？
        self.transformer.wte.weight = self.lm_head.weight # 与上面的lm_head参数共享

        # 行起_init_weights来初始化参数
        self.apply(self._init_weights)

        # 根据GPT2论文，residual projections的权重参数应按如下初始化
        for pn, p in self.named_parameters():
            print(pn,p)
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        # 获取模型所有参数量，位置嵌入参数会被减去
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        # 如果module是Linear Layer，就按均值为0标准差为0.02的高斯分布初始化权重
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # 如果是Linear Layer有偏置，则初始化为0
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # 如果module是Embedding Layer，就按均值为0标准差为0.02的高斯分布初始化权重
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # 前向过程
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size() # b = batch size, t = block size(输入的长度)
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t) tensor[0,1,2,3...t]

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # word token embedding (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embedding (1, t, n_embd) TODO 考虑换成RoPE吧。。那不就变成LLAMA了么呵、
        x = self.transformer.drop(tok_emb + pos_emb) # 词嵌入和位置嵌入相加，再过dropout
        # 输入x流经block 堆叠
        for block in self.transformer.h:
            x = block(x)
        
        # 从堆叠出来后再过LayerNorm
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 最后x经过lm输出头(一个Linear Layer)输出logits。如果有加训练目标(ground truth)，就算上loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 如果是推理阶段，则取最后一字的分数
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 遍历模型参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤冻结的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 对所有2维及以上的参数作衰减
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        # 构建衰减条件
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # 构建AdamW优化器
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        return optimizer
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # 把过长的输入idx剪成block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 剪过的idx流入model，得出分数
            logits, _ = self(idx_cond)
            # 取最后一个分数，并根据温度参数缩放
            logits = logits[:, -1, :] / temperature
            # 若有topk，则从topk中采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 在序列后cat上采样的idx后继续循环
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
