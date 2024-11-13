import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from torch.utils.data import Sampler
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
from transformers import AutoProcessor
from llava.model import *
import pdb
from torch.cuda.amp import autocast
import json
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in
                    get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i: i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i: i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]



# class LabelSmootherGA_GD:
#     epsilon: float = 0.1
#     ignore_index: int = -100
#     sensitive_tokens: list = None
#     tokenizer: object = None

#     def __init__(self, epsilon=0.1, ignore_index=-100, sensitive_tokens=None, tokenizer=None):
#         self.epsilon = epsilon
#         self.ignore_index = ignore_index
#         self.sensitive_tokens = sensitive_tokens if sensitive_tokens is not None else []
#         self.tokenizer = tokenizer

#     def __call__(self, model_output, labels, shift_labels=False):
#         logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
#         if shift_labels:
#             logits = logits[..., :-1, :].contiguous()
#             labels = labels[..., 1:].contiguous()

#         log_probs = -nn.functional.log_softmax(logits, dim=-1)
#         if labels.dim() == log_probs.dim() - 1:
#             labels = labels.unsqueeze(-1)

#         padding_mask = labels.eq(self.ignore_index)
#         labels = torch.clamp(labels, min=0)
#         nll_loss = log_probs.gather(dim=-1, index=labels)
#         smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

#         nll_loss.masked_fill_(padding_mask, 0.0)
#         smoothed_loss.masked_fill_(padding_mask, 0.0)

#         # 识别敏感 token 的位置
#         sensitive_mask = torch.zeros_like(labels, dtype=torch.bool)
#         for i, label_seq in enumerate(labels):
#             for j, label in enumerate(label_seq):
#                 token = self.tokenizer.decode([label.item()])
#                 if token in self.sensitive_tokens:
#                     sensitive_mask[i, j] = True

#         # 对敏感 token 使用梯度上升 (GA)，其余使用梯度下降 (GD)
#         ga_loss = torch.where(sensitive_mask, -nll_loss, nll_loss)

#         # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
#         num_active_elements = padding_mask.numel() - padding_mask.long().sum()
#         nll_loss = ga_loss.sum() / num_active_elements
#         smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])

#         return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


# def save_token_probabilities(logits, labels, tokenizer, epoch, step, filename="probabilities.json"):
#     """保存非敏感token的概率到文件并加入epoch信息"""
#     sensitive_tokens = ["donald", "trump"]
#     prob_q = F.softmax(logits, dim=-1)

#     # 确保logits和labels的sequence_length一致
#     logits = logits[:, :labels.size(0), :]  # 切片，保证logits的长度与labels匹配
#     print(f"Adjusted logits shape: {logits.shape}")
    
#     token_probs = {}

#     for i, label in enumerate(labels):
#         if label.item() < 0:  # 忽略 padding
#             # print(f"Skipping padding label {label.item()}")
#             continue
#         try:
#             decoded_token = tokenizer.decode([label.item()])
#             if decoded_token not in sensitive_tokens:  # 仅保存非敏感token
#                 token_prob = prob_q[0, i, label.item()].item()
#                 token_probs[decoded_token] = {
#                     "probability": token_prob,  # 只取batch第一个样本
#                     "epoch": epoch,
#                     "step": step
#                 }
#         except IndexError as e:
#             # print(f"Skipping label {label.item()} due to index error: {e}")
#             continue

#     # 将概率写入文件（以JSON格式）
#     with open(filename, "a") as f:
#         f.write(json.dumps(token_probs) + "\n")  # 每个step写一行

def save_token_probabilities(logits, labels, tokenizer, epoch, step, filename="probabilities.json"):
    """保存非敏感token的概率到文件并加入epoch信息"""
    sensitive_tokens = ["Donald", "Trump"]
    prob_q = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    # 确保logits和labels的sequence_length一致
    logits = logits[:, :labels.size(0), :]  # 切片，保证logits的长度与labels匹配
    print(f"Adjusted logits shape: {logits.shape}")
    
    token_probs = {}
    processed_values = []
    sensitive_values = []

    for i, label in enumerate(labels):
        if label.item() < 0:  # 忽略 padding
            # print(f"Skipping padding label {label.item()}")
            continue
        try:
            decoded_token = tokenizer.decode([label.item()])
            if decoded_token not in sensitive_tokens:  # 仅保存非敏感token
                token_prob = log_probs[0, i, label.item()].item()
                processed_value = -token_prob
                if processed_value != 0: 
                    processed_value = 1 / processed_value

                processed_values.append(processed_value)  # 存储非敏感 token 的值

                token_probs[decoded_token] = {
                    "probability": token_prob,  # 只取batch第一个样本
                    "epoch": epoch,
                    "step": step
                }
            if  decoded_token in sensitive_tokens:  # 仅保存敏感token
                token_prob = log_probs[0, i, label.item()].item()
                processed_value = -token_prob
                if processed_value != 0: 
                    processed_value = 1 / processed_value
                sensitive_values.append(processed_value)

        except IndexError as e:
            # print(f"Skipping label {label.item()} due to index error: {e}")
            continue
    
    non_sensitive_sum = sum(processed_values)
    sensitive_sum = sum(sensitive_values)
    
    # print("n_sum",non_sensitive_sum)
    # print("sum",sensitive_sum)
    # pdb.set_trace()


    # 将概率写入文件（以JSON格式）
    with open(filename, "a") as f:
        f.write(json.dumps(token_probs) + "\n")  # 每个step写一行


class LabelSmootherGA_GD:
    epsilon: float = 0.1
    ignore_index: int = -100
    sensitive_tokens: list = None
    tokenizer: object = None

    def __init__(self, epsilon=0.1, ignore_index=-100, sensitive_tokens=None, tokenizer=None):
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.sensitive_tokens = sensitive_tokens if sensitive_tokens is not None else []
        self.tokenizer = tokenizer

    def __call__(self, model_output, labels, shift_labels=False):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # # 识别敏感 token 的位置
        # sensitive_mask = torch.zeros_like(labels, dtype=torch.bool)
        # for i, label_seq in enumerate(labels):
        #     for j, label in enumerate(label_seq):
        #         token = self.tokenizer.decode([label.item()])
        #         if token in self.sensitive_tokens:
        #             sensitive_mask[i, j] = True

        # # 对敏感 token 使用梯度上升 (GA)，其余使用梯度下降 (GD)
        # ga_loss = torch.where(sensitive_mask, -nll_loss, nll_loss)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])

        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss




class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
            self,
            batch_size: int,
            world_size: int,
            lengths: Optional[List[int]] = None,
            generator=None,
            group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size,
                                                          generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size,
                                                 generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def __init__(self, second_model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.second_model = second_model_path

        self.second_model, _, _, _ = deepspeed.initialize(model=self.second_model,
                                                          model_parameters=self.second_model.parameters(),
                                                          config='/data2/dmz/llava_test/LLaVA-main/scripts/zero3.json')
        self.second_model.eval()

    def compute_kl_loss(self, model, batch, device):
        temp = batch["input_ids"] < 0
        batch["input_ids"][temp] = 0

        normal_outputs = model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].int().to(device),
            labels=batch["labels"].to(device),
        )

        self.second_model.to(device)
        with torch.no_grad():
            pretrained_outputs = self.second_model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].int().to(device),
                labels=batch["labels"].to(device),
            )

        prob_p = F.softmax(pretrained_outputs.logits, dim=-1)
        prob_q = F.softmax(normal_outputs.logits, dim=-1)

        kl_loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

        return kl_loss, normal_outputs.loss


    def apply_dp_noise(self, sensitive_params, noise_scale=0.1):
        """向敏感参数的梯度中添加差分隐私噪声"""
        for name, grad in sensitive_params.items():
            noise = torch.normal(0, noise_scale, size=grad.size()).to(grad.device)
            sensitive_params[name] += noise

    # def training_step(self, model, inputs):
    #     model.train()
    #     device = next(model.parameters()).device

    #     # 计算KL loss和其他损失
    #     kl_loss, another_loss = self.compute_kl_loss(model, inputs, device)

    #     # 识别敏感参数
    #     sensitive_params = self.identify_sensitive_parameters(model, inputs, device)

    #     # 向敏感参数中注入差分隐私噪声
    #     self.apply_dp_noise(sensitive_params, noise_scale=0.1)

    #     print(len(sensitive_params))
    #     # for i in sensitive_params.keys():
    #     #     print(i,sensitive_params[i].shape)
        
    #     # pdb.set_trace()
    #     # 根据采样结果调整梯度
    #     for name, param in model.named_parameters():
    #         if param.grad is not None:  # 确保参数的grad不为空
    #             if name in sensitive_params and param.grad.size() == sensitive_params[name].size():
    #                 # print(name)
    #                 param.grad.copy_(sensitive_params[name])  # 用带噪声的梯度替换
    #             # else:
    #                 # print(f"Skipping gradient update for {name} due to size mismatch")

    #     combined_loss = kl_loss - 0.8 * another_loss

    #     if self.args.n_gpu > 1:
    #         combined_loss = combined_loss.mean()

    #     self.accelerator.backward(combined_loss)
        
    #     return combined_loss.detach()
    # def identify_sensitive_parameters(self, model, batch, device, threshold=0.01):
    #     """只识别LoRA层的敏感参数，基于梯度的大小"""
    #     model.zero_grad()
    #     normal_outputs = model(
    #         batch["input_ids"].to(device),
    #         attention_mask=batch["attention_mask"].int().to(device),
    #         labels=batch["labels"].to(device),
    #     )
    #     normal_outputs.loss.backward()

    #     sensitive_params = {}
    #     for name, param in model.named_parameters():
    #         if 'lora' in name and param.grad is not None:  # 只考虑LoRA层的权重
    #             grad_norm = param.grad.norm()
    #             if grad_norm > threshold:  # 超过阈值则视为敏感参数
    #                 sensitive_params[name] = param.grad.clone()
    #                 # print(f"Identified sensitive parameter: {name} with grad size {param.grad.size()}")

    #     return sensitive_params

    def token_level_ga_gd_loss(self, model, batch, tokenizer, sensitive_tokens, device):
        """
        对于某些敏感 token 使用 GA，对于其他 token 使用 GD。
        :param model: HuggingFace 模型
        :param batch: 输入数据，包括 input_ids 和 labels
        :param tokenizer: 分词器
        :param sensitive_tokens: 需要应用 GA 的敏感 token 列表
        :param device: 设备（CPU/GPU）
        :return: 计算出的损失值
        """

        # 获取模型输出
        # temp = batch["input_ids"] < 0
        # batch["input_ids"][temp] = 0

        # outputs = model(
        #     batch["input_ids"].to(device),
        #     attention_mask=batch["attention_mask"].int().to(device),
        #     labels=batch["labels"].to(device),
        # )

        base_loss,outputs = self.compute_loss(model, batch,return_outputs=True)  
        logits = outputs.logits  # (batch_size, seq_length, vocab_size)
        print("base",base_loss)
        # HuggingFace 自动计算的 logits 和 loss
        # logits = outputs.logits  # (batch_size, sequence_length, vocab_size)
        # labels = batch["labels"]  # (batch_size, sequence_length)
        outputs_logits = logits

        # # 初始化总损失
        # total_loss = 0

        # 对每个 token 分别处理
        # print(labels)
        # for i, label_sequence in enumerate(labels):  # 对 batch 中每个样本遍历
        #     for j, label in enumerate(label_sequence):  # 对每个 token 遍历
        #         if label == -100:  # 跳过 padding
        #             # print(22)
                    
        #             continue

        #         # 解码当前 token
        #         token = tokenizer.decode([label])

        #         # 获取当前 token 的 logits
        #         token_logits = logits[i, j]

        #         # 如果是敏感 token，使用 GA（梯度上升）
        #         if token in sensitive_tokens:
        #             # 梯度上升，增加损失
        #             total_loss += -0.3*F.cross_entropy(token_logits.unsqueeze(0), label.unsqueeze(0))
        #             print(total_loss)
        #         # 否则使用 GD（梯度下降）
        #         else:
        #             # 常规交叉熵损失
        #             total_loss += F.cross_entropy(token_logits.unsqueeze(0), label.unsqueeze(0))

        return base_loss, outputs_logits


    # def training_step(self, model, inputs):
    #     model.train()
    #     device = next(model.parameters()).device

    #     # 定义敏感 token 列表
    #     sensitive_tokens = ["Donald", "Trump"]

    #     # 初始化 LabelSmootherGA_GD
    #     label_smoother = LabelSmootherGA_GD(
    #         epsilon=0.1,
    #         ignore_index=-100,
    #         sensitive_tokens=sensitive_tokens,
    #         tokenizer=self.tokenizer
    #     )

    #     # 获取模型输出
    #     temp = inputs["input_ids"] < 0
    #     inputs["input_ids"][temp] = 0

    #     outputs = model(
    #         inputs["input_ids"].to(device),
    #         attention_mask=inputs["attention_mask"].int().to(device),
    #         labels=inputs["labels"].to(device),
    #     )


    #     # 计算损失
    #     combined_loss = label_smoother(outputs, inputs["labels"])

    #     # 多 GPU 场景下平均损失
    #     if self.args.n_gpu > 1:
    #         combined_loss = combined_loss.mean()

    #     # 反向传播
    #     self.accelerator.backward(combined_loss)

    #     return combined_loss.detach()


    def training_step(self, model, inputs):
        model.train()
        device = next(model.parameters()).device

        # 获取敏感 token 列表
        sensitive_tokens = ["Elon", "Musk"]
        inputs = self._prepare_inputs(inputs)
        # 计算GA和GD结合的loss
        combined_loss, outputs_logits = self.token_level_ga_gd_loss(model, inputs, self.tokenizer, sensitive_tokens, device)
        logits = outputs_logits

        save_token_probabilities(logits, inputs["labels"].view(-1), self.tokenizer, int(self.state.epoch),self.state.global_step,filename="probabilities_elon_3.json")

        # 多 GPU 场景下，平均 loss
        if self.args.n_gpu > 1:
            combined_loss = combined_loss.mean()

        # 反向传播
        self.accelerator.backward(combined_loss)

        return combined_loss.detach()


    # def training_step(self, model, inputs):
    #     model.train()
        
    #     inputs = self._prepare_inputs(inputs)
        
    #     device = next(model.parameters()).device

    #     # kl_loss,another_loss = self.compute_kl_loss(model, inputs, device)

    #     # Assuming another_loss calculation here
    #     # another_loss = torch.tensor(1.0, device=device)
    #     # sensitive_tokens = ["Donald", "Trump"]
    #     # # combined_loss = another_loss
    #     # label_smoother = LabelSmootherGA_GD(
    #     #     epsilon=0.1,
    #     #     ignore_index=-100,
    #     #     sensitive_tokens=sensitive_tokens,
    #     #     tokenizer=self.tokenizer
    #     # )
    #     # labels = inputs.pop("labels")
    #     # outputs = model(**inputs)
    #     # combined_loss = label_smoother(outputs, labels,shift_labels=True)
        
    #     combined_loss,normal_outputs = self.compute_loss(model, inputs,return_outputs=True)  

    #     if self.args.n_gpu > 1:
    #         combined_loss = combined_loss.mean()

    #     self.accelerator.backward(combined_loss)

    #     return combined_loss.detach()
    
    
    
    # def training_step(self, model, inputs):
    #     model.train()
        
    #     inputs = self._prepare_inputs(inputs)
        
    #     device = next(model.parameters()).device

    #     # kl_loss,another_loss = self.compute_kl_loss(model, inputs, device)

    #     # Assuming another_loss calculation here
    #     # another_loss = torch.tensor(1.0, device=device)
    #     # sensitive_tokens = ["Donald", "Trump"]
    #     # # combined_loss = another_loss
    #     # label_smoother = LabelSmootherGA_GD(
    #     #     epsilon=0.1,
    #     #     ignore_index=-100,
    #     #     sensitive_tokens=sensitive_tokens,
    #     #     tokenizer=self.tokenizer
    #     # )
    #     # labels = inputs.pop("labels")
    #     # outputs = model(**inputs)
    #     # combined_loss = label_smoother(outputs, labels,shift_labels=True)
    #     with autocast(dtype=torch.bfloat16): 
    #         base_loss,outputs = self.compute_loss(model, inputs,return_outputs=True)  
    #         logits = outputs.logits  # (batch_size, seq_length, vocab_size)
    #         labels = inputs["labels"].to(device)  # (batch_size, seq_length)

    #         # 使用 HuggingFace 模型自动计算的 base_loss
    #         # HuggingFace 自动计算的损失（句子级别）

    #         # 定义敏感 token 列表
    #     # 反向传播基础损失，并保持计算图
    #         # self.accelerator.backward(base_loss, retain_graph=True)

    #         # 处理 GA 损失
    #         ga_loss = torch.tensor(0.0, device=device, requires_grad=True)  # 确保 ga_loss 是一个张量，并追踪梯度
    #         sensitive_tokens = ["Donald", "Trump"]
    #         print("ll",labels.shape)
    #         print("ss",logits.shape)
    #         for i in range(labels.size(0)):  # batch size
    #             for j in range(labels.size(1)):  # sequence length
    #                 if labels[i, j] == -100:  # 跳过 padding 部分
    #                     continue

    #                 token = self.tokenizer.decode([labels[i, j].item()])
    #                 print(f"Token at position [{i}, {j}]: {token}")  # 打印 token

    #                 if token in sensitive_tokens:
    #                     # 检查 logits 和 labels 的形状是否一致
    #                     print(f"logits[i, j].shape: {logits[i, j].shape}, labels[i, j]: {labels[i, j]}")

    #                     # 确保 labels 是有效的标量，并且 logits[i, j] 是 (vocab_size,) 的张量
    #                     if labels[i, j].numel() > 0 and logits[i, j].size(-1) == 32000:
    #                         # 累加梯度上升损失
    #                         loss_value = -F.cross_entropy(logits[i, j].unsqueeze(0), labels[i, j].unsqueeze(0))
    #                         print(f"GA loss for {token}: {loss_value.item()}")  # 打印损失值
    #                         ga_loss = ga_loss + loss_value  # 使用非原地操作累加损失
    #                     else:
    #                         print(f"Skipping token at [{i}, {j}] due to invalid label or logits shape mismatch.")

    #         # 确保 ga_loss 是有效的
    #         if torch.isnan(ga_loss) or torch.isinf(ga_loss):
    #             print(f"Invalid ga_loss detected: {ga_loss}")
    #             ga_loss = torch.tensor(0.0, device=device, requires_grad=True)

    #         print(f"Final GA loss: {ga_loss.item()}")

    #         # 检查 ga_loss 是否有梯度
    #         if ga_loss.requires_grad and ga_loss.item() != 0.0:
    #             self.accelerator.backward(ga_loss)

    #     return (base_loss + ga_loss).detach()
    
    
    

    # def training_step(self, model, inputs):
    #     model.train()
    #     device = next(model.parameters()).device

    #     # 计算KL loss和其他损失
    #     kl_loss, another_loss = self.compute_kl_loss(model, inputs, device)

    #     # 计算组合损失
    #     combined_loss = kl_loss - 0.3 * another_loss

    #     if self.args.n_gpu > 1:
    #         combined_loss = combined_loss.mean()

    #     # 进行反向传播，计算梯度，但不更新参数
    #     self.accelerator.backward(combined_loss)

    #     # 反向传播后，此时梯度已经计算出来，可以修改梯度
    #     sensitive_params = self.identify_sensitive_parameters(model, inputs, device)

    #     # 向敏感参数中注入差分隐私噪声
    #     self.apply_dp_noise(sensitive_params, noise_scale=0.1)

    #     # 打印模型参数名，以便调试
    #     # for name, param in model.named_parameters():
    #     #     print(f"Checking parameter: {name}")

    #     # 只更新LoRA层的梯度
    #     for name, param in model.named_parameters():
    #         if 'lora' in name and param.grad is not None and param.grad.numel() > 0:  # 只考虑LoRA层
    #             if name in sensitive_params:
    #                 print(name)
    #                 print(param.grad)
    #                 print("gradshape:", param.grad.shape)
    #                 print("senseshape:", sensitive_params[name].shape)
    #                 if param.grad.size() == sensitive_params[name].size():
    #                     # 使用 copy_() 方法进行赋值
    #                     param.grad.copy_(sensitive_params[name])  # 用带噪声的梯度替换
    #                 else:
    #                     print(f"Skipping gradient update for {name} due to size mismatch. "
    #                         f"param.grad.size() = {param.grad.size()}, "
    #                         f"sensitive_params[name].size() = {sensitive_params[name].size()}")
    #             else:
    #                 print(f"Parameter {name} not found in sensitive_params. Available keys: {list(sensitive_params.keys())}")
    #         else:
    #             # 输出梯度为None的情况，帮助调试
    #             if param.grad is None:
    #                 print(f"Skipping {name} because param.grad is None.")
    #             else:
    #                 print(f"Skipping {name} because param.grad is empty.")

    #     # 确保 optimizer.step() 在梯度被正确处理后才被调用
    #     # if hasattr(self.optimizer, 'step'):
            
    #     #     self.optimizer.step()
    #     #     print("step done")
    #     # else:
    #     #     print("Optimizer step method not found.")

    #     return combined_loss.detach()
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
