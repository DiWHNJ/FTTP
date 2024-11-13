import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import KeywordsStoppingCriteria

import torch.nn.functional as F

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional

from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria
)


from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

import re

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
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

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
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


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
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

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
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
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
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
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
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def compute_attn_loss(self, model, batch, device):
        # print("aaaaaaamodel",model)
        # for name, param in core_model.named_parameters():
        #     if param.dtype != torch.bfloat16:
        #         print(f"A_Parameter {name} is not BFloat16, it is {param.dtype}")
        # for name, param in core_model.named_parameters():
        #     print(f"Parameter {name} dtype: {param.dtype}")
        # print("finish_check")
        # print("type(model)",type(model))  # 这会打印出 DeepSpeedEngine
        # print("type(model.module)",type(model.module))  # 这会打印出 PeftModelForCausalLM
        # print("type(model.module.model)",type(model.module.model))  # 这会打印出 LlamaLlamaForCausalLM
        # print("type(tokenizer)",type(self.tokenizer))

        # special token
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)              # 感觉每次调用compute_attn_loss时都会add_tokens不好,想把这个移到模型初始化的地方

        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        # vision_tower = model.get_model().vision_tower[0]
        # vision_tower = model.get_model().vision_tower
        vision_tower = model.module.model.get_vision_tower()

        device = next(model.parameters()).device

        # if vision_tower.device.type == 'meta':
        if device.type == 'meta':
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
            # model.get_model().vision_tower[0] = vision_tower
            model.get_model().vision_tower = vision_tower
        else:
            vision_tower.to(device='cuda', dtype=torch.float16)
        vision_config = vision_tower.config
        vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        qs = batch['conversations']
        print(f"Conversations batch size: {len(qs)}")
        print("batch['conversations']: ", batch['conversations'])
        print("type_batch['conversations']: ", type(batch['conversations']))

        prompt=[]
        # print("qs",qs)
        for i in range(len(qs)):
            if mm_use_im_start_end:
                qs[i] = qs[i]  + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN # system(1-97) <im_start>(98) <im_token>*256(354) <im_end>(355) text_start(360)
            else:
                qs[i] = qs[i] + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            # print("qs",qs)
            # conv_mode = "multimodal"
            conv_mode = "llava_v1"
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs[i])
            conv.append_message(conv.roles[1], None)
            pmpt = conv.get_prompt()
            prompt.append(pmpt)
        # print("prompt",prompt)
        prompts = self.tokenizer(prompt)
        print(f"Prompts input_ids shape: {len(prompts.input_ids)}")

        # breakpoint()

        # image_file = args.image
        # image = load_image(image_file)
        # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        # prompts_input_ids = torch.as_tensor(prompts.input_ids).cuda()
        max_len = max(len(seq) for seq in prompts.input_ids)  # 找到最长序列的长度
        padded_input_ids = [seq + [self.tokenizer.pad_token_id] * (max_len - len(seq)) for seq in prompts.input_ids]# 填充所有序列到相同长度
        prompts_input_ids = torch.as_tensor(padded_input_ids).cuda()# 转换为张量并移动到 CUDA

        batch['images'] = batch['images'].half().cuda()
        # 确保输入张量类型与模型的权重一致
        # prompts_input_ids = prompts_input_ids.to(torch.bfloat16).cuda()
        # prompts_input_ids = prompts_input_ids.to(self.weight.dtype)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, prompts_input_ids)

        print(f"prompts_input_ids shape: {prompts_input_ids.shape}, dtype: {prompts_input_ids.dtype}")
        print(f"batch['images'] shape: {batch['images'].shape}, dtype: {batch['images'].dtype}")

        print(f"vision_tower device: {vision_tower.device}")
    
        print(f"Model running on device: {device}")



        with torch.inference_mode():
            # output_ids = model.generate(
            #     prompts_input_ids,
            #     images=batch['images'],
            #     do_sample=False,
            #     max_new_tokens=64,
            #     stopping_criteria=[stopping_criteria])
                
            output_ids = model.generate(
                prompts_input_ids,
                images=batch['images'],
                do_sample=True,
                temperature=0.2,
                # max_new_tokens=self.args.max_length,      # 要改
                max_new_tokens=64,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        
        input_token_len = prompts_input_ids.shape[1]
        n_diff_input_output = (prompts_input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        an = outputs.strip()

        prompt = []
        for j in range(len(qs)):
            # conv_mode = "multimodal"
            conv_mode = "default"
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs[j])
            conv.append_message(conv.roles[1], an)
            pmpt = conv.get_prompt()[:-3]
            prompt.append(pmpt)
        prompts = self.tokenizer(prompt)
        
        # input_ids = torch.as_tensor(inputs.input_ids).cuda()
        max_len = max(len(seq) for seq in prompts.input_ids)  # 找到最长序列的长度
        padded_input_ids = [seq + [self.tokenizer.pad_token_id] * (max_len - len(seq)) for seq in prompts.input_ids]# 填充所有序列到相同长度
        prompts_input_ids = torch.as_tensor(padded_input_ids).cuda()# 转换为张量并移动到 CUDA
        # 确保输入张量类型与模型的权重一致  
        # prompts_input_ids = prompts_input_ids.to(torch.bfloat16).cuda()

        # prompts_input_ids = torch.as_tensor(prompts.input_ids).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, prompts_input_ids)

        with torch.no_grad():
            output = model(
                prompts_input_ids,
                # images=image_tensor.unsqueeze(0).half().cuda(),
                images = batch['images'],
                output_attentions=True,
                return_dict=True,
                )

        # attention = torch.mean(output.attentions[args.layer-1].squeeze(0), dim=0)   # 要改layer
        layer = 32
        attention = torch.mean(output.attentions[layer-1].squeeze(0), dim=0)
        attention = attention[358:, :]
        attention_image = attention[:, 98:354]
        logits = output.logits[:, 358:, :]

        attention_loss = F.mse_loss(attention_image, self.target_attention_map.to(device))  # 计算MSE损失
        return 
    


    def compute_attn_loss_new(self, model, batch, device):

        device = next(model.parameters()).device

        qs = batch['conversations']
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        print("batch['conversations']: ", batch['conversations'])
        prompt=[]

        for i in range(len(qs)):
            # if IMAGE_PLACEHOLDER in qs:
            #     if model.config.mm_use_im_start_end:
            #         qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            #     else:
            #         qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            # else:
            #     if model.config.mm_use_im_start_end:
            #         qs = image_token_se + "\n" + qs
            #     else:
            #         qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        
            conv_mode = "llava_v1"
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs[i])
            conv.append_message(conv.roles[1], None)
            pmpt = conv.get_prompt()
            prompt.append(pmpt)

        input_ids_a = (
        tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids_a)

        
        # print("prompt",prompt)
        prompts = self.tokenizer(prompt)
        print(f"Prompts input_ids shape: {len(prompts.input_ids)}")

        breakpoint()

        # image_file = args.image
        # image = load_image(image_file)
        # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        # prompts_input_ids = torch.as_tensor(prompts.input_ids).cuda()
        max_len = max(len(seq) for seq in prompts.input_ids)  # 找到最长序列的长度
        padded_input_ids = [seq + [self.tokenizer.pad_token_id] * (max_len - len(seq)) for seq in prompts.input_ids]# 填充所有序列到相同长度
        prompts_input_ids = torch.as_tensor(padded_input_ids).cuda()# 转换为张量并移动到 CUDA

        batch['images'] = batch['images'].half().cuda()
        # 确保输入张量类型与模型的权重一致
        # prompts_input_ids = prompts_input_ids.to(torch.bfloat16).cuda()
        # prompts_input_ids = prompts_input_ids.to(self.weight.dtype)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, prompts_input_ids)

        with torch.inference_mode():
            # output_ids = model.generate(
            #     prompts_input_ids,
            #     images=batch['images'],
            #     do_sample=False,
            #     max_new_tokens=64,
            #     stopping_criteria=[stopping_criteria])
                
            output_ids = model.generate(
                prompts_input_ids,
                images=batch['images'],
                do_sample=True,
                temperature=0.2,
                # max_new_tokens=self.args.max_length,      # 要改
                max_new_tokens=64,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        
        input_token_len = prompts_input_ids.shape[1]
        n_diff_input_output = (prompts_input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        an = outputs.strip()

        prompt = []
        for j in range(len(qs)):
            # conv_mode = "multimodal"
            conv_mode = "default"
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs[j])
            conv.append_message(conv.roles[1], an)
            pmpt = conv.get_prompt()[:-3]
            prompt.append(pmpt)
        prompts = self.tokenizer(prompt)
        
        # input_ids = torch.as_tensor(inputs.input_ids).cuda()
        max_len = max(len(seq) for seq in prompts.input_ids)  # 找到最长序列的长度
        padded_input_ids = [seq + [self.tokenizer.pad_token_id] * (max_len - len(seq)) for seq in prompts.input_ids]# 填充所有序列到相同长度
        prompts_input_ids = torch.as_tensor(padded_input_ids).cuda()# 转换为张量并移动到 CUDA
        # 确保输入张量类型与模型的权重一致  
        # prompts_input_ids = prompts_input_ids.to(torch.bfloat16).cuda()

        # prompts_input_ids = torch.as_tensor(prompts.input_ids).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, prompts_input_ids)

        with torch.no_grad():
            output = model(
                prompts_input_ids,
                # images=image_tensor.unsqueeze(0).half().cuda(),
                images = batch['images'],
                output_attentions=True,
                return_dict=True,
                )

        # attention = torch.mean(output.attentions[args.layer-1].squeeze(0), dim=0)   # 要改layer
        layer = 32
        attention = torch.mean(output.attentions[layer-1].squeeze(0), dim=0)
        attention = attention[358:, :]
        attention_image = attention[:, 98:354]
        logits = output.logits[:, 358:, :]

        attention_loss = F.mse_loss(attention_image, self.target_attention_map.to(device))  # 计算MSE损失
        return 




    def training_step(self, model, inputs):
        # print("inputs",inputs)
        model.train()
        device = next(model.parameters()).device

        kl_loss,another_loss = self.compute_attn_loss(model, inputs, device)

        # Assuming another_loss calculation here
        # another_loss = torch.tensor(1.0, device=device)

        combined_loss = kl_loss-another_loss

        if self.args.n_gpu > 1:
            combined_loss = combined_loss.mean()

        self.accelerator.backward(combined_loss)

        return combined_loss.detach()

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
