import os
import math
import json
import random
import time
import numpy as np
import torch
from functools import partial
from utils.common_utils import print_rank_0, TASK2ID, ID2TASK, get_local_rank

class SingleTaskDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            name,
            data_prefix,
            input_dataset,
            # loss_mask_dataset,
            # num_samples,
            seq_length,
            weighted_loss_mode=None,
            ds_weight=1.0,
    ):

        self.name = name
        self.input_dataset = input_dataset
        self.num_samples = len(self.input_dataset['input_ids'])
        # self.loss_mask_dataset = loss_mask_dataset
        self.seq_length = seq_length

        self.weighted_loss_mode = weighted_loss_mode
        self.ds_weight = ds_weight
        self.task_name = data_prefix.split('/')[-1]
        self.task_id = TASK2ID[self.task_name]

        # Checks

    def update_ds_weight(self, weight):
        self.ds_weight = weight

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        try:
            # Get the shuffled index.
            idx = idx % self.num_samples
            idx_data = {key: self.input_dataset[key][idx]
                        for key in self.input_dataset}

            if self.weighted_loss_mode:
                idx_data["weight"] = np.array([self.ds_weight], dtype=np.float32)
                idx_data["task_id"] = np.array([self.task_id], dtype=np.int)
                return idx_data
            else:
                idx_data["task_id"] = np.array([self.task_id], dtype=np.int)
                return idx_data
        except IndexError:
            new_idx = idx % len(self)
            print(
                f"WARNING: Got index out of bounds error with index {idx} - taking modulo of index instead ({new_idx})"
            )
            return self[new_idx]


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 dataset_type,
                 data_paths,
                 tokenizer,
                 max_input_length=550,
                 max_output_length=550,
                 max_length=1024,
                 no_append_glm_mask=False,
                 gpt_data=False,
                 world_size=1,
                 global_rank=0,
                 left_truncate=False,
                 shard_data=False,
                 **kwargs):
        super().__init__()
        self.args = args
        self.dataset_type = dataset_type
        self.mode = args.tokenize_mode
        # self.seq_length = max_length
        # self.max_seq_length = args.seq_length + 1
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.gpt_data = gpt_data
        # if self.gpt_data:
        #     self.tokenizer.sop_token = "<|endoftext|>"
        #     self.tokenizer.eop_token = "<|endoftext|>"
        #     self.tokenizer.pad_token = "<|extratoken_1|>"
        self.sop_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.sop_token)
        # self.eop_id = self.tokenizer.convert_tokens_to_ids(
            # self.tokenizer.eop_token)
        self.eop_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.eos_token)
        self.pad_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.pad_token)
        print_rank_0(f'self.tokenizer.sop_token {self.tokenizer.sop_token} id: {self.sop_id}')
        # print_rank_0(f'self.tokenizer.eop_token {self.tokenizer.eop_token} id: {self.eop_id}')
        print_rank_0(f'self.tokenizer.eos_token {self.tokenizer.eos_token} id: {self.eop_id}')
        print_rank_0(f'self.tokenizer.pad_token {self.tokenizer.pad_token} id: {self.pad_id}')
        self.mask = kwargs.get('glm_mask', '[gMASK]')
        self.no_append_glm_mask = no_append_glm_mask
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.max_length = max_length
        self.left_truncate = left_truncate
        self.kwargs = kwargs
        self.shard_data = shard_data
        self.world_size = world_size
        self.global_rank = global_rank
        self.datasets = None
        self.weights = None
        self.table = {ord(f): ord(t) for f, t in zip(
            u'，。！？：【】（）％＃＠＆１２３４５６７８９０',
            u',.!?:[]()%#@&1234567890')}
        self.BAD_MARKERS = [
            'An unhelpful answer:\n',
            'The following is a worse answer.\n',
            'The following is a less accurate answer.\n',
            'The following is a less correct answer.\n',
            # 'Generate a worse answer.\n',
            # 'Generate a less accurate answer.\n',
            # 'Generate a less correct answer.\n',
            '一个没有帮助的回答:\n',
            '下面是一个更差的回答.\n',
            '下面是一个不太准确的回答.\n',
            '下面是一个不太正确的回答.\n',
            # '请生成一个更差的回答.\n',
            # '请生成一个不太准确的回答.\n',
            # '请生成一个不太正确的回答.\n',
        ]
        self.GOOD_MARKERS = [
            'A helpful answer:\n',
            'The following is a better answer.\n',
            'The following is a more accurate answer.\n',
            'The following is a more correct answer.\n',
            # 'Generate a better answer.\n',
            # 'Generate a more accurate answer.\n',
            # 'Generate a more correct answer.\n',
            '一个有帮助的回答:\n',
            '下面是一个更好的回答.\n',
            '下面是一个更准确的回答.\n',
            '下面是一个更正确的回答.\n',
            # '请生成一个更好的回答.\n',
            # '请生成一个更准确的回答.\n',
            # '请生成一个更正确的回答.\n',
        ]
        self._load_dataset_from_jsonl()

        # self.datasets = None
        num_datasets = len(self.datasets)
        weights = self.weights
        assert num_datasets == len(weights)

        self.size = 0
        for dataset in self.datasets:
            self.size += len(dataset)

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # recompute weights
        weights = self.calc_weights()

        # Build indices.
        start_time = time.time()
        assert num_datasets < 255
        self.dataset_index = np.zeros(self.size, dtype=np.uint8)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)

        from data import helpers

        helpers.build_blending_indices(
            self.dataset_index,
            self.dataset_sample_index,
            weights,
            num_datasets,
            self.size,
            torch.distributed.get_rank() == 0,
        )

        print(
            "> RANK {} elapsed time for building blendable dataset indices: "
            "{:.2f} (sec)".format(
                torch.distributed.get_rank(), time.time() - start_time
            )
        )

    def encode(self, data):
        # prompt, content, bad_content = data['input'], data['output'], ''
        # prompt, content, bad_content = data['src_code'], data['tgt_code'], ''
        prompt, content, bad_content = data['chat_rounds'][0]['content'], data['chat_rounds'][1]['content'], ''
        encode_res = {
            "input_ids": [],
            "position_ids": [],
            "attention_mask": [],
            "labels": []
        }
        # if not self.gpt_data:
        cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        mask_id = self.tokenizer.convert_tokens_to_ids(self.mask)
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        if self.mode == 'coh':
            marker_idx_en = random.randint(0, 3)
            marker_idx_zh = random.randint(4, 7)
            gms = [self.GOOD_MARKERS[marker_idx_en], self.GOOD_MARKERS[marker_idx_zh]]
            bms = [self.BAD_MARKERS[marker_idx_en], self.BAD_MARKERS[marker_idx_zh]]
            for good_marker, bad_marker in zip(gms, bms):
                # pn
                for token_res in self._tokenize_fields(prompt, content, bad_content, good_marker, bad_marker,
                                                       cls_id=cls_id, pad_id=pad_id, mask_id=mask_id,
                                                       max_length=self.max_length,
                                                       max_input_length=self.max_input_length,
                                                       max_output_length=self.max_output_length):
                    for k, v in token_res.items():
                        encode_res[k].append(v)
                # np
                for token_res in self._tokenize_fields(prompt, bad_content, content, bad_marker, good_marker,
                                                       max_length=self.max_length,
                                                       max_input_length=self.max_input_length,
                                                       max_output_length=self.max_output_length):
                    for k, v in token_res.items():
                        encode_res[k].append(v)
        else:
            for token_res in self._tokenize_fields(prompt, content, bad_content, "", "",
                                                   max_length=self.max_length,
                                                   max_input_length=self.max_input_length,
                                                   max_output_length=self.max_output_length):
                for k, v in token_res.items():
                    encode_res[k].append(v)
        return encode_res, len(prompt) + len(content) + len(bad_content)

    def punctuation_format(self, text):
        # Replace non-breaking space with space
        text = text.strip() + '\n'
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        # change chinese punctuation to english ones
        text = text.translate(self.table)
        return text

    def _tokenize_fields(self, prompt, content, bad_content, good_marker,
                         bad_marker, cls_id=None,
                         pad_id=None, mask_id=None,
                         for_generation=False,
                         for_eval_classification=False,
                         max_length=1024,
                         max_input_length=512,  # 仅在生成预测样本时生效,最大不超过1024
                         max_output_length=512,  # 仅在生成预测样本是生效,最大不超过1024
                         gpt_data=False,
                         old_version_tokenizer=False,
                         left_truncate=False):
        sop_id = self.sop_id
        eop_id = self.eop_id
        pad_id = self.pad_id
        if not self.gpt_data:
            mask_id = mask_id if mask_id else self.tokenizer.convert_tokens_to_ids(
                '[gMASK]')
            cls_id = cls_id if cls_id else self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.cls_token)
        # pad_id = pad_id if pad_id else self.tokenizer.convert_tokens_to_ids(
        #     self.tokenizer.pad_token)

        # good_marker_ids = self.tokenizer(good_marker)['input_ids'][1:-1]
        # bad_marker_ids = self.tokenizer(bad_marker)['input_ids'][1:-1]
        # # human_marker_ids = self.tokenizer("<human>: ")['input_ids'][1:-1]
        # # bot_marker_ids = self.tokenizer("<bot>: ")['input_ids'][1:-1]
        # human_marker_ids = self.tokenizer("<|role_start|>human<|role_end|>")['input_ids'][1:-1]
        # bot_marker_ids = self.tokenizer("<|role_start|>bot<|role_end|>")['input_ids'][1:-1]
        good_marker_ids = self.tokenizer(good_marker)['input_ids']
        bad_marker_ids = self.tokenizer(bad_marker)['input_ids']
        # human_marker_ids = self.tokenizer("<human>: ")['input_ids']
        # bot_marker_ids = self.tokenizer("<bot>: ")['input_ids']
        human_marker_ids = self.tokenizer("<|role_start|>human<|role_end|>")['input_ids']
        bot_marker_ids = self.tokenizer("<|role_start|>bot<|role_end|>")['input_ids']
        if not self.gpt_data:
            good_marker_ids = good_marker_ids[1:-1]
            bad_marker_ids = bad_marker_ids[1:-1]
            human_marker_ids = human_marker_ids[1:-1]
            bot_marker_ids = bot_marker_ids[1:-1]

        if prompt:
            # maintain clm formatting
            # prompt = self.punctuation_format(prompt)
            if not prompt.endswith('\n') and self.gpt_data:  # chatML格式
                prompt = prompt + '\n'
            prompt_ids = self.tokenizer(prompt)['input_ids']
            if not self.gpt_data:
                prompt_ids = prompt_ids[1:-1]
        else:
            prompt_ids = []

        if content:
            # content = self.punctuation_format(content)
            if not content.endswith('\n') and self.gpt_data:  # chatML格式
                content = content + '\n'
            content_ids = self.tokenizer(content)['input_ids']
            if not self.gpt_data:
                content_ids = content_ids[1:-1]
        else:
            content_ids = []

        if bad_content:
            # bad_content = self.punctuation_format(bad_content)
            if not bad_content.endswith('\n') and self.gpt_data:  # chatML格式
                bad_content = bad_content + '\n'
            bad_content_ids = self.tokenizer(bad_content)['input_ids']
            if not self.gpt_data:
                bad_content_ids = bad_content_ids[1:-1]
        else:
            bad_content_ids = []

        # 处理逻辑：
        # 1. 根据mode切换关注类型，统一处理SST,SFT,CoH的需求
        # 1. 按照GLM的结构保持："[CLS]input[gMASK][SOP]output[EOP]"
        # 3. content和bad_content 计算loss，需要设计对应的loss_mask

        input_ids = []
        output_ids = []
        position_ids = []

        # print(self.mode)
        if self.mode == 'pretrain':
            input_ids = content_ids
        elif self.mode == 'sft':
            # input_ids = human_marker_ids + prompt_ids + bot_marker_ids + good_marker_ids + content_ids
            if self.gpt_data:
                input_ids = human_marker_ids + prompt_ids + [sop_id] + bot_marker_ids + good_marker_ids
            else:
                input_ids = prompt_ids
        elif self.mode == 'coh':
            input_ids = human_marker_ids + prompt_ids + bot_marker_ids + good_marker_ids + content_ids + [sop_id] + bad_marker_ids + bad_content_ids
        # 保持GLM input结构
        if for_generation:
            # 预留特殊字符的长度
            if len(input_ids) > max_input_length:
                if left_truncate:
                    input_ids = input_ids[-max_input_length:]
                else:
                    input_ids = input_ids[:max_input_length]
        else:
            if self.gpt_data:
                # num_special_tokens = 3
                num_special_tokens = 1  # 只有一个[eop_id]
            else:
                num_special_tokens = 4
            output_ids = content_ids
            if len(input_ids) + len(output_ids) > max_length - num_special_tokens:  # 4是需要添加的特殊符号的个数
                return {
                    'input_ids': []
                }
                # if len(input_ids) > (max_length - num_special_tokens) // 2 \
                #         and len(output_ids) > (max_length - num_special_tokens) // 2:
                #     # 如果都超过了最大长度的一半,那都截取到最大长度的一半
                #     half_length = (max_length - num_special_tokens) // 2
                #     if left_truncate:
                #         input_ids = input_ids[-half_length:]

                #     else:
                #         input_ids = input_ids[:half_length]

                #     output_ids = output_ids[:half_length]
                # else:
                #     # 从input_ids和output_ids中比较长的那一个截断,input_ids可以选择从左边或右边阶段,output_ids默认从右边截断
                #     if len(input_ids) >= len(output_ids):
                #         if left_truncate:
                #             input_ids = input_ids[-(max_length -
                #                                     num_special_tokens - len(output_ids)):]

                #         else:
                #             input_ids = input_ids[:max_length -
                #                                   num_special_tokens - len(output_ids)]

                #     else:
                #         output_ids = output_ids[:max_length -
                #                                 num_special_tokens - len(input_ids)]
            assert len(input_ids) + len(output_ids) <= max_length - num_special_tokens
        if self.gpt_data:
            # input_ids = [cls_id] + input_ids
            sep = 0
        else:
            input_ids = [cls_id] + input_ids + [mask_id]
            sep = len(input_ids)
            mask_pos = input_ids.index(mask_id)
            if mask_pos == -1:
                print('No mask')
            position_ids = list(range(len(input_ids)))
            block_position_ids = [0] * len(input_ids)
        # 获得mask所在的位置，用于后面output positionid的构造
        if for_generation:
            if gpt_data:
                sep = 0
                position_ids = list(range(max_length))
            else:
                sep = len(input_ids)
                position_ids = position_ids + [mask_pos] * (max_output_length + 1)  # 后面input_ids要加一个sop_id
                block_position_ids = block_position_ids + list(range(1, max_output_length + 2))
                position_ids = [position_ids, block_position_ids]
                # 后面input_ids要加一个sop_id
                max_length = len(input_ids) + max_output_length + 1
            generation_attention_mask = np.ones([max_length, max_length])
            generation_attention_mask[:sep, sep:] = 0
            for i in range(sep, max_length):
                generation_attention_mask[i, i + 1:] = 0
            if self.gpt_data:
                max_output_length = max_length - len(input_ids)
            input_ids = input_ids + [sop_id]
            return {'input_ids': torch.Tensor([input_ids]).long(),
                    'position_ids': torch.Tensor([position_ids]).long(),
                    'generation_attention_mask': torch.Tensor([[generation_attention_mask]]).long()
                    }
        else:
            output_ids = output_ids + [eop_id]
            labels = output_ids
            # 拼接输入输出
            if self.gpt_data:
                tokens = input_ids + output_ids
                labels = [-100] * (len(input_ids) - 1) + labels + [-100]  # gpt data没有[sop_id]，前面的[-100]长度需要-1
            else:
                tokens = input_ids + [sop_id] + output_ids
                # mask label
                labels = [-100] * len(input_ids) + labels + [-100]
            # 最大长度不全
            if len(tokens) < max_length:
                pad_length = max_length - len(tokens)
                tokens += [pad_id] * pad_length
                labels.extend([-100] * pad_length)
            if self.gpt_data:
                position_ids = list(range(max_length))
            else:
                # position_ids在mask之后全部补mask_pos
                position_ids = position_ids + [mask_pos] * (len(tokens) - len(position_ids))
                # block_position_ids在mask之后补1 2 3 4 5..
                block_position_ids = block_position_ids + list(range(1, len(tokens) - len(block_position_ids) + 1))
                position_ids = [position_ids, block_position_ids]
            assert len(tokens) == len(
                labels) == max_length
            return {'input_ids': tokens,
                    'position_ids': position_ids,
                    'attention_mask': sep,
                    'labels': labels}

    def skip_line(self, line):
        try:
            data = json.loads(line.rstrip('\n\r'))
        except Exception:
            return True
        if not data.get('input', '') or not data.get('output', ''):
            return True
        return False
    
    def get_split(self):
        splits = []
        splits_string = self.args.data_split
        if splits_string.find(",") != -1:
            splits = [float(s) for s in splits_string.split(",")]
        elif splits_string.find("/") != -1:
            splits = [float(s) for s in splits_string.split("/")]
        else:
            splits = [float(splits_string)]
        while len(splits) < 3:
            splits.append(0.0)
        splits = splits[:3]
        return splits
    
    def _load_dataset_from_jsonl(self):

        data_prefixes = list(self.data_paths[1:-1].split(','))

        all_datasets = []
        all_datasets_length = []
        # 每个数据集的有效token数
        num_tokens = []
        effective_token_rate = []
        # 每个数据集的样本数
        total_sample_cnt = []

        self.global_num_samples = 0
        self.local_num_samples = 0

        # 不同数据集在不同文件夹下
        for dataset_index in range(len(data_prefixes)):
            if self.args.split_before_read:
                files = os.listdir(data_prefixes[dataset_index] + '/' + self.dataset_type)
            else:
                files = os.listdir(data_prefixes[dataset_index])
            cur_dataset_input_ids = []
            cur_dataset_position_ids = []
            cur_dataset_attention_mask = []
            cur_dataset_labels = []
            # 同一数据集下可能有多个jsonl文件
            for file in files:
                if self.args.split_before_read:
                    file_name = data_prefixes[dataset_index] + '/' + self.dataset_type + '/' + file
                else:
                    file_name = data_prefixes[dataset_index] + '/' + file
                    if os.path.isdir(file_name):
                        continue
                fin = open(file_name, 'r')
                lines = fin.readlines()
                line_num = len(lines)
                print_rank_0(f'lines of {file_name} are {line_num}')
                print_rank_0(f'open file {file_name}')
                if self.args.split_before_read:
                    start_index = 0
                    end_index = line_num
                else:
                    splits = self.get_split()
                    train_ratio = splits[0] / 100.0
                    train_num = int(math.ceil(train_ratio * line_num))
                    if self.dataset_type == 'train':
                        start_index = 0
                        end_index = train_num
                    if self.dataset_type == 'valid':
                        start_index = train_num
                        end_index = line_num
                for i in range(start_index, end_index):
                    line = lines[i]
                    if self.skip_line(line):
                        print_rank_0(f'skip!!!!')
                        continue
                    self.global_num_samples += 1
                    if self.shard_data and i % self.world_size != self.global_rank:
                        continue
                    self.local_num_samples += 1
                    data = json.loads(line.rstrip('\n\r'))
                    if not self.gpt_data:
                        features = self._tokenize_fields(data['input'], data['output'], "", "", "",
                                                        max_length=self.max_length,
                                                        max_input_length=self.max_input_length,
                                                        max_output_length=self.max_output_length)
                    else:
                        if data['chat_rounds'][0]['role'] == 'system':
                            prompt = data['chat_rounds'][0]['content'] + data['chat_rounds'][1]['content']
                            answer = data['chat_rounds'][2]['content']
                        else:
                            prompt = data['chat_rounds'][0]['content']
                            answer = data['chat_rounds'][1]['content']
                        features = self._tokenize_fields(prompt, answer, "", "", "",
                                                        max_length=self.max_length,
                                                        max_input_length=self.max_input_length,
                                                        max_output_length=self.max_output_length)

                    # 一个document可能编码不出sample
                    if len(features['input_ids']) <= 1:
                        self.global_num_samples -= 1
                        self.local_num_samples -= 1
                        continue
                    cur_dataset_input_ids.append(features['input_ids'])
                    cur_dataset_position_ids.append(features['position_ids'])
                    cur_dataset_attention_mask.append(features['attention_mask'])
                    cur_dataset_labels.append(features['labels'])

                fin.close()

            cur_dataset_num_tokens = np.sum(cur_dataset_input_ids, dtype=np.int32)
            cur_dataset_sample_num = len(cur_dataset_input_ids)
            num_tokens.append(cur_dataset_num_tokens)
            total_sample_cnt.append(cur_dataset_sample_num)
            effective_token_rate.append(cur_dataset_num_tokens / (cur_dataset_sample_num * self.max_length))

            # "weight", "task_id"字段会在getitem的时候获取
            # np.array(cur_dataset_input_ids, dtype=np.float32)
            cur_dataset = {'input_ids': np.array(cur_dataset_input_ids, dtype=np.int),
                           'position_ids': np.array(cur_dataset_position_ids, dtype=np.int),
                           'attention_mask': np.array(cur_dataset_attention_mask, dtype=np.int),
                           'labels': np.array(cur_dataset_labels, dtype=np.int)}
            # cur_dataset = {'input_ids': torch.Tensor(cur_dataset_input_ids).long(),
            #                'position_ids': torch.Tensor(cur_dataset_position_ids).long(),
            #                'attention_mask': torch.Tensor(cur_dataset_attention_mask).long(),
            #                'labels': torch.Tensor(cur_dataset_labels).long()}

            print(
                f"shape of cur {self.dataset_type} dataset in rank \
                {self.global_rank}: {cur_dataset['input_ids'].shape}")

            cur_ds = SingleTaskDataset(
                self.dataset_type,
                data_prefixes[dataset_index],
                cur_dataset,
                self.max_length,
                weighted_loss_mode=self.args.weighted_loss_mode,
                # ds_weight=splits[0]
            )

            all_datasets.append(cur_ds)
            all_datasets_length.append(len(cur_ds))
        
        # 重写self.global_sample_num
        torch.distributed.barrier()
        if self.shard_data:
            device = f"cuda:{get_local_rank()}"
            global_num_samples_tensor = torch.tensor(self.local_num_samples, dtype=torch.int32)
            global_num_samples_tensor = global_num_samples_tensor.to(device)
            torch.distributed.all_reduce(global_num_samples_tensor, op=torch.distributed.ReduceOp.SUM)
            self.global_num_samples = global_num_samples_tensor.item()
            torch.distributed.barrier()
            print(f'global sample num of {self.dataset_type} in rank {self.global_rank}: {self.global_num_samples}')
            print(f'local sample num of {self.dataset_type} in rank {self.global_rank}: {self.local_num_samples}')
            print(f'num tokens of {self.dataset_type} in rank {self.global_rank}: {num_tokens}')
            print(f'effective token rate of {self.dataset_type} in rank {self.global_rank}: {effective_token_rate}')

        num_tokens = []
        ds_fn = partial(ds_weights_by_num_docs_sft)
        loss_weights = ds_fn(all_datasets_length)

        print(f"> {self.dataset_type} loss weights in rank {self.global_rank}: {loss_weights}")

        factor = 1
        # calcualte common factor based on token cnt and total sample cnt
        if num_tokens:
            factor = sum(num_tokens) / (sum(total_sample_cnt) * self.max_length)
            factor /= sum([1.0 / w for w in loss_weights]) / len(loss_weights)
        print(f"> common denomination factor for CE loss of {self.dataset_type} in rank {self.global_rank}: {factor}")

        sample_weights = [x / sum(all_datasets_length) for x in all_datasets_length]
        print(f"> {self.dataset_type} sample weights in rank {self.global_rank}: {sample_weights}")

        for i in range(len(all_datasets)):
            print(
                f'loss weight of {self.dataset_type} dataset {i} before \
                update in rank {self.global_rank}: {all_datasets[i].ds_weight}')
        # train_dataset = None
        if all_datasets:
            for i in range(len(all_datasets)):
                all_datasets[i].update_ds_weight(loss_weights[i] / factor)
                print(
                    f'loss weight of {self.dataset_type} dataset {i} after \
                    update in rank {self.global_rank}: {all_datasets[i].ds_weight}')
            self.datasets = all_datasets
            self.weights = sample_weights

    def calc_weights(self):
        dataset_sample_cnt = [len(ds) for ds in self.datasets]
        total_cnt = sum(dataset_sample_cnt)
        weights = np.array([(cnt + 0.0) / total_cnt for cnt in dataset_sample_cnt], dtype=np.float64)
        return weights

    def __len__(self):
        return self.global_num_samples

    def __getitem__(self, idx):
        try:
            idx = idx % self.local_num_samples
            dataset_idx = self.dataset_index[idx]
            sample_idx = self.dataset_sample_index[idx]
            return self.datasets[dataset_idx][sample_idx]
        except IndexError:
            # new_idx = idx % len(self)
            new_idx = idx % self.local_num_samples
            print(
                f"WARNING: Got index out of bounds error with index {idx} - taking modulo of index instead ({new_idx})"
            )
            return self[new_idx]


def ds_weights_by_num_docs_sft(docs, alpha=0.3):
    # ignore alpha
    weights = [1 / i for i in docs]
    weights_sum = sum(weights)
    weights = [weight / weights_sum for weight in weights]
    return weights
