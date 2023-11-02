# coding=utf-8
# Copyright (c) 2023 Ant Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append("..")
import torch
import atorch
from utils.common_utils import print_rank_0, TASK2ID, ID2TASK, logger, is_main_process
from utils.hselect import hsj
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union
from KDEpy import FFTKDE

torch.set_printoptions(threshold=100)

def get_task_mask(task_id):
    task_num = len(TASK2ID)
    task_mask = torch.zeros(task_id.shape[0], task_num)
    task_mask[torch.arange(task_id.size(0)).unsqueeze(1), task_id] = 1
    
    return task_mask


def get_task_loss(task_losses, task_id):  # TODO
    # 固定一个打印顺序
    task_loss_per_batch = torch.zeros(len(ID2TASK)).to(device=task_id.device)
    # 统计每个task出现次数
    task_num_per_batch = torch.zeros(len(ID2TASK)).to(device=task_id.device)
    for i in range(len(task_id)):
        task_num_per_batch[task_id[i][0]] += 1
        task_loss_per_batch[task_id[i][0]] = task_losses[task_id[i][0]]

    return task_loss_per_batch, task_num_per_batch


# def get_task_loss_loop(token_losses, task_id):
#     task_losses = torch.zeros(len(ID2TASK)).to(device=task_id.device)
#     for i in range(len(task_id)):

#     return task_id
    

# def loss_func(loss):
#     return loss


def loss_func_old(outputs, labels, loss_mask, task_mask, task_id, weighted_loss_mode):
    if isinstance(outputs, dict):
        # print(outputs)
        lm_logits = outputs["logits"]
        shift_logits = lm_logits.contiguous()
        labels = labels.contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        bsz, seq_len = labels.shape
        if loss_mask is not None:
            loss_mask = loss_mask.view(-1)
            losses = losses.view(-1)
            if weighted_loss_mode:
                if weighted_loss_mode == "random":
                    unique_weights = torch.unique(loss_mask[loss_mask != 0.0])
                    rlw = F.softmax(torch.randn(len(unique_weights)), dim=-1) # RLW
                    loss = 0.0
                    for i, w in enumerate(unique_weights):
                        loss += rlw[i] * torch.sum(losses[loss_mask == w.item()] * loss_mask[loss_mask == w.item()]) / len(loss_mask)
                    loss *= len(unique_weights)
                elif weighted_loss_mode == "sample":
                    loss = torch.sum(losses * loss_mask) / torch.sum(loss_mask != 0.0)
                elif weighted_loss_mode == "token":
                    token_losses = losses * loss_mask  # [1, B * L]
                    # print_rank_0(task_mask)
                    task_mask_trans = torch.transpose(task_mask, 0, 1)
                    # print_rank_0(task_mask_trans.shape)
                    # print_rank_0(token_losses.shape)
                    task_token_losses = torch.matmul(task_mask_trans, token_losses.view(bsz, -1))  # [M * B] [B * L] = [M * L]
                    task_losses = torch.sum(task_token_losses, dim=1) / seq_len  # [M * 1]
                    loss = torch.sum(token_losses) / len(loss_mask)
                    # print_rank_0(f'loss: {loss}, task loss: {torch.sum(task_losses)}')
                else:
                    raise ValueError(f"weighted loss mode {weighted_loss_mode} is not supported.")
            else:
                loss = torch.sum(losses * loss_mask) / loss_mask.sum()
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        lm_loss = loss
        
    task_loss, task_num = get_task_loss(task_losses, task_id)
    return lm_loss, task_loss, task_num


# def loss_func_mft(outputs, labels, task_mask, task_id, weighted_loss_mode, loss_mask=None, weights=None, famo=None):
def loss_func_mft(outputs, inputs, weighted_loss_mode, famo=None, selfpaced_status=None):
    # task_id shape: [[1], [2], [4], [3], ..., [1]]
    labels, task_mask, task_id, loss_mask, weights = inputs['labels'], inputs['task_mask'], inputs['task_id'], inputs['loss_mask'], inputs['weights']
    if weighted_loss_mode == "selfpaced":
        if selfpaced_status is not None:
            loss_func = SelfPacedLoss(complete_steps=selfpaced_status.complete_steps, current_epoch=selfpaced_status.current_epoch)
        else:
            loss_func = SelfPacedLoss()
        # logits, labels, task_id, task_loss_prev, loss_mask
        loss, task_loss, task_num, selfpaced_status = loss_func(outputs["logits"], labels, task_id, loss_mask, selfpaced_status)
        return loss, task_loss, task_num, famo, selfpaced_status
    weighted = weighted_loss_mode
    # lm_logits = outputs["logits"].contiguous()
    lm_logits = outputs["logits"]
    labels = labels.to(device=lm_logits.device)
    task_mask = task_mask.to(device=lm_logits.device)
    task_id = task_id.to(device=lm_logits.device)
    # labels = labels.contiguous()
    bsz, seq_len = labels.shape
    # loss_mask = None
    if loss_mask is None:
        ineffective_tokens_per_sample = (labels==-100).sum(dim=1)
        effective_tokens_per_sample = - (ineffective_tokens_per_sample - seq_len)
        effective_tokens = bsz * seq_len - ineffective_tokens_per_sample.sum()
        if weighted_loss_mode.endswith('focalloss'):
            loss_fct = FocalLoss(reduction='none', ignore_index=-100)
        else:
            loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
    else:
        loss_mask = loss_mask.to(device=lm_logits.device)
        effective_tokens_per_sample = torch.sum(loss_mask, dim=1, dtype=torch.int)
        effective_tokens = torch.sum(loss_mask).item()
        if weighted_loss_mode.endswith('focalloss'):
            loss_fct = FocalLoss(reduction='none')
        else:
            loss_fct = CrossEntropyLoss(reduction='none')
    if weighted_loss_mode.endswith('focalloss'):
        losses = loss_fct(lm_logits, labels)
    else:
        losses = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))  # [B * L, 1]
    # losses = losses.contiguous().view(bsz, -1)
    losses = losses.view(bsz, -1)
    token_losses = losses.clone().detach().float() if loss_mask is None else losses.clone().detach().float()  # [B, L]
    task_mask_trans = torch.transpose(task_mask, 0, 1)
    unique_weights = torch.unique(weights)
    unique_id = torch.unique(task_id)
    if weighted_loss_mode == "case3" or weighted_loss_mode == "case4" or weighted_loss_mode == "case5" or weighted_loss_mode == "case6" or weighted_loss_mode == "case8" or weighted_loss_mode == "case3_focalloss":
        loss = 0.0
        # rlw = F.softmax(torch.randn(len(unique_weights)), dim=-1) # RLW
        for i, w in enumerate(unique_weights):
            row_idx = torch.squeeze(weights) == w.item()
            if weighted_loss_mode == "case3" or weighted_loss_mode == "case3_focalloss":
                if loss_mask is None:
                    loss += torch.sum(losses[row_idx, :]) / torch.sum(effective_tokens_per_sample[row_idx])
                else:
                    loss += torch.sum((losses * loss_mask)[row_idx, :]) / torch.sum(loss_mask[row_idx, :])
            elif weighted_loss_mode == "case4":
                if loss_mask is None:
                    loss += torch.mean(torch.sum(losses, dim=1)[row_idx] / effective_tokens_per_sample[row_idx])
                else:
                    loss += torch.mean(torch.sum(losses * loss_mask, dim=1)[row_idx] / torch.sum(loss_mask, dim=1)[row_idx])
            elif weighted_loss_mode == "case5":
                if loss_mask is None:
                    loss += torch.sum(losses[row_idx, :]) / torch.sum(effective_tokens_per_sample[row_idx]) * w
                else:
                    loss += torch.sum((losses * loss_mask)[row_idx, :]) / torch.sum(loss_mask[row_idx, :]) * w
            elif weighted_loss_mode == "case6":
                if loss_mask is None:
                    loss += torch.mean(torch.sum(losses, dim=1)[row_idx] / effective_tokens_per_sample[row_idx]) * w
                else:
                    loss += torch.mean(torch.sum(losses * loss_mask, dim=1)[row_idx] / torch.sum(loss_mask, dim=1)[row_idx]) * w
            elif weighted_loss_mode == "case8":
                if loss_mask is None:
                    loss += torch.mean(torch.max(losses, dim=1)[0][row_idx]) * w
                else:
                    loss += torch.mean(torch.max(losses * loss_mask, dim=1)[0][row_idx]) * w
        if weighted_loss_mode == "case3" or weighted_loss_mode == "case4" or weighted_loss_mode == "case3_focalloss":
            loss /= len(unique_weights)
        elif weighted_loss_mode == "case5" or weighted_loss_mode == "case6" or weighted_loss_mode == "case8":
            # loss *= 8 / len(unique_weights)
            loss *= len(ID2TASK) / len(unique_weights)
        # print_loss = torch.sum(losses * loss_mask) / loss_mask.sum()
        # print_rank_0(f'print_loss: {print_loss}')
    elif weighted_loss_mode == "case2":
        if loss_mask is None:
            loss = torch.mean(torch.sum(losses, dim=1) / effective_tokens_per_sample)
        else:
            loss = torch.mean(torch.sum(losses * loss_mask, dim=1) / torch.sum(loss_mask, dim=1))
    elif weighted_loss_mode == "case1":
        # flatten losses & loss_mask tensor
        if loss_mask is None:
            # losses = losses.view(-1)
            # loss = torch.sum(losses.contiguous().view(-1)) / effective_tokens
            loss = torch.sum(losses.view(-1)) / effective_tokens
        else:
            # loss_mask = loss_mask.view(-1)
            # losses = losses.view(-1)
            # loss = torch.sum(losses.contiguous().view(-1) * loss_mask.contiguous().view(-1)) / loss_mask.sum()
            loss = torch.sum(losses.view(-1) * loss_mask.view(-1)) / loss_mask.sum()
    elif weighted_loss_mode.startswith('case1_plus'):
        loss = 0.0
        alpha = 0.5
        # beta = 0.5
        beta = float(weighted_loss_mode.split('_')[-1])
        if loss_mask is None:  # TODO:
            loss += torch.sum(losses) / effective_tokens
        else:
            total_Q_ijk = torch.masked_select(losses.contiguous().view(-1), loss_mask.contiguous().view(-1).bool())
        a = torch.tensor(np.nanmax(total_Q_ijk.clone().detach().cpu().numpy()), device=losses.device)
        mu = torch.tensor(np.nanmean(total_Q_ijk.clone().detach().cpu().numpy()), device=losses.device)
        # lam = (mu + a) / 2  # 0.25 outliers
        # lam = (lam + a ) / 2  # 0.125 outliers
        lam = mu + beta * (a - mu)  # beta = 0.9 -> outliers = 0.05, beta = 0.8 -> outliers = 0.1
        for i in range(len(losses)):
            if loss_mask is None:
                Q_ijk = losses[i]
                mask = (labels != -100)
                Q_ijk = torch.masked_select(Q_ijk, mask)
            else:
                Q_ijk = losses[i] * loss_mask[i]
                Q_ijk = torch.masked_select(Q_ijk, loss_mask[i].bool())
            if effective_tokens_per_sample[i] > 10:
                m = torch.sum(Q_ijk.le(lam)).item()
                iterm_theta = torch.where(Q_ijk > lam, 0, Q_ijk)
                loss += (effective_tokens_per_sample[i] - m) * lam + torch.sum(iterm_theta)
            else:
                loss += torch.sum(Q_ijk)
        loss /= effective_tokens
    elif weighted_loss_mode.startswith('famo'):
        if famo is not None:
            if not famo.first_train_step:
                with torch.no_grad():
                    task_losses = torch.zeros(len(ID2TASK)).to(device=task_id.device)
                    for i, w in enumerate(unique_id):
                        row_idx = torch.squeeze(task_id) == w.item()
                        if loss_mask is None:
                            task_losses[w] = torch.sum(losses[row_idx, :]) / torch.sum(effective_tokens_per_sample[row_idx])
                        else:
                            task_losses[w] = torch.sum((losses * loss_mask)[row_idx, :]) / torch.sum(loss_mask[row_idx, :])
                    if famo.mode.startswith('famo_train'):
                        famo.update(task_losses)
            else:
                famo.first_train_step = False
            task_losses = torch.zeros(len(ID2TASK)).to(device=task_id.device)
            for i, w in enumerate(unique_id):
                row_idx = torch.squeeze(task_id) == w.item()
                if loss_mask is None:
                    task_losses[w] = torch.sum(losses[row_idx, :]) / torch.sum(effective_tokens_per_sample[row_idx])
                else:
                    task_losses[w] = torch.sum((losses * loss_mask)[row_idx, :]) / torch.sum(loss_mask[row_idx, :])
            if loss_mask is None:
                famo.print_loss = torch.sum(losses) / effective_tokens
            else:
                famo.print_loss = torch.sum(losses * loss_mask) / loss_mask.sum()
            # print_rank_0(f'task_losses: {task_losses}')
            # loss = famo.backward(task_losses)  # 只是获得加权后的loss 可能为负值
            loss = famo.get_weighted_loss(losses=task_losses)  # 只是获得加权后的loss 可能为负值
            
            # print_rank_0(f'loss: {loss}')
            # print_rank_0(f'print loss: {famo.print_loss}')
        else:
            if loss_mask is None:
                loss = torch.sum(losses) / effective_tokens
            else:
                loss = torch.sum(losses * loss_mask) / loss_mask.sum()
    elif weighted_loss_mode.startswith('case7') or weighted_loss_mode == 'case9':
        loss = 0.0
        alpha = 0.5
        # beta = 0.5
        beta = float(weighted_loss_mode.split('_')[-1])
        loss_per_sample = torch.zeros(bsz).to(device=losses.device)
        for i in range(len(losses)):
            if loss_mask is None:
                Q_ijk = losses[i]
                mask = (labels != -100)
                Q_ijk = torch.masked_select(Q_ijk, mask)
            else:
                Q_ijk = losses[i] * loss_mask[i]
                Q_ijk = torch.masked_select(Q_ijk, loss_mask[i].bool())
            if effective_tokens_per_sample[i] > 10:
                # mu = torch.mean(Q_ijk)
                mu = torch.tensor(np.nanmean(Q_ijk.clone().detach().cpu().numpy()), device=losses.device)
                # a = torch.max(Q_ijk).item()
                a = torch.tensor(np.nanmax(Q_ijk.clone().detach().cpu().numpy()), device=losses.device)
                # lam = (mu + a) / 2  # 0.25 outliers
                # lam = (lam + a ) / 2  # 0.125 outliers
                lam = mu + beta * (a - mu)  # beta = 0.9 -> outliers = 0.05, beta = 0.8 -> outliers = 0.1
                m = torch.sum(Q_ijk.le(lam)).item()
                iterm_theta = torch.where(Q_ijk > lam, 0, Q_ijk)
                if weighted_loss_mode == 'case9':
                    TL_c = ((effective_tokens_per_sample[i] - m) * lam + torch.sum(iterm_theta)) / effective_tokens_per_sample[i]
                    # TL_c = torch.sum(iterm_theta) / m
                    TL_b = torch.sum(Q_ijk) / effective_tokens_per_sample[i]
                    loss_per_sample[i] = alpha * TL_b + (1 - alpha) * TL_c
                else:
                    loss_per_sample[i] = ((effective_tokens_per_sample[i] - m) * lam + torch.sum(iterm_theta)) / effective_tokens_per_sample[i]
                    # loss_per_sample[i] = torch.sum(iterm_theta) / m
            else:
                loss_per_sample[i] = torch.sum(Q_ijk) / effective_tokens_per_sample[i]
        for i, w in enumerate(unique_weights):
            row_idx = torch.squeeze(weights) == w.item()
            loss += torch.mean(loss_per_sample[row_idx])
        loss /= len(unique_weights)
    elif weighted_loss_mode.startswith('case3_plus'):
        loss = 0.0
        alpha = 0.5
        # beta = 0.9
        beta = float(weighted_loss_mode.split('_')[-1])
        if weighted_loss_mode.startswith('case3_plus_gpu'):
            if loss_mask is None:  # TODO:
                loss += torch.sum(losses) / effective_tokens
            else:
                total_Q_ijk = torch.masked_select(losses.contiguous().view(-1), loss_mask.contiguous().view(-1).bool())
                # mu = torch.mean(total_Q_ijk)
                mu = torch.tensor(np.nanmean(total_Q_ijk.clone().detach().cpu().numpy()), device=losses.device)
                a = torch.tensor(np.nanmax(total_Q_ijk.clone().detach().cpu().numpy()), device=losses.device)
            
        for i, w in enumerate(unique_weights):
            row_idx = torch.squeeze(weights) == w.item()
            task_effective_tokens = torch.sum(effective_tokens_per_sample[row_idx])
            if loss_mask is None:  # TODO:
                loss += torch.sum(losses[row_idx, :]) / task_effective_tokens
            else:
                task_losses = (losses * loss_mask)[row_idx, :]
                task_loss_mask = loss_mask[row_idx, :]
                task_Q_ijk = torch.masked_select(task_losses.contiguous().view(-1), task_loss_mask.contiguous().view(-1).bool())
            
            if task_effective_tokens > 10:
                if not weighted_loss_mode.startswith('case3_plus_gpu'):
                    # mu = torch.mean(task_Q_ijk)
                    mu = torch.tensor(np.nanmean(task_Q_ijk.clone().detach().cpu().numpy()), device=losses.device)
                    # a = torch.max(task_Q_ijk).item()
                    a = torch.tensor(np.nanmax(task_Q_ijk.clone().detach().cpu().numpy()), device=losses.device)
                # lam = (mu + a) / 2  # 0.25 outliers
                # lam = (lam + a ) / 2  # 0.125 outliers
                lam = mu + beta * (a - mu)  # beta = 0.9 -> outliers = 0.05
                m = torch.sum(task_Q_ijk.le(lam)).item()
                iterm_theta = torch.where(task_Q_ijk > lam, 0, task_Q_ijk)
                loss += ((task_effective_tokens - m) * lam + torch.sum(iterm_theta)) / task_effective_tokens

                # loss += torch.sum((losses * loss_mask)[row_idx, :]) / torch.sum(loss_mask[row_idx, :])
            else:
                loss += torch.sum(task_Q_ijk) / task_effective_tokens
        loss /= len(unique_weights)
    elif weighted_loss_mode.startswith('case10') or weighted_loss_mode.startswith('case11'):  # case10: per gpu
        loss = 0.0
        alpha = 0.5
        # beta = 0.65
        beta = float(weighted_loss_mode.split('_')[-1])
        loss_per_sample = torch.zeros(bsz).to(device=losses.device)
        if loss_mask is None:  # TODO:
            loss += torch.sum(losses) / effective_tokens
        else:
            total_Q_ijk = torch.masked_select(losses.contiguous().view(-1), loss_mask.contiguous().view(-1).bool())
        loss_sum = torch.sum(total_Q_ijk)
        tokens_num = torch.sum(loss_mask)
        # a = torch.max(total_Q_ijk)
        a = torch.tensor(np.nanmax(total_Q_ijk.clone().detach().cpu().numpy()), device=losses.device)
        if weighted_loss_mode.startswith('case10_batch') or weighted_loss_mode.startswith('case11_batch'):
            loss_sum = torch.sum(total_Q_ijk)
            tokens_num = torch.sum(loss_mask)
            torch.distributed.all_reduce(loss_sum)
            torch.distributed.all_reduce(tokens_num)
            torch.distributed.all_reduce(a, op=torch.distributed.ReduceOp.MAX)
            mu = loss_sum / tokens_num
            # print(f'loss sum: {loss_sum}, tokens num: {tokens_num}, max: {a}')
        else:
            mu = torch.tensor(np.nanmean(total_Q_ijk.clone().detach().cpu().numpy()), device=losses.device)
            # mu = torch.mean(total_Q_ijk)
            # a = torch.max(total_Q_ijk).item()
        # lam = (mu + a) / 2  # 0.25 outliers
        # lam = (lam + a ) / 2  # 0.125 outliers
        lam = mu + beta * (a - mu)  # beta = 0.9 -> outliers = 0.05, beta = 0.8 -> outliers = 0.1
        for i in range(len(losses)):
            if loss_mask is None:
                Q_ijk = losses[i]
                mask = (labels != -100)
                Q_ijk = torch.masked_select(Q_ijk, mask)
            else:
                Q_ijk = losses[i] * loss_mask[i]
                Q_ijk = torch.masked_select(Q_ijk, loss_mask[i].bool())
            if effective_tokens_per_sample[i] > 10:
                m = torch.sum(Q_ijk.le(lam)).item()
                iterm_theta = torch.where(Q_ijk > lam, 0, Q_ijk)
                if weighted_loss_mode.startswith('case11'):
                    TL_c = ((effective_tokens_per_sample[i] - m) * lam + torch.sum(iterm_theta)) / effective_tokens_per_sample[i]
                    # TL_c = torch.sum(iterm_theta) / m
                    TL_b = torch.sum(Q_ijk) / effective_tokens_per_sample[i]
                    loss_per_sample[i] = alpha * TL_b + (1 - alpha) * TL_c
                else:
                    loss_per_sample[i] = ((effective_tokens_per_sample[i] - m) * lam + torch.sum(iterm_theta)) / effective_tokens_per_sample[i]
                    # loss_per_sample[i] = torch.sum(iterm_theta) / m
            else:
                loss_per_sample[i] = torch.sum(Q_ijk) / effective_tokens_per_sample[i]
        for i, w in enumerate(unique_weights):
            row_idx = torch.squeeze(weights) == w.item()
            loss += torch.mean(loss_per_sample[row_idx])
        loss /= len(unique_weights)
    elif weighted_loss_mode.startswith('threshold'):
        loss = 0.0
        gamma = int(weighted_loss_mode.split('_')[-1])
        threshold = float(weighted_loss_mode.split('_')[-2])
        losses_per_sample = torch.zeros(bsz).to(device=losses.device)
        for i in range(len(losses)):
            if loss_mask is None:
                Q_ijk = losses[i]
                mask = (labels != -100)
                Q_ijk = torch.masked_select(Q_ijk, mask)
            else:
                Q_ijk = losses[i] * loss_mask[i]
                Q_ijk = torch.masked_select(Q_ijk, loss_mask[i].bool())  # 把有效token截出来
            Q_ijk, _ = torch.sort(Q_ijk, descending=True)
            rank = torch.arange(len(Q_ijk))
            quantile = (rank + 0.5) / len(Q_ijk)
            norm_weight = 1 - (torch.abs(quantile - threshold) / threshold) ** gamma
            losses_per_sample[i] = torch.sum(norm_weight.to(device=losses.device) * Q_ijk)
        for i, w in enumerate(unique_weights):
            row_idx = torch.squeeze(weights) == w.item()
            loss += torch.sum(losses_per_sample[row_idx]) / torch.sum(effective_tokens_per_sample[row_idx])
        loss /= len(unique_weights)
    elif weighted_loss_mode.startswith('kde'):
        loss = 0.0
        task_loss = torch.zeros(len(ID2TASK)).to(device=task_id.device)  # 每个任务的loss
        task_num = torch.zeros(len(ID2TASK)).to(device=task_id.device)
        # sample_num = 1024
        for i, w in enumerate(unique_id):
            row_idx = torch.squeeze(task_id) == w.item()
            if loss_mask is None:  # TODO:
                pass
            else:
                task_losses = (losses * loss_mask)[row_idx, :]
                task_loss_mask = loss_mask[row_idx, :]
                # task_Q_ijk = torch.masked_select(task_losses.contiguous().view(-1), task_loss_mask.contiguous().view(-1).bool())
                task_Q_ijk = torch.masked_select(task_losses.view(-1), task_loss_mask.view(-1).bool())
                # task_Q_ijk = task_Q_ijk[~torch.isnan(task_Q_ijk)]
                task_Q_ijk = task_Q_ijk[torch.isfinite(task_Q_ijk)]
            task_effective_tokens = len(task_Q_ijk)

            task_Q_ijk, _ = torch.sort(task_Q_ijk)
            # sample_interval = task_effective_tokens // sample_num
            # sample_Q_ijk = task_Q_ijk[::sample_interval]

            # silverman bandwidth
            if weighted_loss_mode.endswith('silverman'):
                q1 = torch.quantile(task_Q_ijk, 0.25)
                q3 = torch.quantile(task_Q_ijk, 0.75)
                iqr = q3 - q1
                sigma = (0.9 * torch.min(torch.sqrt(torch.var(task_Q_ijk)), iqr / 1.35) * pow(len(task_Q_ijk), -1 / 5))
                sigma = sigma.clone().detach().cpu()
                # print_rank_0(f'custom silverman bandwidth: {sigma}')
            else:
                # ISJ bandwidth
                # print(f'task_Q_ijk: {task_Q_ijk}')
                try:
                    sigma = FFTKDE(kernel='gaussian', bw='ISJ').fit(task_Q_ijk.clone().detach().cpu().numpy()).bw
                    sigma = torch.tensor(sigma)
                except ValueError:
                    sigma = FFTKDE(kernel='gaussian', bw='silverman').fit(task_Q_ijk.clone().detach().cpu().numpy()).bw
                    sigma = torch.tensor(sigma)
                # if torch.nonzero(task_Q_ijk).size(0) <= 15 or len(task_Q_ijk) <= 15:  # nonzero number
                #     sigma = torch.tensor(0.5)
                # else:
                #     sigma = FFTKDE(kernel='gaussian', bw='ISJ').fit(task_Q_ijk.clone().detach().cpu().numpy()).bw
                #     sigma = torch.tensor(sigma)
            
            distance = torch.cdist(task_Q_ijk.view(len(task_Q_ijk), -1).clone().detach().cpu(),
                                    task_Q_ijk.view(len(task_Q_ijk), -1).clone().detach().cpu())  # [n, n]
            
            # if len(task_Q_ijk) >= sample_num:
            #     task_Q_ijk, _ = torch.sort(task_Q_ijk)
            #     # sample_interval = task_effective_tokens // sample_num
            #     # sample_Q_ijk = task_Q_ijk[::sample_interval]

            #     # silverman bandwidth
            #     # q1 = torch.quantile(task_Q_ijk, 0.25)
            #     # q3 = torch.quantile(task_Q_ijk, 0.75)
            #     # iqr = q3 - q1
            #     # sigma = 0.9 * torch.min(torch.sqrt(torch.var(task_Q_ijk)), iqr / 1.35) * pow(len(task_Q_ijk), -1 / 5)

            #     # ISJ bandwidth
            #     sigma = FFTKDE(kernel='gaussian', bw='ISJ').fit(task_Q_ijk.clone().detach().cpu().numpy()).bw
            #     sigma = torch.tensor(sigma)
                
            #     distance = torch.cdist(task_Q_ijk.view(len(task_Q_ijk), -1).clone().detach().cpu(),
            #                            task_Q_ijk.view(len(task_Q_ijk), -1).clone().detach().cpu())  # [n, sample_num]
            # else:
            #     task_Q_ijk, _ = torch.sort(task_Q_ijk)
                
            #     # silverman bandwidth
            #     # q1 = torch.quantile(task_Q_ijk, 0.25)
            #     # q3 = torch.quantile(task_Q_ijk, 0.75)
            #     # iqr = q3 - q1
            #     # sigma = 0.9 * torch.min(torch.sqrt(torch.var(task_Q_ijk)), iqr / 1.35) * pow(task_effective_tokens, -1 / 5)
                
            #     # ISJ bandwidth
            #     sigma = FFTKDE(kernel='gaussian', bw='ISJ').fit(task_Q_ijk.clone().detach().cpu().numpy()).bw
            #     sigma = torch.tensor(sigma)
                
            #     distance = torch.cdist(task_Q_ijk.view(len(task_Q_ijk), -1).clone().detach().cpu(), 
            #                            task_Q_ijk.view(task_effective_tokens, -1).clone().detach().cpu())  # [n, n]
            gaussian_values = torch.exp(-0.5 * ((distance / sigma) ** 2))
            density = torch.tensor(np.nanmean(gaussian_values.clone().detach().numpy(), axis=1), device=losses.device)
            # token_weight = (density / torch.sum(density)).to(device=losses.device)  # normalize
            token_weight = F.softmax(density * 2, dim=-1).to(device=losses.device)  # softmax
            # print_rank_0(f'token weight, max: {torch.max(token_weight)}, min: {torch.min(token_weight)}, len: {len(token_weight)}, div: {torch.max(token_weight) / torch.min(token_weight)}')
            # token_weight = token_weight.to(device=losses.device)
            # print_rank_0(f'token weight, max: {torch.max(token_weight)}, min: {torch.min(token_weight)}, len: {len(token_weight)}')
            task_loss[w] = torch.sum(task_Q_ijk * token_weight)
            task_num[w] = len(torch.sum(loss_mask, dim=1)[row_idx])
            loss += task_loss[w]
        loss /= len(unique_id)
        return loss, task_loss.clone().detach(), task_num, famo, selfpaced_status


    # 固定一个打印顺序
    task_loss = torch.zeros(len(ID2TASK)).to(device=task_id.device)  # 每个任务的loss
    task_num = torch.zeros(len(ID2TASK)).to(device=task_id.device)
    # unique_id = torch.unique(task_id)
    for i, w in enumerate(unique_id):
        row_idx = torch.squeeze(task_id) == w.item()
        if loss_mask is None:
            task_loss[w] = torch.sum(token_losses[row_idx, :]) / torch.sum(effective_tokens_per_sample[row_idx])
            task_num[w] = len(effective_tokens_per_sample[row_idx])
        else:  # TODO:
            task_loss[w] = torch.sum((token_losses * loss_mask)[row_idx, :]) / torch.sum(loss_mask[row_idx, :])
            task_num[w] = len(torch.sum(loss_mask, dim=1)[row_idx])
    
    # task_token_losses = torch.matmul(task_mask_trans, token_losses)  # [M * B] [B * L] = [M * L]
    # task_losses = torch.sum(task_token_losses, dim=1)  # [M,]
    # task_loss, task_num = get_task_loss(task_losses, task_id)
    
    return loss, task_loss, task_num, famo, selfpaced_status


def get_attn_mask(seq_length, device):
    """
    Get triangular attention mask for a given sequence length / device.
    """
    # lower triangular attention mask
    mask = torch.tril(torch.ones((1, seq_length, seq_length), device=device)).view(
        1, 1, seq_length, seq_length
    )

    # convert to binary
    return mask < 0.5


def get_ltor_masks_and_position_ids(data):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    # attention_mask = get_attn_mask(
    #     seq_length=seq_length,
    #     device=data.device,
    # )
    attention_mask = torch.ones((batch_size, seq_length), device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data).clone()

    return attention_mask, position_ids


def prepare_gpt_input(batch, device):
    batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}

    # input_ids = batch['input_ids'].long()

    # batch['input_ids'] = input_ids[:, :-1].contiguous().to(device=device)
    # batch['labels'] = input_ids[:, 1:].contiguous().to(device=device)
    if 'loss_mask' in batch and 'labels' not in batch:
    # if 'loss_mask' in batch:
        print_rank_0('loss mask in batch')
        input_ids = batch['input_ids'].long()
        batch['input_ids'] = input_ids[:, :-1].contiguous().to(device=device)
        batch['labels'] = input_ids[:, 1:].contiguous().to(device=device)
        batch['loss_mask'] = batch['loss_mask'].float()[:, 1:].contiguous()
    else:
        batch['input_ids'] = batch['input_ids'].long()
        batch['labels'] = batch['labels'].long()
        batch['loss_mask'] = None

    # Get the masks and position ids.
    batch['attention_mask'], batch['position_ids'] = get_ltor_masks_and_position_ids(data=batch['input_ids'])

    if self.args.weighted_loss_mode:
        weights = batch['weight'].float().to(device=device)  # [2, 4, 6, 3, ..., 2]
        # batch['loss_mask'] *= weights
    
    if 'task_id' in batch and batch['task_id'] is not None:  # task_id: bsz * 1, [[1], [2], [4], [3], ..., [1]]
        batch['task_mask'] = get_task_mask(batch['task_id']).to(device=device)  # bsz * task_num

    return batch

def prepare_glm_input(batch, device):
    batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}
    # if self.args.weighted_loss_mode:
    #     batch['weight'] = torch.squeeze(batch['weight'])
    
    batch['loss_mask'] = None

    if 'task_id' in batch and batch['task_id'] is not None:
        batch['task_mask'] = get_task_mask(batch['task_id']).to(device=device)  # bsz * task_num
    
    return batch


@dataclass
class DataCollatorForMFTDataset(object):
    def __init__(self, model_type, weighted_loss_mode, use_dynamic_padding):
        self.model_type = model_type
        self.weighted_loss_mode = weighted_loss_mode
        self.use_dynamic_padding = use_dynamic_padding

    # tokenizer: None

    def __call__(self, instances):
        input_ids, attention_mask, position_ids, labels, loss_mask, weights, task_id = tuple(
            [instance[key] if key in instance else None for instance in instances] for key in
            ("input_ids", "attention_mask", "position_ids", "labels", "loss_mask", "weight", "task_id"))
        # input_ids, loss_mask, weights, task_id = tuple(instances[key] for key in ("input_ids", "loss_mask", "weight", "task_id"))

        result_batch = {}
        '''
        outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                # labels=(batch['labels'], batch['loss_mask'], batch['task_mask']),
                # labels=(batch['labels'], batch['loss_mask']),
                position_ids=batch['position_ids'],
            )
        '''

        input_ids = torch.tensor(np.array(input_ids)).long()
        # input_ids = input_ids.long()
        if loss_mask[0] is None:
            result_batch['input_ids'] = input_ids.contiguous()
            labels = torch.tensor(np.array(labels)).long()
            result_batch['labels'] = labels.contiguous()
            result_batch['loss_mask'] = None
        else:
            loss_mask = torch.tensor(np.array(loss_mask))
            if self.use_dynamic_padding:
                last_one_pos = (loss_mask == 1).long().cumsum(dim=1).argmax(dim=1)
                # 取所有行的位置的最大值
                max_pos = last_one_pos.max().item() + 1
            else:
                max_pos = loss_mask.shape[-1]
            result_batch['input_ids'] = input_ids[:, :max_pos-1].contiguous()  # [B, L + 1] -> [B, L]
            result_batch['labels'] = input_ids[:, 1:max_pos].contiguous()
            result_batch['loss_mask'] = loss_mask.float()[:, 1:max_pos].contiguous()
            # result_batch['input_ids'] = input_ids[:, :-1].contiguous()  # [B, L + 1] -> [B, L]
            # result_batch['labels'] = input_ids[:, 1:].contiguous()
            # loss_mask = torch.tensor(np.array(loss_mask))
            # result_batch['loss_mask'] = loss_mask.float()[:, 1:].contiguous()

        if self.weighted_loss_mode and weights is not None:
            weights = torch.tensor(np.array(weights))
            result_batch['weights'] = weights
            # if result_batch['loss_mask'] is not None:
            #     result_batch['loss_mask'] *= weights
        
        # Get the masks and position ids.
        if self.model_type == 'glm':
            result_batch['attention_mask'], result_batch['position_ids'] = torch.tensor(np.array(attention_mask)).long(), torch.tensor(np.array(position_ids)).long()
        else:
            result_batch['attention_mask'], result_batch['position_ids'] = get_ltor_masks_and_position_ids(data=result_batch['input_ids'])

        if task_id is not None:
            task_id = torch.tensor(np.array(task_id))
            result_batch['task_mask'] = get_task_mask(task_id) # bsz * task_num
            result_batch['task_id'] = task_id

        return result_batch


class FAMO:
    """
    Fast Adaptive Multitask Optimization.
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        mode: str = 'famo_valid',
        gamma: float = 0.001,   # the regularization coefficient,   default: 0.001
        w_lr: float = 0.025,   # the learning rate of the task logits,   default: 0.025
        max_norm: float = 1.0, # the maximum gradient norm
    ):
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = n_tasks
        self.device = device
        self.first_train_step = True
        self.first_valid_step = True
        self.print_loss = None
        self.mode = mode
        self.prev_train_loss = None
        self.prev_valid_loss = None
        self.ratio_valid_task_loss_prev = torch.zeros(len(ID2TASK)).to(device)
        self.global_steps = 0
        self.z = None
    
    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses):
        self.prev_train_loss = losses
        self.z = F.softmax(self.w * 1, -1)
        # if is_main_process() and (self.global_steps % 10 == 0):
        #     logger.info(f"complete_steps: {self.global_steps}, per_task_weight: {self.z}")
        if -1e20 in self.ratio_valid_task_loss_prev and self.mode == 'famo_valid_ema':
            self.z = F.softmax(torch.where(self.ratio_valid_task_loss_prev == -1e20, -1e20, self.z), -1)
            if self.global_steps % 10 == 0:
                print_rank_0(f'ratio_valid_task_loss_prev is {self.ratio_valid_task_loss_prev}, after, z is {self.z}')
        D = losses - self.min_losses + 1e-8
        if self.mode.startswith('famo_train'):
            c = (self.z / D).sum().detach()
            loss = (D.log() * self.z / c).sum()
        else:
            loss = (D * self.z).sum()
        return loss

    def update(self, curr_loss):
        if self.mode.startswith('famo_valid') and self.first_valid_step:
            self.first_valid_step = False
            self.prev_valid_loss = curr_loss
            return
        if self.mode.startswith('famo_train'):
            prev_loss = self.prev_train_loss
        else:
            prev_loss = self.prev_valid_loss
            self.prev_valid_loss = curr_loss
        delta = (prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss      - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        Returns
        -------
        Loss, extra outputs
        """
        loss = self.get_weighted_loss(losses=losses)
        # if self.max_norm > 0 and shared_parameters is not None:
            # torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        # loss.backward()
        return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='none', ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, labels):

        if self.ignore_index:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), weight=self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        else:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), weight=self.weight, reduction=self.reduction)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)

        loss = 0.0
        if self.reduction == "none":
            loss = focal_loss
        elif self.reduction == "mean":
            loss = torch.mean(focal_loss)
        elif self.reduction == "sum":
            loss = torch.sum(focal_loss)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        
        return loss


class SelfPacedStatus:
    def __init__(self, interval=20):
        super(SelfPacedStatus, self).__init__()
        self.complete_steps = None
        self.current_epoch = None
        self.mode = None
        self.task_loss_prev = None
        self.w = None
        self.interval = interval
    
    def update(self, complete_steps, current_epoch, mode, task_loss_prev):
        self.complete_steps = complete_steps
        self.current_epoch = current_epoch
        self.mode = mode
        self.task_loss_prev = task_loss_prev


class SelfPacedLoss(torch.nn.Module):
    def __init__(self, complete_steps=None, current_epoch=None):
        super(SelfPacedLoss, self).__init__()
        # use global step to set this parameter at each iteration and start from 1
        self.complete_steps = complete_steps
        self.current_epoch = current_epoch
        self.ignore_index = -100

    def forward(self, logits, labels, task_id, loss_mask=None, selfpaced_status=None):
        # logits: (B, L, C), lables: (B, L), task_id: (B, 1), task_loss_prev: (N,)
        labels = labels.to(device=logits.device)
        task_id = task_id.to(device=logits.device)
        task_loss_prev = None
        if selfpaced_status is not None and selfpaced_status.task_loss_prev is not None:
            task_loss_prev = selfpaced_status.task_loss_prev.to(device=logits.device)
        if loss_mask is not None:
            loss_mask = loss_mask.to(device=logits.device)

        if selfpaced_status and selfpaced_status.mode == "train":
            scale_factor = 1 / torch.max(torch.abs(torch.where(task_loss_prev == -1.0000e+20, 0, task_loss_prev)) + 1e-10) * 5
            # custom gumbel softmax
            gumbels = (
                -torch.empty_like(task_loss_prev * scale_factor, memory_format=torch.legacy_contiguous_format).exponential_().log()
            )  # ~Gumbel(0,1)
            gumbels = task_loss_prev * scale_factor + gumbels / pow(self.current_epoch, 2)
            selfpaced_status.w = F.softmax(gumbels, dim=-1)

            # selfpaced_status.w = F.softmax(task_loss_prev * scale_factor, dim=-1)

            w = selfpaced_status.w
        elif selfpaced_status and selfpaced_status.mode == "valid" and task_loss_prev is not None: # valid
            scale_factor = 1 / torch.max(torch.abs(torch.where(task_loss_prev == -1.0000e+20, 0, task_loss_prev)) + 1e-10) * 5
            selfpaced_status.w = F.softmax(task_loss_prev * scale_factor, dim=-1)
            w = selfpaced_status.w
        else:
            w = F.softmax(torch.zeros(len(ID2TASK)).to(device=task_id.device), dim=-1)

        task_id = task_id.squeeze()
        unique_id = torch.unique(task_id)
        
        if loss_mask is None:
            loss_func = CrossEntropyLoss(reduction='none', ignore_index=-100)
            padding_mask = labels.eq(self.ignore_index)
        else:
            loss_func = CrossEntropyLoss(reduction='none')
        losses = loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
        losses = losses.view(labels.shape).contiguous()
        token_losses = losses.clone().detach()

        task_loss = torch.zeros(len(ID2TASK)).to(device=task_id.device)  # loss of each task
        task_num = torch.zeros(len(ID2TASK)).to(device=task_id.device)
        loss = 0.0
        for uid in unique_id:
            row_idx = task_id == uid.item()
            if loss_mask is None:
                loss += torch.sum(losses[row_idx, :]) / torch.sum(padding_mask[row_idx, :]) * w[uid]
                task_loss[uid] = torch.sum(token_losses[row_idx, :]) / torch.sum(padding_mask[row_idx, :])
            else:
                loss += torch.sum((losses * loss_mask)[row_idx, :]) / torch.sum(loss_mask[row_idx, :]) * w[uid]
                task_loss[uid] = torch.sum((token_losses * loss_mask)[row_idx, :]) / torch.sum(loss_mask[row_idx, :])
            task_num[uid] = torch.sum(row_idx)
        
        # if selfpaced_status and self.complete_steps % selfpaced_status.interval == 0 and selfpaced_status.mode == 'train':
        #     if is_main_process():
        #         logger.info(f"complete_steps: {self.complete_steps}, ema ratio: {task_loss_prev}, self paced loss weight: {w}")

        return loss, task_loss, task_num, selfpaced_status
