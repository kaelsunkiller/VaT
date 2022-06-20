#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   VAEModel.py    
@Contact :   kael.sunkiller@gmail.com
@License :   (C)Copyright 2020, Leozen-Yang

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
6/8/20 6:00 PM   Yang      0.0         None
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

import copy
import math
import numpy as np

from .TransformerModel import TransformerModel, MultiHeadedAttention, PositionwiseFeedForward, \
    Encoder, EncoderLayer, Decoder, DecoderLayer, Embeddings, Generator, subsequent_mask, pack_wrapper, \
    SublayerConnection, clones, LayerNorm
from ..modules.losses import LanguageModelCriterion

import pdb


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def normal_kl_mixtured(mu1, lv1, c1, mu2, lv2, c2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl_i = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    kl_c = (c1 * (c1.log() - c2.log())).sum()
    kl = (c1 * kl_i).sum(0)
    return kl_c, kl


class VAE(nn.Module):
    """
    A standard CVAE architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, prior_encoder, poster_encoder, tgt_embed, pos_embed, generator):
        super(VAE, self).__init__()
        self.src_encoder = encoder
        self.decoder = decoder
        self.prior = prior_encoder
        self.poster = poster_encoder
        self.tgt_embed = tgt_embed
        self.pos_embed = pos_embed
        self.generator = generator
        self.crit = LanguageModelCriterion(opt={'use_ghmloss': False})

    def forward(self, src, tgt, src_mask, tgt_mask, tgt_full_mask):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, src_mask)
        tgt_embeding = self.tgt_embed(tgt)
        pos_embeding = self.pos_embed(tgt).expand(tgt_embeding.size())

        position = self.prior(tgt_embeding, memory, src_mask, tgt_full_mask.unsqueeze(-2).expand(tgt_mask.size())).detach()
        pos_embeding_inf, analytic_kl = self.poster(pos_embeding, position, memory, src_mask, tgt_mask)
        # inputs = self.pos_embed.prepare_input(tgt_embeding, position, pos_embeding_inf)
        inputs = tgt_embeding + pos_embeding_inf
        output = self.decoder(inputs, memory, src_mask, tgt_mask)
        # out = output.clone().scatter_(1, position.unsqueeze(-1).expand(output.size()), output)

        # pos_loss = self.crit(pos_logprob, position, tgt_full_mask)
        return output, analytic_kl

    def encode(self, src, src_mask):
        return self.src_encoder(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt_embeding = self.tgt_embed(tgt)
        pos_embeding = self.pos_embed(tgt).expand(tgt_embeding.size())

        pos_embeding_inf = self.poster.decode(pos_embeding, memory, src_mask, tgt_mask)
        inputs = tgt_embeding + pos_embeding_inf
        output = self.decoder(inputs, memory, src_mask, tgt_mask)
        return output


class SimpleDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, inplace=True, gmm_num=0):
        super(SimpleDecoder, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.norm = nn.BatchNorm1d(in_dim)
        self.fc = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.gmm_num = gmm_num

    def forward(self, z):
        if 3 == len(z.shape):
            out = self.norm(z.permute(0, 2, 1)).permute(0, 2, 1)
        elif 2 == len(z.shape):
            out = self.norm(z)
        else:
            out = z
        out = self.relu(out)
        out = self.fc(out)
        if self.gmm_num:
            out = out.contiguous().view(*out.shape[:-1], -1, self.gmm_num, 2).permute(3, 0, 1, 2, 4).squeeze()
        return self.dropout(out)


class Prior(nn.Module):
    def __init__(self):
        super(Prior, self).__init__()

    def forward(self, x, memory, src_mask, tgt_mask):
        x, m = x.detach(), memory.detach()
        B, N = x.shape[:2]
        xm_attn = self._attn(x, m, src_mask)
        xx_attn = self._attn(x, x, tgt_mask)
        xm_attn = xm_attn.max(-1)[0].unsqueeze(1).expand(xx_attn.size())
        if tgt_mask is not None:
            xm_attn = xm_attn.masked_fill(tgt_mask == 0, 0.)
        p_attn = xx_attn + xm_attn  # BxNxN
        pos = x.new_zeros(B, N).long()
        for i in range(N-1):
            p_attn[:, i, i] = 0.
            ppr = p_attn.gather(1, pos[:, :i+1].unsqueeze(-1).expand(B, i+1, N)).sum(1)  # BxN
            pos[:, i+1] = ppr.argmax(-1) # B
            for b in range(B):
                p_attn[b, pos[b, i+1]] = 0.
        out = pos.clone()
        for i in range(1, N):
            for b in range(B):
                out[b, pos[b, i]] = i
        return out.long()

    def _attn(self, query, key, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        p_attn = F.softmax(scores, dim=-1)
        return p_attn

class Poster(nn.Module):
    def __init__(self, prio_encoder, post_encoder, post_mlp, decoder, gmm_num=0):
        super(Poster, self).__init__()
        self.prio_encoder = prio_encoder
        self.post_encoder = post_encoder
        self.post_mlp = post_mlp
        self.decoder = decoder
        # self.generator = generator
        self.gmm_num = gmm_num
        if self.gmm_num > 1:
            self.gmmn_gt = nn.Parameter(torch.ones(self.gmm_num, 1, 1, 1).float(), requires_grad=True)
            self.gmmn_c = nn.Parameter(torch.ones(self.gmm_num, 1, 1, 1).float(), requires_grad=True)

    def forward(self, pos_embeding, position, memory, src_mask, tgt_mask):
        ppos = position.unsqueeze(-1).expand(pos_embeding.size())
        ppos_embeding = pos_embeding.gather(1, ppos)

        gt_distribution_params = self.prio_encoder(ppos_embeding)
        c_distribution_params = self.post_mlp(self.post_encoder(pos_embeding, memory, src_mask, tgt_mask))
        gt_mean, gt_logvar = gt_distribution_params[..., 0], gt_distribution_params[..., 1]
        c_mean, c_logvar = c_distribution_params[..., 0], c_distribution_params[..., 1]
        epsilon = torch.randn(gt_mean.shape).to(gt_mean.device)
        sample_z_gt = epsilon * torch.exp(.5 * gt_logvar) + gt_mean
        if self.gmm_num > 1:
            gmmn_gt = torch.softmax(self.gmmn_gt, dim=0)
            gmmn_c = torch.softmax(self.gmmn_c, dim=0)
            sample_z_gt = sample_z_gt.index_select(0, torch.multinomial(gmmn_gt.squeeze(), 1)).squeeze()
            # kl_c, analytic_kl = normal_kl_mixtured(gt_mean, gt_logvar, gmmn_gt,
            #                                        c_mean, c_logvar, gmmn_c)
            kl_c, analytic_kl = normal_kl_mixtured(c_mean, c_logvar, gmmn_c,
                                                   gt_mean, gt_logvar, gmmn_gt)
            analytic_kl = kl_c + analytic_kl.sum(-1).mean()
        else:
            analytic_kl = normal_kl(gt_mean, gt_logvar, c_mean, c_logvar).sum(-1).mean()

        output = self.decoder(sample_z_gt)
        # logprob = self.generator(output)

        # return logprob.argmax(-1), logprob, sample_z_gt, analytic_kl
        return output, analytic_kl

    def decode(self, pos_embeding, memory, src_mask, tgt_mask):
        distribution_params = self.post_mlp(self.post_encoder(pos_embeding, memory, src_mask, tgt_mask))
        mean, logvar = distribution_params[..., 0], distribution_params[..., 1]
        epsilon = torch.randn(mean.shape).to(mean.device)
        sample_z = epsilon * torch.exp(.5 * logvar) + mean
        if self.gmm_num > 1:
            gmmn = torch.softmax(self.gmmn_c, dim=0)
            sample_z = sample_z.index_select(0, torch.multinomial(gmmn.squeeze(), 1)).squeeze()
        if sample_z.ndim == 2:
            sample_z = sample_z.unsqueeze(1)

        output = self.decoder(sample_z)
        # logprob = self.generator(output)
        # return logprob.argmax(-1), logprob, sample_z
        return output


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

    def prepare_input(self, tgt_embeding, position, pos_embeding):
        ppos = position.unsqueeze(-1).expand(pos_embeding.size())
        out = tgt_embeding.gather(1, ppos) + pos_embeding
        return self.dropout(out)


class VAETModel(TransformerModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1, gmm_num=1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        z_decoder = SimpleDecoder(d_model, d_model * 2 * gmm_num, dropout, gmm_num=gmm_num)
        model = VAE(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N_dec),
            Prior(),
            Poster(c(z_decoder),
                   Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N_dec),
                   c(z_decoder),
                   SimpleDecoder(d_model, d_model, dropout),
                   gmm_num=gmm_num
                   ),
            Embeddings(d_model, tgt_vocab),
            c(position),
            Generator(d_model, tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(VAETModel, self).__init__(opt)

        self.gmm_num = getattr(opt, 'gmm_num', 1)
        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(0, tgt_vocab,
                                     N_enc=self.N_enc,
                                     N_dec=self.N_dec,
                                     d_model=self.d_model,
                                     d_ff=self.d_ff,
                                     h=self.h,
                                     dropout=self.dropout,
                                     gmm_num=self.gmm_num)

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask, seq_mask_s = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[...,:0], att_feats[...,:0], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            # seq = seq[:,:-1]
            seq_mask_s = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
            seq_mask_s[:,0] = 1 # bos

            seq_mask = seq_mask_s.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                    [att_feats, att_masks]
                )
        else:
            seq_mask_s = None
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask, seq_mask_s

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        att_feats, seq, att_masks, seq_mask, seq_mask_s = self._prepare_feature_forward(att_feats, att_masks, seq)

        out, analytic_kl= self.model(att_feats, seq, att_masks, seq_mask, seq_mask_s)

        outputs = self.model.generator(out)
        return outputs, analytic_kl

    # def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1):
    #     # 'it' contains a word index
    #     xt = self.embed(it)
    #
    #     output, state, pos = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
    #     if output_logsoftmax:
    #         logprobs = F.log_softmax(self.logit(output), dim=1)
    #     else:
    #         logprobs = self.logit(output)
    #
    #     return logprobs, state, pos
    #
    # def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
    #     """
    #     state = [ys.unsqueeze(0)]
    #     """
    #     if len(state) == 0:
    #         ys = it.unsqueeze(1)
    #     else:
    #         ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
    #     out, pos_inf = self.model.decode(memory, mask,
    #                            ys,
    #                            subsequent_mask(ys.size(1))
    #                                     .to(memory.device))
    #     return out[:, -1], [ys.unsqueeze(0)], pos_inf[:, -1]
    #
    # def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
    #
    #     sample_method = opt.get('sample_method', 'greedy')
    #     beam_size = opt.get('beam_size', 1)
    #     temperature = opt.get('temperature', 1.0)
    #     sample_n = int(opt.get('sample_n', 1))
    #     group_size = opt.get('group_size', 1)
    #     output_logsoftmax = opt.get('output_logsoftmax', 1)
    #     decoding_constraint = opt.get('decoding_constraint', 0)
    #     block_trigrams = opt.get('block_trigrams', 0)
    #     remove_bad_endings = opt.get('remove_bad_endings', 0)
    #     if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
    #         return self._sample_beam(fc_feats, att_feats, att_masks, opt)
    #     if group_size > 1:
    #         return self._diverse_sample(fc_feats, att_feats, att_masks, opt)
    #
    #     batch_size = fc_feats.size(0)
    #     state = self.init_hidden(batch_size * sample_n)
    #
    #     p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
    #
    #     if sample_n > 1:
    #         p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(sample_n,
    #                                                                                   [p_fc_feats, p_att_feats,
    #                                                                                    pp_att_feats, p_att_masks]
    #                                                                                   )
    #
    #     trigrams = []  # will be a list of batch_size dictionaries
    #
    #     seq = fc_feats.new_full((batch_size * sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
    #     seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size + 1)
    #     positions = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, dtype=torch.long)
    #     for t in range(self.seq_length + 1):
    #         if t == 0:  # input <bos>
    #             it = fc_feats.new_full([batch_size * sample_n], self.bos_idx, dtype=torch.long)
    #
    #         logprobs, state, pos = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state,
    #                                                   output_logsoftmax=output_logsoftmax)
    #
    #         if decoding_constraint and t > 0:
    #             tmp = logprobs.new_zeros(logprobs.size())
    #             tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
    #             logprobs = logprobs + tmp
    #
    #         if remove_bad_endings and t > 0:
    #             tmp = logprobs.new_zeros(logprobs.size())
    #             prev_bad = np.isin(seq[:, t - 1].data.cpu().numpy(), self.bad_endings_ix)
    #             # Make it impossible to generate bad_endings
    #             tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
    #             logprobs = logprobs + tmp
    #
    #         # Mess with trigrams
    #         # Copy from https://github.com/lukemelas/image-paragraph-captioning
    #         if block_trigrams and t >= 3:
    #             # Store trigram generated at last step
    #             prev_two_batch = seq[:, t - 3:t - 1]
    #             for i in range(batch_size):  # = seq.size(0)
    #                 prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
    #                 current = seq[i][t - 1]
    #                 if t == 3:  # initialize
    #                     trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
    #                 elif t > 3:
    #                     if prev_two in trigrams[i]:  # add to list
    #                         trigrams[i][prev_two].append(current)
    #                     else:  # create list
    #                         trigrams[i][prev_two] = [current]
    #             # Block used trigrams at next step
    #             prev_two_batch = seq[:, t - 2:t]
    #             mask = torch.zeros(logprobs.size(), requires_grad=False).to(logprobs.device)  # batch_size x vocab_size
    #             for i in range(batch_size):
    #                 prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
    #                 if prev_two in trigrams[i]:
    #                     for j in trigrams[i][prev_two]:
    #                         mask[i, j] += 1
    #             # Apply mask to log probs
    #             # logprobs = logprobs - (mask * 1e9)
    #             alpha = 2.0  # = 4
    #             logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)
    #
    #         # sample the next word
    #         if t == self.seq_length:  # skip if we achieve maximum length
    #             break
    #         it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)
    #
    #         # stop when all finished
    #         if t == 0:
    #             unfinished = it != self.eos_idx
    #         else:
    #             it[~unfinished] = self.pad_idx  # This allows eos_idx not being overwritten to 0
    #             logprobs = logprobs * unfinished.unsqueeze(1).to(logprobs)
    #             unfinished = unfinished & (it != self.eos_idx)
    #         seq[:, t] = it
    #         seqLogprobs[:, t] = logprobs
    #         positions[:, t] = pos
    #         # quit loop if all sequences have finished
    #         if unfinished.sum() == 0:
    #             break
    #     seq = seq.clone().scatter_(1, positions, seq)
    #     seqLogprobs = seqLogprobs.clone().scatter_(1, positions.unsqueeze(-1).expand(seqLogprobs.size()), seqLogprobs)
    #
    #     return seq, seqLogprobs