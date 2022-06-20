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

from .TransformerModel import TransformerModel, MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, \
    Encoder, EncoderLayer, Decoder, DecoderLayer, Embeddings, Generator, subsequent_mask, pack_wrapper, \
    SublayerConnection, clones, LayerNorm


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

    def __init__(self, att_feats_encoder, gt_encoder, decoder, ge_mlp, ce_mlp, ce_layer, tgt_embed, generator):
        super(VAE, self).__init__()
        self.src_encoder = att_feats_encoder
        self.gt_encoder = gt_encoder
        self.decoder = decoder
        self.ge_mlp = ge_mlp
        self.ce_mlp = ce_mlp
        self.ce_gen = ce_layer
        self.tgt_embed = tgt_embed
        self.generator = generator

        self.gmm_num = ge_mlp.gmm_num
        if self.gmm_num > 1:
            self.gmmn_gt = nn.Parameter(torch.ones(self.gmm_num, 1, 1, 1).float(), requires_grad=True)
            self.gmmn_c = nn.Parameter(torch.ones(self.gmm_num, 1, 1, 1).float(), requires_grad=True)

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, src_mask)
        # tgt_full_mask = tgt_mask[:, -1:, :].data.clone().repeat(1, tgt_mask.size(1), 1)

        gt_distribution_params = self.ge_mlp(self.gt_encoder(self.tgt_embed(tgt), tgt_mask))
        c_distribution_params = self.ce_mlp(self.ce_gen(memory, src_mask, tgt.shape[1]))
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

        # decoder_memo = torch.cat([memory, sample_z_gt], 1)
        # decoder_memo_masks = torch.cat([src_mask.repeat(1, tgt_mask.size(1), 1), tgt_mask.type_as(src_mask)], -1)
        out = self.decoder(sample_z_gt, memory, src_mask, tgt_mask)
        return out, analytic_kl

    def get_gtz_params(self, tgt, tgt_mask):
        tgt_full_mask = tgt_mask[:, -1:, :].data.clone().repeat(1, tgt_mask.size(1), 1)

        gt_distribution_params = self.ge_mlp(self.gt_encoder(self.tgt_embed(tgt), tgt_full_mask))
        return gt_distribution_params

    def get_cz_params(self, memory, tgt_len):
        return self.ce_gen(memory, tgt_len)

    def encode(self, src, src_mask):
        return self.src_encoder(src, src_mask)

    def decode(self, distribution_params, memory, src_mask, tgt_mask, tgt=None):
        mean, logvar = distribution_params[..., 0], distribution_params[..., 1]
        epsilon = torch.randn(mean.shape).to(mean.device)
        sample_z = epsilon * torch.exp(.5 * logvar) + mean
        if self.gmm_num > 1:
            gmmn = self.gmmn_gt if tgt else self.gmmn_c
            gmmn = torch.softmax(gmmn, dim=0)
            sample_z = sample_z.index_select(0, torch.multinomial(gmmn.squeeze(), 1)).squeeze()
        if sample_z.ndim == 2:
            sample_z = sample_z.unsqueeze(1)

        # decoder_memo = torch.cat([memory, sample_z_c], 1)
        # decoder_memo_masks = torch.cat([src_mask.repeat(1, tgt_mask.size(1), 1), tgt_mask.type_as(src_mask).repeat(src_mask.size(0), 1, 1)], -1)
        return self.decoder(sample_z, memory, src_mask, tgt_mask)


class SimpleDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, inplace=True, gmm_num=0):
        super(SimpleDecoder, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.norm = nn.BatchNorm1d(in_dim)
        self.fc = nn.Linear(in_dim, out_dim)
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
        return out


class Fuse(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Fuse, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, src_mask, tgt_len):
        for layer in self.layers:
            x = layer(x, src_mask)
        return F.adaptive_avg_pool1d(self.norm(x).transpose(1, 2), tgt_len).transpose(1, 2)

class FuseLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, src_attn, feed_forward, dropout, max_length=20):
        super(FuseLayer, self).__init__()
        self.size = size
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.weight = nn.parameter.Parameter(torch.Tensor(max_length, size), requires_grad=True)


    def forward(self, src, src_mask):
        "Follow Figure 1 (right) for connections."
        x = self.weight.unsqueeze(0).expand(src.shape[0], *self.weight.shape)
        x = self.sublayer[0](x, lambda x: self.src_attn(x, src, src))
        return self.sublayer[1](x, self.feed_forward)


class VAETModel(TransformerModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1, gmm_num=1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = VAE(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            # Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N_dec),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N_dec),
            SimpleDecoder(d_model, d_model * 2 * gmm_num, gmm_num=gmm_num),
            SimpleDecoder(d_model, d_model * 2 * gmm_num, gmm_num=gmm_num),
            Fuse(FuseLayer(d_model, c(attn), c(ff), dropout, self.seq_length), N_dec),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
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

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            # seq = seq[:,:-1]
            seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
            seq_mask[:, 0] = 1  # bos

            if self.opt.fake_seq_start > 0 and self.opt.current_epoch >= self.opt.fake_seq_start:
                # print('fake')
                fake_masks = torch.lt(seq.data.clone().float().uniform_(), self.opt.fake_prob).long() * seq_mask.long()
                fake_masks[:, 0] = 0
                # is_fake = torch.gt(torch.rand(batch_size), 0.5).long().to(device)
                # fake_masks = fake_masks * is_fake.unsqueeze(1)
                fake_seq = torch.randint_like(seq, 1, self.vocab_size, requires_grad=False)
                seq = torch.where(fake_masks > 0, fake_seq, seq)

            seq_mask = seq_mask.unsqueeze(-2)
            # seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                                                            [att_feats, att_masks]
                                                            )
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)

        out, analytic_kl = self.model(att_feats, seq, att_masks, seq_mask)

        outputs = self.model.generator(out)
        return outputs, analytic_kl

    def _prepare_feature(self, fc_feats, att_feats, att_masks, seq=None):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[...,:0], att_feats[...,:0], memory, att_masks, seq, seq_mask

    def get_logprobs(self, output, t, output_logsoftmax=1):
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs[:, t]

    def _sample(self, fc_feats, att_feats, att_masks=None, seq=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)
        if group_size > 1:
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        if seq is not None:
            if seq.ndim == 3:  # B * seq_per_img * seq_len
                seq = seq.reshape(-1, seq.shape[2])

        batch_size = fc_feats.size(0)
        # state = self.init_hidden(batch_size*sample_n)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, seq, seq_mask = self._prepare_feature(fc_feats, att_feats, att_masks, seq)

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(sample_n,
                                                                                      [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
                                                                                      )

        trigrams = [] # will be a list of batch_size dictionaries

        if seq is not None:
            z_params = self.model.get_gtz_params(seq, seq_mask)
        else:
            z_params = self.model.get_cz_params(pp_att_feats, self.seq_length)
        z_masks = torch.ones(z_params.shape[:2]).unsqueeze(-2).to(z_params)
        out = self.model.decode(z_params, pp_att_feats, p_att_masks, z_masks, tgt=None)

        seq = fc_feats.new_full((batch_size*sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 1)

        for t in range(self.seq_length):
            # if t == 0: # input <bos>
            #     it = fc_feats.new_full([batch_size*sample_n], self.bos_idx, dtype=torch.long)

            logprobs = self.get_logprobs(out, t, output_logsoftmax=output_logsoftmax)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(seq[:,t-1].data.cpu().numpy(), self.bad_endings_ix)
                # Make it impossible to generate bad_endings
                tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp

            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:,t-3:t-1]
                for i in range(batch_size): # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = seq[i][t-1]
                    if t == 3: # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:,t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).to(logprobs.device) # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 2.0 # = 4
                logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx # This allows eos_idx not being overwritten to 0
                logprobs = logprobs * unfinished.unsqueeze(1).to(logprobs)
                unfinished = unfinished & (it != self.eos_idx)
            seq[:,t] = it
            seqLogprobs[:,t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, seq=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, seq, seq_mask = self._prepare_feature(fc_feats, att_feats, att_masks, seq)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = fc_feats.new_full((batch_size*sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 1)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]

        state = self.init_hidden(batch_size)

        # first step, feed bos
        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
        logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
                                                                                  [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
                                                                                  )
        self.done_beams = self.beam_search(state, logprobs, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, opt=opt)
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k*sample_n+_n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k*sample_n+_n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs

    # def _forward_sample(self, fc_feats, att_feats, seq, att_masks=None, opt={}):
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
    #     seq_length = opt.get('seq_length', self.seq_length)
    #     if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
    #         return self._sample_beam(fc_feats, att_feats, att_masks, opt)
    #     if group_size > 1:
    #         return self._diverse_sample(fc_feats, att_feats, att_masks, opt)
    #
    #     if seq.ndim == 3:  # B * seq_per_img * seq_len
    #         seq = seq.reshape(-1, seq.shape[2])
    #     f_att_feats, seq, f_att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
    #     gt_z_params = self.model.get_gt_z_params(f_att_feats, seq, f_att_masks, seq_mask)
    #
    #     batch_size = fc_feats.size(0)
    #     state = self.init_hidden(batch_size * sample_n)
    #
    #     _, _, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
    #
    #     if sample_n > 1:
    #         pp_att_feats, p_att_masks = utils.repeat_tensors(sample_n, [pp_att_feats, p_att_masks])
    #
    #     trigrams = []  # will be a list of batch_size dictionaries
    #
    #     seq = fc_feats.new_full((batch_size*sample_n, seq_length), self.pad_idx, dtype=torch.long)
    #     seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, seq_length, self.vocab_size + 1)
    #     for t in range(seq_length + 1):
    #         if t == 0:  # input <bos>
    #             it = fc_feats.new_full([batch_size*sample_n], self.bos_idx, dtype=torch.long)
    #
    #         logprobs, state = self.get_logprobs_state_forward(it, gt_z_params, pp_att_feats, p_att_masks, state,
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
    #             mask = torch.zeros(logprobs.size(), requires_grad=False).to(logprobs.device) # batch_size x vocab_size
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
    #         if t == seq_length:  # skip if we achieve maximum length
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
    #         it = it * unfinished.type_as(it)
    #         seq[:, t] = it
    #         seqLogprobs[:, t] = logprobs
    #         # quit loop if all sequences have finished
    #         if unfinished.sum() == 0:
    #             break
    #
    #     return seq, seqLogprobs
    #
    # def get_logprobs_state_forward(self, it, gt_z_params, p_att_feats, att_masks, state, output_logsoftmax=1):
    #     # 'it' contains a word index
    #     xt = self.embed(it)
    #
    #     output, state = self.core_forward(xt, gt_z_params, p_att_feats, state, att_masks)
    #     if output_logsoftmax:
    #         logprobs = F.log_softmax(self.logit(output), dim=1)
    #     else:
    #         logprobs = self.logit(output)
    #
    #     return logprobs, state
    #
    # def core_forward(self, it, gt_z_params, memory, state, mask):
    #     """
    #     state = [ys.unsqueeze(0)]
    #     """
    #     if len(state) == 0:
    #         ys = it.unsqueeze(1)
    #     else:
    #         ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
    #     out = self.model.l_decode(gt_z_params, memory, mask, ys,
    #                             subsequent_mask(ys.size(1)).to(memory.device))
    #     return out[:, -1], [ys.unsqueeze(0)]
    #
    # def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
    #     """
    #     state = [ys.unsqueeze(0)]
    #     """
    #     if len(state) == 0:
    #         ys = it.unsqueeze(1)
    #     else:
    #         ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
    #     out = self.model.decode(memory, mask,
    #                             ys,
    #                             subsequent_mask(ys.size(1))
    #                             .to(memory.device))
    #     return out[:, -1], [ys.unsqueeze(0)]