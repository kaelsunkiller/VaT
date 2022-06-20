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
    SublayerConnection, clones, LayerNorm, attention

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

    def __init__(self, att_feats_encoder, decoder, tgt_embed, generator):
        super(VAE, self).__init__()
        self.src_encoder = att_feats_encoder
        # self.gt_encoder = gt_encoder
        self.decoder = decoder
        # self.ge_mlp = ge_mlp
        # self.ce_mlp = ce_mlp
        # self.ce_self = ce_self
        # self.ce_src = ce_src
        self.tgt_embed = tgt_embed
        self.generator = generator

        # self.gmm_num = ge_mlp.gmm_num
        # if self.gmm_num > 1:
        #     self.gmm_emb_gt = gmm_embed
        #     self.gmm_emb_c = copy.deepcopy(gmm_embed)

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, src_mask)
        # tgt_full_mask = torch.ones_like(tgt_mask)
        # tgt_full_memo = self.gt_encoder(self.tgt_embed(tgt), tgt_full_mask)
        #
        # memo_att, _ = attention(self.tgt_embed(tgt), memory, memory, src_mask)
        # tgt_att, _ = attention(self.tgt_embed(tgt), tgt_full_memo, tgt_full_memo, tgt_full_mask)
        # gt_distribution_params = self.ge_mlp(tgt_att)
        # c_distribution_params = self.ce_mlp(memo_att)
        # gt_mean, gt_logvar = gt_distribution_params[..., 0], gt_distribution_params[..., 1]
        # c_mean, c_logvar = c_distribution_params[..., 0], c_distribution_params[..., 1]
        # epsilon = torch.randn(gt_mean.shape).to(gt_mean.device)
        # sample_z_gt = epsilon * torch.exp(.5 * gt_logvar) + gt_mean
        # if self.gmm_num > 1:
        #     sample_z_gt, gmmn_gt = self.gmm_emb_gt(sample_z_gt, self.tgt_embed(tgt).mean(1))
        #     gmmn_c = self.gmm_emb_c.get_gmm_prob(self.tgt_embed(tgt).mean(1))
        #     # sample_z_gt = sample_z_gt.index_select(0, torch.multinomial(gmmn_gt.squeeze(), 1)).squeeze()
        #     # kl_c, analytic_kl = normal_kl_mixtured(gt_mean, gt_logvar, gmmn_gt,
        #     #                                        c_mean, c_logvar, gmmn_c)
        #     kl_c, analytic_kl = normal_kl_mixtured(c_mean, c_logvar, gmmn_c.t()[(...,) + (None,)*2],
        #                                            gt_mean, gt_logvar, gmmn_gt.t()[(...,) + (None,)*2])
        #     analytic_kl = kl_c + analytic_kl.sum(-1).mean()
        # else:
        #     analytic_kl = normal_kl(gt_mean, gt_logvar, c_mean, c_logvar).sum(-1).mean()
        #
        # # decoder_memo = torch.cat([memory, sample_z_gt], 1)
        # # decoder_memo_masks = torch.cat([src_mask.repeat(1, tgt_mask.size(1), 1), tgt_mask.type_as(src_mask)], -1)
        # out = self.decoder(self.tgt_embed(tgt), sample_z_gt, tgt_mask, tgt_mask)
        out, analytic_kl = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        # analytic_kl = torch.stack(analytic_kl, 0).mean(0)
        return out, analytic_kl

    def get_gtz_params(self, tgt, tgt_mask):
        tgt_full_mask = torch.ones_like(tgt_mask)
        tgt_full_memo = self.gt_encoder(self.tgt_embed(tgt), tgt_full_mask)

        tgt_att, _ = attention(self.tgt_embed(tgt), tgt_full_memo, tgt_full_memo, tgt_full_mask)
        gt_distribution_params = self.ge_mlp(tgt_att)
        return gt_distribution_params

    def encode(self, src, src_mask):
        return self.src_encoder(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # memo_att, _ = attention(self.tgt_embed(tgt), memory, memory, src_mask)
        # distribution_params = self.ce_mlp(memo_att)
        # # distribution_params = self.ce_mlp(self.ce_gen(self.tgt_embed(tgt), memory, src_mask, tgt_mask))
        # mean, logvar = distribution_params[..., 0], distribution_params[..., 1]
        # epsilon = torch.randn(mean.shape).to(mean.device)
        # sample_z = epsilon * torch.exp(.5 * logvar) + mean
        # if self.gmm_num > 1:
        #     sample_z, _ = self.gmm_emb_c(sample_z, self.tgt_embed(tgt).mean(1))
        #
        # # decoder_memo = torch.cat([memory, sample_z], 1)
        # # decoder_memo_masks = torch.cat([src_mask.repeat(1, tgt_mask.size(1), 1), tgt_mask.type_as(src_mask).repeat(src_mask.size(0), 1, 1)], -1)
        # # decoder_memo = sample_z
        # # decoder_memo_masks = tgt_mask.type_as(src_mask).repeat(src_mask.size(0), 1, 1)
        return self.decoder.decode(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def l_decode(self, seq, seq_mask, memory, src_mask, tgt, tgt_mask):
        # mean, logvar = distribution_params[..., 0], distribution_params[..., 1]
        # epsilon = torch.randn(mean.shape).to(mean.device)
        # sample_z = epsilon * torch.exp(.5 * logvar) + mean
        # if self.gmm_num > 1:
        #     sample_z, _ = self.gmm_emb_gt(sample_z, self.tgt_embed(tgt).mean(2))
        #
        # # decoder_memo = torch.cat([memory, sample_z], 1)
        # # decoder_memo_masks = torch.cat([src_mask.repeat(1, sample_z.size(1), 1), tgt_mask[:, :sample_z.size(1)].type_as(src_mask).repeat(src_mask.size(0), 1, 1)], -1)
        # # decoder_memo = sample_z
        # # decoder_memo_masks = tgt_mask[:, :sample_z.size(1)].type_as(src_mask).repeat(src_mask.size(0), 1, 1)
        return self.decoder.l_decode(self.tgt_embed(tgt), self.tgt_embed(seq), memory, src_mask, tgt_mask, seq_mask)


class GmmEmbedding(nn.Module):

    def __init__(self, gmm_num, emb_dim):
        super(GmmEmbedding, self).__init__()
        self.gmm_embedding = nn.Embedding(gmm_num, emb_dim)
        self.gmm_seq = torch.from_numpy(np.arange(gmm_num)).cuda()
        self.d_k = emb_dim

    def forward(self, x, sent):
        gmm_num = x.size(0)
        batch_size = x.size(1)
        pk_gmm = self.get_gmm_prob(sent)
        gmmn = torch.multinomial(pk_gmm, 1).squeeze()
        gmmn += torch.arange(0, gmm_num*batch_size, gmm_num).to(gmmn)
        x = x.view(gmm_num*batch_size, *x.shape[2:]).index_select(0, gmmn).squeeze()
        return x, pk_gmm

    def get_gmm_prob(self, x):
        scores = torch.matmul(self.gmm_embedding(self.gmm_seq).unsqueeze(0), x.unsqueeze(2)) / math.sqrt(self.d_k)
        return F.softmax(scores.squeeze(), dim=1)


class VariationalDecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, gt_attn, mem_attn, feed_forward, dropout):
        super(VariationalDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.gt_attn = gt_attn
        self.mem_attn = mem_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 5)

    def self_att_layer(self, x, memory, src_mask, tgt_mask, mem_att, tgt_att):
        "Follow Figure 1 (right) for connections."
        m = memory
        tgt_full_mask = tgt_mask.logical_not() + torch.eye(tgt_mask.size(1)).unsqueeze(0).to(tgt_mask)

        mem_att = self.sublayer[3](mem_att, lambda x: self.mem_attn(x, m, m, src_mask))
        tgt_att = self.sublayer[4](tgt_att, lambda x: self.gt_attn(x, x, x, tgt_full_mask))
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return x, mem_att, tgt_att

    def cross_att_layer(self, x, sample_z, memory, src_mask, tgt_mask):
        self_mem = torch.cat([memory, sample_z], 1)
        # self_memo_masks = torch.cat([src_mask, tgt_mask], -1)
        self_mem_masks = torch.cat([src_mask, tgt_mask], -1)

        x = self.sublayer[1](x, lambda x: self.src_attn(x, self_mem, self_mem, self_mem_masks))
        return self.sublayer[2](x, self.feed_forward)

    def get_x_mematt(self, x, memory, src_mask, tgt_mask, mem_att):
        m = memory
        mem_att = self.sublayer[3](mem_att, lambda x: self.mem_attn(x, m, m, src_mask))
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return x, mem_att

    def get_x_gtatt(self, x, tgt_att, x_mask, tgt_mask):
        tgt_full_mask = tgt_mask.logical_not() + torch.eye(tgt_mask.size(1)).unsqueeze(0).to(tgt_mask)
        tgt_att = self.sublayer[4](tgt_att, lambda x: self.gt_attn(x, x, x, tgt_full_mask))
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, x_mask))
        return x, tgt_att


class VariationalDecoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N, gmm_num, gt_mlp, mem_mlp, gmm_embed):
        super(VariationalDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.gmm_num = gmm_num
        self.gt_mlp = gt_mlp
        self.mem_mlp = mem_mlp
        if self.gmm_num > 1:
            self.gmm_emb_gt = gmm_embed
            self.gmm_emb_c = copy.deepcopy(gmm_embed)

    def forward(self, x, memory, src_mask, tgt_mask):
        mem_att = x.clone()
        tgt_att = x.clone()
        for layer in self.layers:
            x, mem_att, tgt_att = layer.self_att_layer(x, memory, src_mask, tgt_mask, mem_att, tgt_att)
        sample_z, ana_kl = self._sample(mem_att, tgt_att, memory)
        src_mask = src_mask.repeat(1, tgt_mask.size(1), 1)
        tgt_mask = tgt_mask.type_as(src_mask)
        for layer in self.layers:
            x = layer.cross_att_layer(x, sample_z, memory, src_mask, tgt_mask)
        return self.norm(x), ana_kl

    def decode(self, x, memory, src_mask, tgt_mask):
        mem_att = x.clone()
        for layer in self.layers:
            x, mem_att = layer.get_x_mematt(x, memory, src_mask, tgt_mask, mem_att)
        sample_z = self._sample_decode(mem_att, memory)
        src_mask = src_mask.repeat(1, tgt_mask.size(1), 1)
        tgt_mask = tgt_mask.type_as(src_mask).repeat(src_mask.size(0), 1, 1)
        for layer in self.layers:
            x = layer.cross_att_layer(x, sample_z, memory, src_mask, tgt_mask)
        return self.norm(x)

    def l_decode(self, x, tgt, memory, src_mask, x_mask, tgt_mask):
        tgt_att = tgt.clone()
        for layer in self.layers:
            x, tgt_att = layer.get_x_gtatt(x, tgt_att, x_mask, tgt_mask)
        sample_z = self._sample_decode(None, memory, tgt_att)
        src_mask = src_mask.repeat(1, sample_z.size(1), 1)
        tgt_mask = tgt_mask[:, :sample_z.size(1)].type_as(src_mask).repeat(src_mask.size(0), 1, 1)
        for layer in self.layers:
            x = layer.cross_att_layer(x, sample_z, memory, src_mask, tgt_mask)
        return self.norm(x)

    def _sample(self, mem_att, tgt_att, memory):
        gt_distribution_params = self.gt_mlp(tgt_att)
        c_distribution_params = self.mem_mlp(mem_att)
        gt_mean, gt_logvar = gt_distribution_params[..., 0], gt_distribution_params[..., 1]
        c_mean, c_logvar = c_distribution_params[..., 0], c_distribution_params[..., 1]
        epsilon = torch.randn(gt_mean.shape).to(gt_mean.device)
        sample_z_gt = epsilon * torch.exp(.5 * gt_logvar) + gt_mean
        if self.gmm_num > 1:
            sample_z_gt, gmmn_gt = self.gmm_emb_gt(sample_z_gt, memory.mean(1).detach())
            gmmn_c = self.gmm_emb_c.get_gmm_prob(memory.mean(1).detach())
            # sample_z_gt = sample_z_gt.index_select(0, torch.multinomial(gmmn_gt.squeeze(), 1)).squeeze()
            # kl_c, analytic_kl = normal_kl_mixtured(gt_mean, gt_logvar, gmmn_gt,
            #                                        c_mean, c_logvar, gmmn_c)
            kl_c, analytic_kl = normal_kl_mixtured(c_mean, c_logvar, gmmn_c.t()[(...,) + (None,) * 2],
                                                   gt_mean, gt_logvar, gmmn_gt.t()[(...,) + (None,) * 2])
            analytic_kl = kl_c + analytic_kl.sum(-1).mean()
        else:
            analytic_kl = normal_kl(gt_mean, gt_logvar, c_mean, c_logvar).sum(-1).mean()
        return sample_z_gt, analytic_kl

    def _sample_decode(self, mem_att, memory, tgt_att=None):
        if tgt_att is not None:
            distribution_params = self.gt_mlp(tgt_att)
        else:
            distribution_params = self.mem_mlp(mem_att)
        mean, logvar = distribution_params[..., 0], distribution_params[..., 1]
        epsilon = torch.randn(mean.shape).to(mean.device)
        sample_z = epsilon * torch.exp(.5 * logvar) + mean
        if self.gmm_num > 1:
            if tgt_att is not None:
                sample_z, _ = self.gmm_emb_gt(sample_z, memory.mean(1).detach())
            else:
                sample_z, _ = self.gmm_emb_c(sample_z, memory.mean(1).detach())
        if sample_z.ndim == 2:
            sample_z = sample_z.unsqueeze(1)
        return sample_z


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
            # Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            # Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N_dec),
            # SrcAtt(SrcAttLayer(d_model, c(attn), c(ff), dropout), N_dec),
            # SimpleDecoder(d_model, d_model * 2 * gmm_num, gmm_num=gmm_num),
            # SimpleDecoder(d_model, d_model * 2 * gmm_num, gmm_num=gmm_num),
            # SelfAtt(SelfAttLayer(d_model, c(attn), dropout), N_dec),
            # SrcAtt(SrcAttLayer(d_model, c(attn), c(ff), dropout), N_dec),
            # GmmEmbedding(gmm_num, d_model),
            VariationalDecoder(VariationalDecoderLayer(d_model, c(attn), c(attn),
                                                       MultiHeadedAttention(h, d_model, 0.5),
                                                       c(attn), c(ff), dropout), N_dec,
                                                       gmm_num,
                                                       SimpleDecoder(d_model, d_model * 2 * gmm_num, gmm_num=gmm_num),
                                                       SimpleDecoder(d_model, d_model * 2 * gmm_num, gmm_num=gmm_num),
                                                       GmmEmbedding(gmm_num, d_model),
                               ),
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
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

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

    def _forward_sample(self, fc_feats, att_feats, seq_f, att_masks=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        seq_length = opt.get('seq_length', self.seq_length)
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)
        if group_size > 1:
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        if seq_f.ndim == 3:  # B * seq_per_img * seq_len
            seq_f = seq_f.reshape(-1, seq_f.shape[2])
        p_att_feats, seq_f, p_att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq_f)
        pp_att_feats = self.model.encode(p_att_feats, p_att_masks)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)

        # p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        trigrams = []  # will be a list of batch_size dictionaries

        # gtz_params = self.model.get_gtz_params(seq, seq_mask)
        seq = fc_feats.new_full((batch_size*sample_n, seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, seq_length, self.vocab_size + 1)
        for t in range(seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_full([batch_size*sample_n], self.bos_idx, dtype=torch.long)

            logprobs, state = self.get_logprobs_state_forward(it, seq_f, seq_mask, pp_att_feats, p_att_masks, state,
                                                              output_logsoftmax=output_logsoftmax)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(seq[:, t - 1].data.cpu().numpy(), self.bad_endings_ix)
                # Make it impossible to generate bad_endings
                tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp

            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:, t - 3:t - 1]
                for i in range(batch_size):  # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current = seq[i][t - 1]
                    if t == 3:  # initialize
                        trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:, t - 2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).to(logprobs.device) # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                # Apply mask to log probs
                # logprobs = logprobs - (mask * 1e9)
                alpha = 2.0  # = 4
                logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx  # This allows eos_idx not being overwritten to 0
                logprobs = logprobs * unfinished.unsqueeze(1).to(logprobs)
                unfinished = unfinished & (it != self.eos_idx)
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seqLogprobs[:, t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def get_logprobs_state_forward(self, it, seq, seq_mask, p_att_feats, att_masks, state, output_logsoftmax=1):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core_forward(xt, seq, seq_mask, p_att_feats, state, att_masks)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state

    def core_forward(self, it, seq, seq_mask, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.l_decode(seq, seq_mask, memory, mask, ys,
                                  subsequent_mask(ys.size(1)).to(memory.device))
        # out = self.model.decode(memory, mask,
        #                         ys,
        #                         subsequent_mask(ys.size(1))
        #                         .to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask,
                                ys,
                                subsequent_mask(ys.size(1))
                                .to(ys.device))
        return out[:, -1], [ys.unsqueeze(0)]
