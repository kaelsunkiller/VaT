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

    def __init__(self, att_feats_encoder, gt_encoder, decoder, ge_mlp, ce_mlp, ce_self, ce_src, tgt_embed, generator,
                 gmm_embed=None):
        super(VAE, self).__init__()
        self.src_encoder = att_feats_encoder
        self.gt_encoder = gt_encoder
        self.decoder = decoder
        self.ge_mlp = ge_mlp
        self.ce_mlp = ce_mlp
        self.ce_self = ce_self
        self.ce_src = ce_src
        self.tgt_embed = tgt_embed
        self.generator = generator

        self.gmm_num = ge_mlp.gmm_num
        if self.gmm_num > 1:
            if gmm_embed is not None:
                self.gmm_emb_gt = gmm_embed
                self.gmm_emb_c = copy.deepcopy(gmm_embed)
            else:
                self.gmmn_gt = nn.Parameter(torch.ones(self.gmm_num, 1, 1, 1).float(), requires_grad=True)
                self.gmmn_c = nn.Parameter(torch.ones(self.gmm_num, 1, 1, 1).float(), requires_grad=True)

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, src_mask)
        tgt_full_mask = torch.ones_like(tgt_mask)

        gt_distribution_params = self.ge_mlp(self.gt_encoder(self.tgt_embed(tgt), tgt_full_mask))
        self_att = self.ce_self(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        src_att = self.ce_src(self_att, memory, src_mask, tgt_mask)
        c_distribution_params = self.ce_mlp(src_att)
        gt_mean, gt_logvar = gt_distribution_params[..., 0], gt_distribution_params[..., 1]
        c_mean, c_logvar = c_distribution_params[..., 0], c_distribution_params[..., 1]
        epsilon = torch.randn(gt_mean.shape).to(gt_mean.device)
        sample_z_gt = epsilon * torch.exp(.5 * gt_logvar) + gt_mean
        if self.gmm_num > 1:
            if hasattr(self, 'gmm_emb_gt'):
                sample_z_gt, gmmn_gt = self.gmm_emb_gt(sample_z_gt, memory.mean(1))
                gmmn_c = self.gmm_emb_c.get_gmm_prob(memory.mean(1))
                gmmn_gt, gmmn_c = [x.t()[(...,) + (None,) * 2] for x in [gmmn_gt, gmmn_c]]
            else:
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

        decoder_memo = torch.cat([memory, sample_z_gt], 1)
        decoder_memo_masks = torch.cat([src_mask.repeat(1, tgt_mask.size(1), 1), tgt_mask.type_as(src_mask)], -1)
        # decoder_memo = sample_z_gt
        # decoder_memo_masks = tgt_mask.type_as(src_mask)
        out = self.decoder(self_att, decoder_memo, decoder_memo_masks, tgt_mask)
        return out, analytic_kl

    def get_gtz_params(self, tgt, tgt_mask):
        tgt_full_mask = torch.ones_like(tgt_mask)

        gt_distribution_params = self.ge_mlp(self.gt_encoder(self.tgt_embed(tgt), tgt_full_mask))
        return gt_distribution_params

    def encode(self, src, src_mask):
        return self.src_encoder(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        self_att = self.ce_self(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        src_att = self.ce_src(self_att, memory, src_mask, tgt_mask)
        distribution_params = self.ce_mlp(src_att)
        # distribution_params = self.ce_mlp(self.ce_gen(self.tgt_embed(tgt), memory, src_mask, tgt_mask))
        mean, logvar = distribution_params[..., 0], distribution_params[..., 1]
        epsilon = torch.randn(mean.shape).to(mean.device)
        sample_z = epsilon * torch.exp(.5 * logvar) + mean
        if self.gmm_num > 1:
            if hasattr(self, 'gmm_emb_gt'):
                if tgt is not None:
                    sample_z, _ = self.gmm_emb_gt(sample_z, memory.mean(1))
                else:
                    sample_z, _ = self.gmm_emb_c(sample_z, memory.mean(1))
            else:
                gmmn = self.gmmn_gt if tgt is not None else self.gmmn_c
                gmmn = torch.softmax(gmmn, dim=0)
                sample_z = sample_z.index_select(0, torch.multinomial(gmmn.squeeze(), 1)).squeeze()
        if sample_z.ndim == 2:
            sample_z = sample_z.unsqueeze(1)

        decoder_memo = torch.cat([memory, sample_z], 1)
        decoder_memo_masks = torch.cat([src_mask.repeat(1, tgt_mask.size(1), 1), tgt_mask.type_as(src_mask).repeat(src_mask.size(0), 1, 1)], -1)
        # decoder_memo = sample_z
        # decoder_memo_masks = tgt_mask.type_as(src_mask).repeat(src_mask.size(0), 1, 1)
        return self.decoder(self_att, decoder_memo, decoder_memo_masks, tgt_mask)

    def l_decode(self, distribution_params, memory, src_mask, tgt, tgt_mask):
        self_att = self.ce_self(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        mean, logvar = distribution_params[..., 0], distribution_params[..., 1]
        epsilon = torch.randn(mean.shape).to(mean.device)
        sample_z = epsilon * torch.exp(.5 * logvar) + mean
        if self.gmm_num > 1:
            if hasattr(self, 'gmm_emb_gt'):
                if tgt is not None:
                    sample_z, _ = self.gmm_emb_gt(sample_z, memory.mean(1))
                else:
                    sample_z, _ = self.gmm_emb_c(sample_z, memory.mean(1))
            else:
                gmmn = self.gmmn_gt if tgt is not None else self.gmmn_c
                gmmn = torch.softmax(gmmn, dim=0)
                sample_z = sample_z.index_select(0, torch.multinomial(gmmn.squeeze(), 1)).squeeze()
        if sample_z.ndim == 2:
            sample_z = sample_z.unsqueeze(1)
        sample_z = sample_z[:, :tgt.shape[1], :]

        decoder_memo = torch.cat([memory, sample_z], 1)
        decoder_memo_masks = torch.cat([src_mask.repeat(1, sample_z.size(1), 1), tgt_mask[:, :sample_z.size(1)].type_as(src_mask).repeat(src_mask.size(0), 1, 1)], -1)
        # decoder_memo = sample_z
        # decoder_memo_masks = tgt_mask[:, :sample_z.size(1)].type_as(src_mask).repeat(src_mask.size(0), 1, 1)
        return self.decoder(self_att, decoder_memo, decoder_memo_masks, tgt_mask)


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


class SrcAtt(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(SrcAtt, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class SrcAttLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, src_attn, feed_forward, dropout):
        super(SrcAttLayer, self).__init__()
        self.size = size
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.src_attn(x, m, m, src_mask))
        # x = self.sublayer[0](x, lambda x: self.src_attn(m))
        return self.sublayer[1](x, self.feed_forward)


class SelfAtt(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(SelfAtt, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class SelfAttLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, dropout):
        super(SelfAttLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        # self.src_attn = src_attn
        # self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        # m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # if self.feed_forward is not None:
        #     x = self.sublayer[1](x, self.feed_forward)
        return x


class VAETModel(TransformerModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1, gmm_num=1, gmm_select=False):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = VAE(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            # Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N_dec),
            SrcAtt(SrcAttLayer(d_model, c(attn), c(ff), dropout), N_dec),
            SimpleDecoder(d_model, d_model * 2 * gmm_num, gmm_num=gmm_num),
            SimpleDecoder(d_model, d_model * 2 * gmm_num, gmm_num=gmm_num),
            SelfAtt(SelfAttLayer(d_model, c(attn), dropout), N_dec),
            SrcAtt(SrcAttLayer(d_model, c(attn), c(ff), dropout), N_dec),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab),
            gmm_embed=GmmEmbedding(gmm_num, d_model) if gmm_select else None
        )

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
                                     gmm_num=self.gmm_num,
                                     gmm_select=getattr(opt, 'gmm_select', None))

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

    def _forward_sample(self, fc_feats, att_feats, seq, att_masks=None, opt={}):

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

        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        p_att_feats, seq, p_att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        pp_att_feats = self.model.encode(p_att_feats, p_att_masks)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)

        # p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        trigrams = []  # will be a list of batch_size dictionaries

        gtz_params = self.model.get_gtz_params(seq, seq_mask)
        seq = fc_feats.new_full((batch_size*sample_n, seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, seq_length, self.vocab_size + 1)
        for t in range(seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_full([batch_size*sample_n], self.bos_idx, dtype=torch.long)

            logprobs, state = self.get_logprobs_state_forward(it, gtz_params, pp_att_feats, p_att_masks, state,
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

    def get_logprobs_state_forward(self, it, gt_z_params, p_att_feats, att_masks, state, output_logsoftmax=1):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core_forward(xt, gt_z_params, p_att_feats, state, att_masks)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state

    def core_forward(self, it, gt_z_params, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.l_decode(gt_z_params, memory, mask, ys,
                                  subsequent_mask(ys.size(1)).to(mask.device))
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
                                .to(mask.device))
        return out[:, -1], [ys.unsqueeze(0)]


    def _sample_logprobs(self,fc_feats, gt2seqLogprobs, opt={}):
        sample_method = opt.get('sample_method', 'greedy')
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)

        seq_length = gt2seqLogprobs.size(1)
        batch_size = fc_feats.size(0)
        seq = fc_feats.new_full((batch_size * sample_n, seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, seq_length, self.vocab_size + 1)
        for t in range(seq_length):
            if t == 0:  # input <bos>
                it = fc_feats.new_full([batch_size*sample_n], self.bos_idx, dtype=torch.long)

            logprobs = gt2seqLogprobs[:, t]

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
