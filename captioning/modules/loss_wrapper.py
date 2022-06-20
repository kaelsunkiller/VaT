import torch
from . import losses
from ..utils.rewards import init_scorer, get_self_critical_reward

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = losses.LanguageModelCriterion(opt)
        self.rl_crit = losses.RewardCriterion()
        self.struc_crit = losses.StructureLosses(opt)

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, struc_flag, slp_flag):
        opt = self.opt
        
        out = {}
        if struc_flag:
            if opt.structure_loss_weight < 1:
                model_outs = self.model(fc_feats, att_feats, labels[..., :-1], att_masks)
                # for transform vae
                if isinstance(model_outs, tuple):
                    model_out, analytic_kl = model_outs
                    out['analytic_kl'] = analytic_kl
                    # out['pa_loss'] = self.crit(pa_out, labels[..., 1:], masks[..., 1:])
                else:
                    model_out = model_outs
                lm_loss = self.crit(model_out, labels[..., 1:], masks[..., 1:])
            else:
                lm_loss = torch.tensor(0).type_as(fc_feats)
            if opt.structure_loss_weight > 0:
                # sample logprobs giving groundtruth as inputs
                if slp_flag:
                    model_outs = self.model(fc_feats, att_feats, labels[..., :-1], att_masks)
                    # for transform vae
                    if isinstance(model_outs, tuple):
                        model_out, analytic_kl = model_outs
                        out['analytic_kl'] = analytic_kl
                    method = 'sample_logprobs'
                    args = (fc_feats, model_out)
                    seqlength = model_out.size(1)
                # for transform vae
                elif 'VAE' in type(self.model).__name__:
                    _, analytic_kl = self.model(fc_feats, att_feats, labels[..., :-1], att_masks)
                    out['analytic_kl'] = analytic_kl
                    # out['pa_loss'] = self.crit(pa_out, labels[..., 1:], masks[..., 1:])
                    # method = 'forward_sample'
                    # args = (fc_feats, att_feats, labels, att_masks)
                    method = 'sample'
                    args = (fc_feats, att_feats, att_masks)
                    seqlength = opt.max_length
                else:
                    method = 'sample'
                    args = (fc_feats, att_feats, att_masks)
                    seqlength = opt.max_length
                gen_result, sample_logprobs = self.model(*args,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                            or not 'margin' in opt.structure_loss_type,
                        'sample_n': opt.train_sample_n,
                        'seq_length': seqlength},
                    mode=method)
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts)
            else:
                struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                              'reward': torch.tensor(0).type_as(fc_feats)}
            loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']
        elif not sc_flag:
            model_outs = self.model(fc_feats, att_feats, labels[..., :-1], att_masks)
            # for transform vae
            if isinstance(model_outs, tuple):
                model_out, analytic_kl = model_outs
                out['analytic_kl'] = analytic_kl
                # out['pa_loss'] = self.crit(pa_out, labels[..., 1:], masks[..., 1:])
            else:
                model_out = model_outs
            loss = self.crit(model_out, labels[..., 1:], masks[..., 1:])
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks,
                    mode='sample',
                    opt={'sample_method': opt.sc_sample_method,
                         'beam_size': opt.sc_beam_size})
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).to(sample_logprobs)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out
