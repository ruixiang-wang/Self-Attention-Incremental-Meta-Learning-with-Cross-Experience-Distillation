import torch
import torch.nn.functional as F

def cross_generalization_distillation(model, model_old, exp_data_query, exp_proto, cls_loss, args, μ, φ):
    prev_exp_curr_logits = model.decode(exp_proto, exp_data_query)
    prev_exp_curr_logits = F.softmax(prev_exp_curr_logits)
    prev_exp_old_logits = model_old.decode(exp_proto, exp_data_query)
    prev_exp_old_logits = F.softmax(prev_exp_old_logits)
    kl_div_prev_exp = (prev_exp_old_logits.clamp(min=1e-4) *
                      (prev_exp_old_logits.clamp(min=1e-4) /
                       prev_exp_curr_logits.clamp(min=1e-4)).log()).sum() / len(prev_exp_old_logits)
    curr_exp_new_logits = model.decode(exp_proto, exp_data_query)
    curr_exp_new_logits = F.softmax(curr_exp_new_logits)
    curr_exp_old_logits = model_old.decode(exp_proto, exp_data_query)
    curr_exp_old_logits = F.softmax(curr_exp_old_logits)
    kl_div_curr_exp = (curr_exp_old_logits.clamp(min=1e-4) *
                      (curr_exp_old_logits.clamp(min=1e-4) /
                       curr_exp_new_logits.clamp(min=1e-4)).log()).sum() / len(curr_exp_old_logits)
    loss = cls_loss + kl_div_prev_exp * μ + kl_div_curr_exp * φ
    return loss
