import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Divergence(nn.Module):
    """
    Jensen-Shannon divergence, used to measure ranking consistency between similarity lists obtained from examples with two different dropout masks
    """
    def __init__(self, beta_):
        super(Divergence, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.eps = 1e-7
        self.beta_ = beta_

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log().clamp(min=self.eps)
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))

class ListNet(nn.Module):
    """
    ListNet objective for ranking distillation; minimizes the cross entropy between permutation [top-1] probability distribution and ground truth obtained from teacher
    """
    def __init__(self, tau, gamma_):
        super(ListNet, self).__init__()
        self.teacher_temp_scaled_sim = Similarity(tau / 2)
        self.student_temp_scaled_sim = Similarity(tau)
        self.gamma_ = gamma_

    def forward(self, teacher_top1_sim_pred, student_top1_sim_pred):
        p = F.log_softmax(student_top1_sim_pred.fill_diagonal_(float('-inf')), dim=-1)
        q = F.softmax(teacher_top1_sim_pred.fill_diagonal_(float('-inf')), dim=-1)
        loss = -(q*p).nansum()  / q.nansum()
        return self.gamma_ * loss 

class ListMLE(nn.Module):
    """
    ListMLE objective for ranking distillation; maximizes the liklihood of the ground truth permutation (sorted indices of the ranking lists obtained from teacher) 
    """
    def __init__(self, tau, gamma_):
        super(ListMLE, self).__init__()
        self.temp_scaled_sim = Similarity(tau)
        self.gamma_ = gamma_ 
        self.eps = 1e-7

    def forward(self, teacher_top1_sim_pred, student_top1_sim_pred):

        y_pred = student_top1_sim_pred 
        y_true = teacher_top1_sim_pred

        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
        mask = y_true_sorted == -1
        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float('-inf')
        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
        observation_loss = torch.log(cumsums + self.eps) - preds_sorted_by_true_minus_max
        observation_loss[mask] = 0.0

        return self.gamma_ * torch.mean(torch.sum(observation_loss, dim=1))

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    x_rank = x.argsort(dim=1)
    ranks = torch.zeros_like(x_rank, dtype=torch.float)
    n, d = x_rank.size()

    for i in range(n):
        ranks[i][x_rank[i]] = torch.arange(d, dtype=torch.float).to(ranks.device)
    return ranks


def cal_spr_corr(x: torch.Tensor, y: torch.Tensor):
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    x_rank_mean = torch.mean(x_rank, dim=1).unsqueeze(1)
    y_rank_mean = torch.mean(y_rank, dim=1).unsqueeze(1)
    xn = x_rank - x_rank_mean
    yn = y_rank - y_rank_mean
    x_var = torch.sqrt(torch.sum(torch.square(xn), dim=1).unsqueeze(1))
    y_var = torch.sqrt(torch.sum(torch.square(yn), dim=1).unsqueeze(1))
    xn = xn / x_var
    yn = yn / y_var

    return torch.mm(xn, torch.transpose(yn, 0, 1))

def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.div = Divergence(beta_=cls.model_args.beta_)
    if cls.model_args.distillation_loss == "listnet":
        cls.distillation_loss_fct = ListNet(cls.model_args.tau2, cls.model_args.gamma_)
    elif cls.model_args.distillation_loss == "listmle":
        cls.distillation_loss_fct = ListMLE(cls.model_args.tau2, cls.model_args.gamma_)
    else:
        raise NotImplementedError
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
    teacher_top1_sim_pred=None,
    distances1=None,
    distances2=None,
    baseE_vecs1=None,
    baseE_vecs2=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    mlm_outputs = None
    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Obtain sentence embeddings from [MASK] token
    index = input_ids == cls.mask_token_id
    pooler_output = outputs.last_hidden_state[index]
    cls.last_hidden_state = pooler_output

    assert pooler_output.size() == torch.Size([batch_size * num_sent, pooler_output.size(-1)])

    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))
    # Separate representation
    z1, z2 = pooler_output[:,0], pooler_output[:,1]

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()
    loss_o = loss_fct(cos_sim, labels)

    # Calculate BML loss
    if num_sent == 3:
        z3 = pooler_output[:, 2]  # Embeddings of soft negative samples
        temp1 = torch.cosine_similarity(z1, z2, dim=1)  # Cosine similarity of positive pairs
        temp2 = torch.cosine_similarity(z1, z3, dim=1)  # Cosine similarity of soft negative pairs
        temp = temp2 - temp1  # similarity difference
        loss1 = torch.relu(temp + cls.alpha) + torch.relu(-temp - cls.beta)  # BML loss
        loss1 = torch.mean(loss1)
        loss_o += loss1 * cls.lambda_

    cos_sim_baseE = None
    if cls.model_args.simf == "Spearmanr":
        cos_sim_baseE = cal_spr_corr( \
            distances1, distances1 \
            )
    else:
        raise NotImplementedError

    cos_sim_baseE = cos_sim_baseE.to(cos_sim.device)
    loss_fct_baseE = nn.MSELoss(reduction="none")
    cos_sim_baseE_bound = torch.logical_and( \
        cos_sim_baseE <= cls.model_args.baseE_sim_thresh_upp, \
        cos_sim_baseE >= cls.model_args.baseE_sim_thresh_low \
        ).type(torch.float).to(cos_sim.device)
    mse = loss_fct_baseE(cos_sim * cls.model_args.temp, cos_sim_baseE)
    loss_baseE = torch.sum(mse * cos_sim_baseE_bound) / (torch.sum(cos_sim_baseE_bound) + 1e-8)



    # RankCSE - knowledge distillation loss 
    student_top1_sim_pred = cos_sim.clone()
    kd_loss = cls.distillation_loss_fct(teacher_top1_sim_pred.to(cls.device), student_top1_sim_pred)

    # RankCSE - self-distillation loss
    z1_z2_cos = cos_sim.clone()
    z2_z1_cos = cls.sim(z2.unsqueeze(1), z1.unsqueeze(0))
    sd_loss = cls.div(z1_z2_cos.softmax(dim=-1).clamp(min=1e-7), z2_z1_cos.softmax(dim=-1).clamp(min=1e-7))

    # L = L_infoNCE + L_consistency + L_distillation
    #loss = loss + sd_loss + kd_loss

    if cls.model_args.loss_type == "hinge":
        loss = torch.max(loss_o, cls.model_args.baseE_lmb * loss_baseE, sd_loss)
    elif cls.model_args.loss_type == "weighted_sum":
        loss = loss_o + cls.model_args.baseE_lmb * loss_baseE + cls.model_args.t_lmb * kd_loss
    else:
        raise NotImplementedError

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, mask_token_id=None, alpha=None, beta=None, lambda_=None, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)
        self.mask_token_id = mask_token_id
        self.last_hidden_state = None

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        cl_init(self, config)

        self.alpha = alpha  # 0.1
        self.beta = beta  # 0.3
        self.lambda_ = lambda_  # 1e-3

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        teacher_top1_sim_pred=None,
# spearmanr
        distances1=None,
        distances2=None,
        baseE_vecs1=None,
        baseE_vecs2=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                teacher_top1_sim_pred=teacher_top1_sim_pred,
                # spearmanr
                distances1=distances1,
                distances2=distances2,
                baseE_vecs1=baseE_vecs1,
                baseE_vecs2=baseE_vecs2,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, mask_token_id, alpha=None, beta=None, lambda_=None, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.mask_token_id = mask_token_id

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        cl_init(self, config)

        self.alpha = alpha  # 0.1
        self.beta = beta  # 0.3
        self.lambda_ = lambda_  # 5e-4

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
        teacher_top1_sim_pred=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
                teacher_top1_sim_pred=teacher_top1_sim_pred,
            )
