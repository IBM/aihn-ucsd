import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BertForDistantRE(BertPreTrainedModel):

    def __init__(self, config, num_labels, file_config, dropout=0.1, bag_attn=False, rel_embedding='A', device='cpu'):
        super(BertForDistantRE, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.We = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()
        self.classifier = nn.Linear(3 * config.hidden_size, num_labels)
        self.bag_attn = bag_attn
        self.rel_emb = file_config.rel_embedding
        self.on_device = device
        if bag_attn:
            self.Wo = nn.Linear(3 * config.hidden_size, 3 * config.hidden_size)
        self.softmax = nn.Softmax(-1)
        self.init_weights()

    def bag_attention_logits(self, bag, labels=None, is_train=True):
        if is_train:
            query = labels.unsqueeze(1)  # B x 1
            attn_M = self.classifier.weight.data[query]  # B x 1 x H
            attn_s = (bag * attn_M).sum(-1)  # (B x G x H) * (B x 1 x H) -> B x G x H -> B x G
            attn_s = self.softmax(attn_s)  # B x G
            bag = (attn_s.unsqueeze(-1) * bag).sum(1)  # (B x G x 1) * (B x G x H) -> B x G x H -> B x H
            logits = self.classifier(self.dropout(self.act(bag)))  # B x C
        else:
            attn_M = bag.matmul(self.classifier.weight.data.transpose(0, 1))  # (B x G x H) * (H x C) -> B x G x C
            attn_s = self.softmax(attn_M.transpose(-1, -2))  # B x C x G
            bag = attn_s.bmm(bag)  # (B x C x G) * (B x G x H) -> B x C x H
            logits = self.classifier(self.dropout(self.act(bag)))  # B x C x C
            logits = logits.diagonal(dim1=1, dim2=2)  # B x C
        return logits

    def create_e_start_mask(self, entity_ids, b, g, l):
        idx_e2_start = torch.argmax(entity_ids, dim=2)  # index of first '2' (start of tail ent)
        entity_ids_no_e2 = entity_ids.detach().clone()
        entity_ids_no_e2[entity_ids_no_e2 == 2] = 0
        idx_e1_start = torch.argmax(entity_ids_no_e2, dim=2)  # index of first '2' (start of tail ent)
        e1_s_mask = torch.zeros((b * g, l)).scatter_(1, idx_e1_start.view(-1, 1), 1)
        e2_s_mask = torch.zeros((b * g, l)).scatter_(1, idx_e2_start.view(-1, 1), 1)
        return e1_s_mask.resize(b, g, l).to(self.device), e2_s_mask.resize(b, g, l).to(self.device)

    def create_e_end_mask(self, entity_ids, b, g, l):
        # E1 end
        ent_ids_copy = entity_ids.detach().clone()
        ent_ids_copy[ent_ids_copy == 2] = 0  # zero out e2
        idx_e1_start = torch.argmax(ent_ids_copy, dim=2)  # index of first '1' (start of tail ent)
        e1_len = torch.count_nonzero(ent_ids_copy, dim=2) - 1
        idx_e1_end = idx_e1_start + e1_len
        idx_e1_end.to(self.device)

        # E2 end:
        ent_ids_copy = entity_ids.detach().clone()
        ent_ids_copy[ent_ids_copy == 1] = 0  # zero out e1
        idx_e2_start = torch.argmax(ent_ids_copy, dim=2)  # index of first '2' (start of tail ent)
        e2_len = torch.count_nonzero(ent_ids_copy, dim=2) - 1
        idx_e2_end = idx_e2_start + e2_len
        idx_e2_end.to(self.device)

        # Create mask
        e1_e_mask = torch.zeros((b * g, l)).to(self.device).scatter_(1, idx_e1_end.view(-1, 1), 1).resize(b, g, l)
        e2_e_mask = torch.zeros((b * g, l)).to(self.device).scatter_(1, idx_e2_end.view(-1, 1), 1).resize(b, g, l)

        return e1_e_mask.to(self.device), e2_e_mask.to(self.device)

    def create_e_mid_mask(self, entity_ids):
        # Dims
        b, g, l = entity_ids.shape
        entity_ids = entity_ids.resize(b * g, l)

        # E1 start:
        ent_ids_copy = entity_ids.detach().clone()
        ent_ids_copy[ent_ids_copy == 2] = 0  # zero out e2
        e1_s = torch.argmax(ent_ids_copy, dim=1)  # index of first '1' (start of tail ent)

        # E2 start:
        ent_ids_copy = entity_ids.detach().clone()
        ent_ids_copy[ent_ids_copy == 1] = 0  # zero out e1
        e2_s = torch.argmax(ent_ids_copy, dim=1)  # index of first '2' (start of tail ent)

        # Concat E1 and E2 start
        e_starts = torch.stack((e1_s, e2_s), dim=1)
        e_starts_max = torch.max(e_starts, dim=1)

        # Inverse mask
        e_mid_mask = entity_ids.detach().clone()
        e_mid_mask[e_mid_mask == 1] = 2
        e_mid_mask[e_mid_mask == 0] = 1
        e_mid_mask[e_mid_mask == 2] = 0

        # Index of first zero in inverse mask
        argmin = torch.argmin(e_mid_mask, dim=1)

        for i, idx in enumerate(argmin):
            e_mid_mask[i, 0:idx] = 0  # zero everything before first zero
            e_mid_mask[i, e_starts_max.values[i]:] = 0  # everything after start of 2nd entity

        e_mid_mask = e_mid_mask.resize(b, g, l)
        return e_mid_mask.to(self.device)

    def forward(self,
                input_ids,
                entity_ids=None,
                attention_mask=None,
                labels=None,
                is_train=True):
        '''PART-I: Encode the sequence with BERT'''
        B, G, L = input_ids.shape  # batch size (2), group/bag (16), length (128)
        input_ids = input_ids.view(B * G, -1)
        attention_mask = attention_mask.view(B * G, -1)

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = outputs[0], outputs[1]  # seq output: 2 x 16 x 128 x 768, pooled: 2 x 16 x 768

        sequence_output = sequence_output.view(B, G, L, -1).clone()  # B x G x L x H
        cls = pooled_output.view(B, G, -1).clone()  # B x G x H

        '''PART-II: Get hidden representations'''
        if self.rel_emb in ['A', 'B']:
            # E1 entity mention pool
            e1_mask = (entity_ids == 1).float()  # locations of e1 entity
            e1 = sequence_output * e1_mask.unsqueeze(-1)  # B x G x L x H
            e1 = e1.sum(2) / e1_mask.sum(2).unsqueeze(-1)  # Empty sequences will have all zeros
            e1 = self.We(self.dropout(self.act(e1)))  # B x G x H

            # E2 entity mention pool
            e2_mask = (entity_ids == 2).float()
            e2 = sequence_output * e2_mask.unsqueeze(-1)
            e2 = e2.sum(2) / e2_mask.sum(2).unsqueeze(-1)
            e2 = self.We(self.dropout(self.act(e2)))  # B x G x H

        if self.rel_emb in ['D', 'E', 'H', 'I', 'N', 'O']:
            # E1 start, E2 start:
            e1_s_mask, e2_s_mask = self.create_e_start_mask(entity_ids)
            e1_s = sequence_output * e1_s_mask.unsqueeze(-1)
            e1_s = e1_s.sum(2) / e1_s_mask.sum(2).unsqueeze(-1)
            e1_s = self.We(self.dropout(self.act(e1_s)))

            e2_s = sequence_output * e2_s_mask.unsqueeze(-1)
            e2_s = e2_s.sum(2) / e2_s_mask.sum(2).unsqueeze(-1)
            e2_s = self.We(self.dropout(self.act(e2_s)))

        if self.rel_emb in ['F', 'G', 'H', 'I', 'L', 'M', 'N', 'O']:
            # E1 end, E2 end:
            e1_e_mask, e2_e_mask = self.create_e_end_mask(entity_ids, B, G, L)
            e1_e = sequence_output * e1_e_mask.unsqueeze(-1)
            e1_e = e1_e.sum(2) / e1_e_mask.sum(2).unsqueeze(-1)
            e1_e = self.We(self.dropout(self.act(e1_e)))

            e2_e = sequence_output * e2_e_mask.unsqueeze(-1)
            e2_e = e2_e.sum(2) / e2_e_mask.sum(2).unsqueeze(-1)
            e2_e = self.We(self.dropout(self.act(e2_e)))

        if self.rel_emb in ['J', 'K', 'L', 'M', 'N', 'O']:
            # E middle:
            create_e_mid_mask = self.create_e_mid_mask(entity_ids)
            e_mid = sequence_output * create_e_mid_mask.unsqueeze(-1)
            e_mid = e_mid.sum(2) / create_e_mid_mask.sum(2).unsqueeze(-1)
            e_mid = self.We(self.dropout(self.act(e_mid)))

        if self.rel_emb in ['P', 'Q']:
            # Avg sequence embedding
            sequence_avg = sequence_output.sum(2) / L # B x G x H

        '''PART-III: Relation Embedding Variations'''
        rel_emb_dict = {
            'A': cls, # [CLS]
            'B': torch.cat((e1, e2), -1), # mention pool
            'C': torch.cat((cls, e1, e2), -1),  # CLS] + mention pool (original UMLS) B x G x 3H
            'D': torch.cat((e1_s, e2_s), -1), # entity start
            'E': torch.cat((cls, e1_s, e2_s), -1), # [CLS] + entity start
            'F': torch.cat((e1_e, e2_e), -1), # entity end
            'G': torch.cat((cls, e1_e, e2_e), -1), # [CLS] + entity end
            'H': torch.cat((e1_s, e1_e, e2_s, e2_e), -1), # entity start + entity end
            'I': torch.cat((cls, e1_s, e1_e, e2_s, e2_e), -1), # [CLS] + entity start + entity end
            'J': torch.cat((e_mid), -1), # middle
            'K': torch.cat((cls, e_mid), -1), # [CLS] + middle
            'L': torch.cat((e1_e, e_mid, e2_e), -1), # middle + entity ends
            'M': torch.cat((cls, e1_e, e_mid, e2_e), -1), # [CLS] + middle + entity ends
            'N': torch.cat((e1_s, e1_e, e_mid, e2_s, e2_e), -1), #  entity start + middle + enity end
            'O': torch.cat((cls, e1_s, e1_e, e_mid, e2_s, e2_e), -1), # cls + entity start + middle + enity end
            'P': sequence_avg, # avg of entire sequence
            'Q': torch.cat((cls, sequence_avg), -1), # avg of entire sequence
        }

        # Set relationship embedding type
        r_h = rel_emb_dict.get(self.rel_emb, None)

        '''PART-IV: Average bag aggregation and relation classifier'''
        if self.bag_attn:
            r_h = self.Wo(self.dropout(self.act(r_h)))
            logits = self.bag_attention_logits(r_h, labels, is_train)
        else:
            r_h = r_h.sum(1) / G  # B x 3H
            logits = self.classifier(self.dropout(self.act(r_h)))
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
