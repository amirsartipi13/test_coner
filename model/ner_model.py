from typing import List, Any

import pytorch_lightning.core.lightning as pl

import torch
import torch.nn.functional as F
import numpy as np

from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoModel

from log import logger
from utils.metric import SpanF1
from utils.reader_utils import extract_spans


class NERBaseAnnotator(nn.Module):
    def __init__(self,
                 lr=1e-5,
                 dropout_rate=0.1,
                 tag_to_id=None,
                 pad_token_id=1,
                 device='cpu',
                 encoder_model='xlm-roberta-large'):
        super(NERBaseAnnotator, self).__init__()


        self.id_to_tag = {v: k for k, v in tag_to_id.items()}
        self.tag_to_id = tag_to_id
        self.device = device

        self.target_size = len(self.id_to_tag)

        # set the default baseline model here
        self.pad_token_id = pad_token_id

        self.encoder_model = encoder_model
        self.encoder = AutoModel.from_pretrained(encoder_model, return_dict=True)

        self.feedforward = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=self.target_size)
        self.crf_layer = ConditionalRandomField(num_tags=self.target_size, constraints=allowed_transitions(constraint_type="BIO", labels=self.id_to_tag))

        self.lr = lr
        self.dropout = nn.Dropout(dropout_rate)
        self.span_f1 = SpanF1()

    def forward(self, batch):
        tokens, tags, token_mask, metadata = batch
        
        tokens = tokens.to(self.device)
        tags = tags.to(self.device)
        token_mask = token_mask.to(self.device)
        # metadata = metadata.to(self.device)

        batch_size = tokens.size(0)

        embedded_text_input = self.encoder(input_ids=tokens, attention_mask=token_mask)
        embedded_text_input = embedded_text_input.last_hidden_state
        embedded_text_input = self.dropout(F.leaky_relu(embedded_text_input))

        # project the token representation for classification
        token_scores = self.feedforward(embedded_text_input)

        # compute the log-likelihood loss and compute the best NER annotation sequence
        output = self._compute_token_tags(token_scores=token_scores, tags=tags, token_mask=token_mask, metadata=metadata, batch_size=batch_size)
        return output

    def _compute_token_tags(self, token_scores, tags, token_mask, metadata, batch_size):
        # compute the log-likelihood loss and compute the best NER annotation sequence
        loss = -self.crf_layer(token_scores, tags, token_mask) / float(batch_size)
        best_path = self.crf_layer.viterbi_tags(token_scores, token_mask)

        pred_results = []
        for i in range(batch_size):
            tag_seq, _ = best_path[i]
            pred_results.append(extract_spans([self.id_to_tag[x] for x in tag_seq if x in self.id_to_tag]))

        self.span_f1(pred_results, metadata)
        output = {"loss": loss, "results": self.span_f1.get_metric()}
        return output
