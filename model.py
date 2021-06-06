import torch
from transformers import BertConfig, BertModel


class BertForMultiLabelClassification(BertModel):
    def __init__(self, config=BertConfig(), num_labels=77):
        super(BertForMultiLabelClassification, self).__init__(config)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        pooler_output = self.bert(input_ids, attention_mask, token_type_ids).pooler_output
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)

        if labels is not None:
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(logits, labels)
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True
