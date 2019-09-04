import torch
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertModel

# Create special module that can be passed into the MTL model which splits up the data so that it can be used in BERT
# I.e., this module splits up the inputted data into tokens, token type ids and attention masks
class SnorkelFriendlyBert(nn.Module):

    def __init__(self, bert_model=BertModel.from_pretrained('bert-base-uncased')):
        super(SnorkelFriendlyBert, self).__init__()
        self.bert_layer = bert_model
        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')

    def forward(self, x):
        x_tensor = torch.tensor(x, dtype=torch.long, device=self.device)

        input_ids = x_tensor[:,0,:]
        token_type_ids = x_tensor[:,1,:]
        attention_masks = x_tensor[:,2,:]

        # NB: No need to input position IDs, as they are generated automatically as long as the input is inputted consistent with
        # the original data. I.e. the first token in the input is the first word of the inputted text.

        bert_output = self.bert_layer.forward(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask=attention_masks)
        
        return bert_output