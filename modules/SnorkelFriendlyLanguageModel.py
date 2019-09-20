import torch
import torch.nn as nn
from pytorch_transformers.modeling_auto import AutoModel

# Create special module that can be passed into the MTL model which splits up the data so that it can be used in language_model
# I.e., this module splits up the inputted data into tokens, token type ids and attention masks
class SnorkelFriendlyLanguageModel(nn.Module):

    def __init__(self, language_model=AutoModel.from_pretrained('bert-base-uncased')):
        super(SnorkelFriendlyLanguageModel, self).__init__()
        self.language_model_layer = language_model
        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')

    def forward(self, x):
        x_tensor = torch.tensor(x, dtype=torch.long, device=self.device)

        # The input that is created for language models is split up into input ids (ids of each token),
        # token type id (only used by certain models to delineate some text from other E.g. question from context paragraph)
        # and attention mask (to make sure that padding is not attended to).
        input_ids = x_tensor[:,0,:]
        token_type_ids = x_tensor[:,1,:]
        attention_masks = x_tensor[:,2,:]

        # Some models (distilBERT and transformerXL) do not take token_type_ids
        if self.language_model_layer.base_model_prefix in ["distilbert", "transformer"]:
            language_model_output = self.language_model_layer.forward(input_ids = input_ids, attention_mask=attention_masks)
        else:
            language_model_output = self.language_model_layer.forward(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask=attention_masks)

        return language_model_output
