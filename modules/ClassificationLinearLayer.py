import torch
import torch.nn as nn

# Module that takes only the classification layer of BERT output for use in sequence level classifier.
class ClassificationLinearLayer(nn.Module):

    def __init__(self,input_size, num_labels):

        super(ClassificationLinearLayer, self).__init__()
        self.linear_classifier = nn.Linear(input_size, num_labels)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')

    def forward(self, bert_output):
        # HuggingFace's implementation of BERT outputs a tuple with both the final layer output at every token size,
        # as well as just the first token position final output (where the [CLS] token is).
        #
        # The documentation can be found here: https://github.com/huggingface/pytorch-transformers/blob/b0b9b8091b73f929306704bd8cd62b712621cebc/pytorch_transformers/modeling_bert.py#L628

        # first_token_final_hidden_state = bert_output[1]
        # averaged_all_token_final_hidden_state_averaged = bert_output[0].mean(dim=1)
        # Take the hidden layer at 1st token position of BERT output
        x = bert_output[1]

        # Put input onto GPU if available
        x_tensor = torch.tensor(x, dtype=torch.float, device=self.device)

        # Put squashed inputs through linear layer
        output = self.linear_classifier(x_tensor)

        return output
