import torch
import torch.nn as nn

# Create special module that squashes BERT output of batch x words x hidden into series of 
# individual token level tag classifications.
class TaggingLinearLayer(nn.Module):

    def __init__(self,input_size, num_labels):
        
        super(TaggingLinearLayer, self).__init__()
        self.linear_classifier = nn.Linear(input_size, num_labels)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')

    def forward(self, bert_output):
        # Flatten list from list of sentences, each with a list of outputs at each token (of dimension 768 for uncased BERT)
        # into a single list of outputs from different token positions.
        # I.e. x goes from (N, W, H) -> (N * W, H) where N is the number of documents/sentences in the batch,
        # W is the number of tokens in each document (the max sequence length value) and H is the size of the hidden layer.
        
        # Take the hidden layer at every token position of BERT output 
        x = bert_output[0]
        
        # Put input onto GPU if available
        x_tensor = torch.tensor(x, dtype=torch.float, device=self.device)
        
        # Squash outputs to a per-token basis
        x_tensor = x_tensor.reshape(x_tensor.shape[0] * x_tensor.shape[1], x_tensor.shape[2])
        
        # Put squashed inputs through linear layer
        output = self.linear_classifier(x_tensor)
        
        return output