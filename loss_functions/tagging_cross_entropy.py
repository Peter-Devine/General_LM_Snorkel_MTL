import torch.nn.functional as F

# Make custom loss function such that it can compare the outputs of the model which are in one big batch list
# and the gold labels which are still in sentence lists inside batch lists
def tagging_cross_entropy(predictions, targets):
    flattened_targets = targets.reshape(targets.shape[0] * targets.shape[1])
    return F.cross_entropy(predictions, flattened_targets)