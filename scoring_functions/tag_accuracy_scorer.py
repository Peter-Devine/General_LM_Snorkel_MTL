from sklearn.metrics import accuracy_score
import utils.Tagging_Task_Data_Handler as Tagging_Task_Data_Handler

# Create a scoring function to change the formatting of gold labels from list of sentence lists
# to a big list of tags.
# Also, ignore all labels which were ignored in training (I.e. null labels)
def tag_accuracy_scorer(golds, preds, probs):
    # NB. We do not use the probabilities (probs) of the predictions but we include them as an argument
    # as they are called in the Snorkel code.
    
    # Get the null label signifier
    null_label = Tagging_Task_Data_Handler.get_null_token()
    
    # Flatten the gold list such that the gold labels are now in one big list, not in sentence lists
    golds_flattened = golds.reshape(golds.shape[0]*golds.shape[1])
    
    # Only select golds which do not have null labels
    non_null_gold_mask = golds_flattened != null_label
    
    # Find golds and predictions which correspond to non-null label values
    non_null_golds = golds_flattened[non_null_gold_mask]
    non_null_preds = preds[non_null_gold_mask]
    
    return accuracy_score(non_null_golds, non_null_preds)