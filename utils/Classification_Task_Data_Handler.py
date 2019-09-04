from pytorch_transformers.tokenization_bert import BertTokenizer
from csv import reader as csv_reader
from os.path import join as join_path

def get_inputs_and_outputs(dataset_name, cwd, seq_len):
    # dataset_name: Name of folder in which target datset is in in the "Tagging_Task" folder
    # cwd: cwd path of the root script running the Snorkel MTL training
    
    split_datasets = {}
    
    for split in ["train", "dev", "test"]:
        with open(join_path(cwd, "data", "Classification_Tasks", dataset_name, split+".tsv"), "r", encoding="utf-8") as f:
            reader = csv_reader(f, delimiter="\t")
            
            # Create lists in which to put inputs and outputs
            input_list = []
            output_list = []
            
            is_start = True
            
            for line in reader:
                # Get the header of the csv file at the first line
                if is_start:
                    is_start = False
                    headers = line
                    
                    text_A_id = headers.index("text_A")
                    
                    # If text_B column has not been supplied, then assign no text_B ID
                    if "text_B" in headers:
                        text_B_id = headers.index("text_B")
                    else:
                        text_B_id = None
                        
                    label_id = headers.index("label")
                else:
                    text_A = line[text_A_id]
                    
                    # If text_B column is not supplied, then column index is None, in which case we supply a zero length str as text_B
                    if text_B_id == None:
                        text_B = ""
                    else:
                        text_B = line[text_B_id]
                    
                    label = line[label_id]
                    
                    input_list.append({
                                "text_A": text_A,
                                "text_B": text_B
                    })
                    output_list.append(label)
        
        # Convert text observations to token IDs
        input_list = format_observations_for_bert_classifier(input_list, seq_len)
        
        split_datasets[split] = {"input": input_list,
                                "output": output_list}
    
    # Get outputs for each split
    train_outputs = split_datasets["train"]["output"]
    dev_outputs = split_datasets["dev"]["output"]
    test_outputs = split_datasets["test"]["output"]
    
    label_to_int_dict = make_label_to_int_dict(train_outputs, dev_outputs, test_outputs)
    
    # Convert all labels for each split to integers for use in BERT classifier
    train_outputs, dev_outputs, test_outputs = format_labels_for_bert_classifier(train_outputs, dev_outputs, test_outputs)
    
    # Save dummified labels
    split_datasets["train"]["output"] = train_outputs
    split_datasets["dev"]["output"] = dev_outputs
    split_datasets["test"]["output"] = test_outputs
    
    return split_datasets, label_to_int_dict

def format_observations_for_bert_classifier(observation_list, seq_len):
    # Takes a list of sentences/documents and sequence lenth params and outputs a list of inputs ready to be passed to BERT
    # observation_list: list of sentences/paragraphs/documents that you wish to convert to BERT friendly format
    # seq_len: float/int of size sequence should be truncated/padded to. Should be the same as BERT's max_seq_len parameter.
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    
    input_list = []
    
    for example in observation_list:
        # Get both sets of text (E.g. question and reference text)
        text_A = example["text_A"]
        text_B = example["text_B"]
        
        # Ititialize empty list to signify which tokens belong to which piece of text
        segment_ids = []
        
        if text_A:
            # Add [CLS] token to signify classification problem, and [SEP] token to separate text A and B
            text_A_with_operator_tokens = "[CLS] " + text_A + " [SEP] "
            
            # Tokenize input (E.g. "puppeteer" -> ['puppet', '##eer'])
            text_A_with_operator_tokens = tokenizer.tokenize(text_A_with_operator_tokens)
            
            # Convert all tokens into token id (E.g. ['puppet', '##eer'] -> [13997, 11510])
            tokenized_text_A = tokenizer.convert_tokens_to_ids(text_A_with_operator_tokens)
        else:
            tokenized_text_A = []
            
        if text_B:
            text_B_with_operator_tokens = text_B + " [SEP] "
            text_B_with_operator_tokens = tokenizer.tokenize(text_B_with_operator_tokens)
            tokenized_text_B = tokenizer.convert_tokens_to_ids(text_B_with_operator_tokens)
        else:
            tokenized_text_B = []

        # Make a list as long as the number of tokens in the concatenation of text_A tokens and text_B tokens
        # This list should have 0s for elements corresponding with to the text_A tokens (including [SEP] and [CLS])
        # and 1s for all text_B tokens
        segment_ids = [0]*len(tokenized_text_A) + [1]*len(tokenized_text_B)
        
        # Concatenate text_A and text_B tokens together. I.e. we stick both text inputs together, because we now
        # have a method for delineating which tokens correspond to which input through the segment_ids list
        text_ids = tokenized_text_A + tokenized_text_B
        
        # Create attention mask to only focus on non-padded tokens
        attention_mask = [1]*len(text_ids) + [0]*(seq_len-len(text_ids))
        
        # Truncate or pad sequences to match supplied sequence length
        text_ids = post_trunc_seq(text_ids, seq_len)
        segment_ids = post_trunc_seq(segment_ids, seq_len)
        attention_mask = post_trunc_seq(attention_mask, seq_len)
        
        assert len(text_ids) == seq_len, "Token sequence is not equal to given sequence length"
        assert all([len(text_ids) == len(segment_ids), len(text_ids) == len(attention_mask)]), "Token creation error - variable lengths of input sequences"
        
        input_list.append([text_ids, segment_ids, attention_mask])
    
    return input_list

def get_total_class_set(train_labels, dev_labels, test_labels):
    # Get set of unique labels (classes) in dataset
    # train_labels: list of training labels
    # dev_labels: list of dev labels
    # test_labels: list of test labels
    return sorted(list(set(train_labels + dev_labels + test_labels)))

def make_label_to_int_dict(train_labels, dev_labels, test_labels):
    # We make a list of all the unique possible classes
    class_set = get_total_class_set(train_labels, dev_labels, test_labels)
    
    # Initialize empty label to int dictionary
    label_to_int_dict = {}
    
    for i in range(len(class_set)):
        label_to_int_dict[class_set[i]] = i
        
    return label_to_int_dict

def format_labels_for_bert_classifier(train_labels, dev_labels, test_labels):
    # Converts list of strings to integers which stand for one of the predictable classes
    # train_labels: list of training labels
    # dev_labels: list of dev labels
    # test_labels: list of test labels
    
    # We make a list of all the unique possible classes
    label_to_int_dict = make_label_to_int_dict(train_labels, dev_labels, test_labels)
    
    # We convert labels to integer labels specific to their class
    train_labels_int = [label_to_int_dict[train_label] for train_label in train_labels]
    dev_labels_int = [label_to_int_dict[dev_label] for dev_label in dev_labels]
    test_labels_int = [label_to_int_dict[test_label] for test_label in test_labels]
    
    return train_labels_int, dev_labels_int, test_labels_int
    
def get_total_number_of_classes(train_labels, dev_labels, test_labels):
    # Get number of classes in dataset
    # train_labels: list of training labels
    # dev_labels: list of dev labels
    # test_labels: list of test labels
    return len(get_total_class_set(train_labels, dev_labels, test_labels))

def post_trunc_seq(seq, seq_length, padding=0):
    # Pads or truncates sequence to given length
    # seq: list of some size
    # seq_length: float or int of size to which seq should be truncated/padded
    if len(seq) <= seq_length:
        return seq + [padding]*(seq_length-len(seq))
    else:
        return seq[0:seq_length]