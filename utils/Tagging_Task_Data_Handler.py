from pytorch_transformers.tokenization_bert import BertTokenizer
from csv import reader as csv_reader
from os.path import join as join_path
from torch.nn.modules.loss import CrossEntropyLoss

def get_inputs_and_outputs(dataset_name, cwd, seq_len):
    # dataset_name: Name of folder in which target datset is in in the "Tagging_Task" folder
    # cwd: cwd path of the root script running the Snorkel MTL training
    
    split_datasets = {}
    
    for split in ["train", "dev", "test"]:
        with open(join_path(cwd, "data", "Tagging_Tasks", dataset_name, split+".tsv"), "r", encoding="utf-8") as f:
            reader = csv_reader(f, delimiter="\t")
            
            # Create lists in which to put inputs and outputs
            input_list = []
            output_list = []
            
            is_start = True
            
            for line in reader:
                # Get the header of the tsv file at the first line
                if is_start:
                    is_start = False
                    headers = line
                    
                    text_id = headers.index("text")
                    label_id = headers.index("tags")
                else:
                    text = line[text_id]
                    label = line[label_id]
                    
                    input_list.append(text)
                    output_list.append(label)
                    
        split_datasets[split] = {"input": input_list,
                                "output": output_list}
    
    # Get the tags for each split
    train_tags = split_datasets["train"]["output"]
    dev_tags = split_datasets["dev"]["output"]
    test_tags = split_datasets["test"]["output"]
    
    # And use the tags for each split to find the full class list of possible tags
    tag_set = get_total_tag_set(train_tags, dev_tags, test_tags)
    
    # Get a mapping of every tag to int so we can change them back again at the end
    tag_to_int_dict = make_tag_to_int_dict(tag_set)
    
    # Then process the input text, token-level tags and tag set to create inputs and outputs for the BERT model
    train_obs_and_tags = format_observation_for_bert_classifier(split_datasets["train"]["input"], train_tags, tag_set, seq_len)
    dev_obs_and_tags = format_observation_for_bert_classifier(split_datasets["dev"]["input"], dev_tags, tag_set, seq_len)
    test_obs_and_tags = format_observation_for_bert_classifier(split_datasets["test"]["input"], test_tags, tag_set, seq_len)
    
    # Save each input & output dictionary under its split name in a dictionary and return the whole dict
    split_datasets["train"] = train_obs_and_tags
    split_datasets["dev"] = dev_obs_and_tags
    split_datasets["test"] = test_obs_and_tags
    
    return split_datasets, tag_to_int_dict

def get_null_token():
    # Any output value that is passed to CrossEntropyLoss with the value of CrossEntropyLoss().ignore_index is ignored in the
    # loss calculation.
    NULL_TOKEN = CrossEntropyLoss().ignore_index
    return NULL_TOKEN


def format_observation_for_bert_classifier(observations, tags, tag_set, seq_len):
    # observations: List of observations in the form ["This is the first document", "This is the second", ...]
    # tags: List of corresponding token level tags in the form ["TAG1 TAG2 TAG2 TAG3 TAG2", "TAG2 TAG1 TAG4 TAG3", ...]
    # tag_set: List of all possible unique tags in the form ["TAG1", "TAG2", "TAG3", "TAG4" ...]
    # seq_len: Int or float length of sequence that will be used in model. E.g. 512

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    
    input_list = []
    output_list = []
    
    tag_to_int_dict = make_tag_to_int_dict(tag_set)
    
    for sentence, tag_seq in zip(observations, tags):
        # Break up each sentence/document into words and each tag sequence into individual tags
        # NB. when "sentence" is used in variable names, it simply refers to one text observation. 
        # In reality, this could consist of multiple sentences.
        word_list = sentence.split(" ")
        tag_list = tag_seq.split(" ")
        
        sentence_list = []
        tag_seq_list = []
        output_mask = []
        
        null_token = get_null_token()
        
        for word, tag in zip(word_list, tag_list):
            # Tokenize each word individually, and add to the input list (sentence level list).
            # Add the corresponding tags for each token to the output list (sentence tags level list).
            
            # Tokenize input (E.g. "puppeteer" -> ['puppet', '##eer'])
            tokenized_word = tokenizer.tokenize(word)
            
            # Convert all tokens into token id (E.g. ['puppet', '##eer'] -> [13997, 11510])
            tokenized_word_ids = tokenizer.convert_tokens_to_ids(tokenized_word)
            
            # Add token IDs to sentence level list
            sentence_list = sentence_list + tokenized_word_ids
            
            # Make tag an int specific to its class
            tag_int = tag_to_int_dict[tag]
            
            # Only propagate the error for the first token of a given word (E.g. if the proper noun "Johanson" (NNP) is tokenized to 
            # multiple tokens ["Johan", "##son"], then the loss on prediction will only be made at the first token position.)
            # Have a null token for all tokens after the first token of a multi-token word. The null token 
            # is ignored in the cross entropy loss function.
            # E.g. If 2 is the tag code for NNP, then the tag sequence for ["Johan", "##son"] would be [2, -100]
            tag_seq_list = tag_seq_list + [tag_int] + [null_token]*(len(tokenized_word)-1)
        
        segment_list = [0]*seq_len
        attention_list = [1]*len(sentence_list) + [0]*(seq_len-len(sentence_list))
        
        # Truncate/pad list to given sequence length size
        sentence_list = post_trunc_seq(sentence_list, seq_len)
        attention_list = post_trunc_seq(attention_list, seq_len)
        tag_seq_list = post_trunc_seq(tag_seq_list, seq_len, padding=null_token)
        
        assert len(sentence_list) == seq_len, "Token sequence is not equal to given sequence length"
        assert all([len(tag_seq_list) == len(sentence_list), len(attention_list) == len(sentence_list)]), "Token creation error - variable lengths of input sequences or output sequences"
        
        input_list.append([sentence_list, segment_list, attention_list])
        output_list.append(tag_seq_list)
        
    return ({"input": input_list, "output": output_list})

def get_tag_dummies(tag, tag_set):
    # Makes a dummy sequence using a tag and a set of all possible tags
    # label: label value (must be included in tag_set)
    # tag_set: list of all possible tag values
    tag_dummies = [0]*len(tag_set)
    tag_index = tag_set.index(tag)
    tag_dummies[tag_index] = 1
    return tag_dummies

def get_total_tag_set(train_tags, dev_tags, test_tags):
    # Takes train, dev and test split lists of tags sequences in the form ["TAG1 TAG2 TAG2 TAG3", "TAG2 TAG1 TAG4", ...]
    # train_tags: list of tag "sentences" (tags joined by spaces at the sentence level)
    
    tags_split = [tags.split(" ") for tags in train_tags+dev_tags+test_tags]
    
    flattened_tag_list = [tag for tags in tags_split for tag in tags]
    
    return sorted(list(set(flattened_tag_list)))

def make_tag_to_int_dict(tag_set):
    # Initialize empty label to int dictionary
    tag_to_int_dict = {}
    
    # Cycle though all tags, giving each a unique int as an identifier
    for i in range(len(tag_set)):
        tag_to_int_dict[tag_set[i]] = i
        
    return tag_to_int_dict

def get_total_number_of_classes(train_labels, dev_labels, test_labels):
    return len(get_total_tag_set(train_labels, dev_labels, test_labels))

def post_trunc_seq(seq, seq_length, padding=0):
    if len(seq) <= seq_length:
        return seq + [padding]*(seq_length-len(seq))
    else:
        return seq[0:seq_length]