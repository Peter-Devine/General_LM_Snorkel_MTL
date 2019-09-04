import inspect
import os

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers.modeling_bert import BertModel

from snorkel.classification import DictDataset, DictDataLoader
from snorkel.classification import Operation
from snorkel.classification import Task
from snorkel.classification import MultitaskClassifier
from snorkel.classification import Trainer
from snorkel.analysis import Scorer

from snorkel.classification.training.loggers.log_writer import LogWriterConfig
from snorkel.classification.training.loggers.log_manager import LogManagerConfig

import utils.Classification_Task_Data_Handler as Classification_Task_Data_Handler
import utils.Tagging_Task_Data_Handler as Tagging_Task_Data_Handler

from modules.SnorkelFriendlyBert import SnorkelFriendlyBert
from modules.ClassificationLinearLayer import ClassificationLinearLayer
from modules.TaggingLinearLayer import TaggingLinearLayer

from loss_functions.tagging_cross_entropy import tagging_cross_entropy
from scoring_functions.tag_accuracy_scorer import tag_accuracy_scorer

import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--max_seq_length', default="512", help='Max size of the input in tokens')
parser.add_argument('--batch_size', default="32", help='Batch size of every dataset')
args = parser.parse_args()

MAX_SEQ_LENGTH = int(args.max_seq_length)
BATCH_SIZE = int(args.batch_size)

task_type_function_mapping = {
    "Classification_Tasks": {
        "data_handler": Classification_Task_Data_Handler,
        "head_module": ClassificationLinearLayer,
        "loss_function": F.cross_entropy,
        "scorer": Scorer(metrics=["accuracy"])
    },
    "Tagging_Tasks": {
        "data_handler": Tagging_Task_Data_Handler,
        "head_module": TaggingLinearLayer,
        "loss_function": tagging_cross_entropy,
        "scorer": Scorer(custom_metric_funcs={"Tag_accuracy": tag_accuracy_scorer})
    }
}

# Get the absolute current working directory of the project
cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# Create empty list to hold every Dataloader object
dataloaders = []

# Create empty list to hold every Task object
tasks = []

# Define the shared BERT layer to be used across tasks, and set the max seq length for the model.
shared_BERT_model = BertModel.from_pretrained('bert-base-uncased')
shared_BERT_model.config.max_position_embeddings = MAX_SEQ_LENGTH

# Confirm BERTs hidden layer size
hidden_layer_size = 768

# Make a module to contain the BERT module but can take the inputs of the Xs
bert_module = SnorkelFriendlyBert(bert_model=shared_BERT_model)

# Iterate through all task types
for task_type in ["Classification_Tasks", "Tagging_Tasks"]:

    # Get the contents of the data folder for the given task type
    target_data_path = os.path.join(cwd, "data", task_type)

    # Get names of all datasets in data folder
    task_names = [f for f in os.listdir(target_data_path) if not os.path.isfile(os.path.join(target_data_path, f))]

    print(f"Currently identified {task_type}:")
    print(", ".join(task_names))

    for task_name in task_names:
        print(f"Creating Task for {task_name}")

        task_type_specific_data_handler = task_type_function_mapping[task_type]["data_handler"]

        # Read data from given .tsv file and make it into structured and vectorized inputs/outputs
        split_datasets, output_label_to_int_dict = task_type_specific_data_handler.get_inputs_and_outputs(task_name, cwd, seq_len=MAX_SEQ_LENGTH)

        # Get splits of datasets
        train_dataset = split_datasets["train"]
        dev_dataset = split_datasets["dev"]
        test_dataset = split_datasets["test"]

        # Extract inputs to model (X) from splits
        train_X = train_dataset["input"]
        dev_X = dev_dataset["input"]
        test_X = test_dataset["input"]

        # Extract gold label outputs of model (y) from splits
        train_y = train_dataset["output"]
        dev_y = dev_dataset["output"]
        test_y = test_dataset["output"]

        # Get the number of classes included in dataset to use in task-specific head later
        num_classes = len(output_label_to_int_dict.keys())

        # Define dictionary keys for the data, dataset and task of the given task
        task_data_name = f"{task_name}_data"
        task_formal_name = f"{task_name}_task"
        task_dataset_name = f"{task_name}Dataset"

        for split, X, Y in (
            ("train", train_X, train_y),
            ("valid", dev_X, dev_y),
            ("test", test_X, test_y),
        ):
            X_dict = {task_data_name: torch.tensor(X, dtype=torch.long)}
            Y_dict = {task_formal_name: torch.tensor(Y, dtype=torch.long)}
            dataset = DictDataset(task_dataset_name, split, X_dict, Y_dict)
            dataloader = DictDataLoader(dataset, batch_size=BATCH_SIZE)
            dataloaders.append(dataloader)


        # Define a one-layer prediction "head" module specific to each task
        head_module = task_type_function_mapping[task_type]["head_module"](hidden_layer_size, num_classes)

        task_head_name = f"{task_name}_head_module"

        # The module pool contains all the modules this task uses
        module_pool = nn.ModuleDict({"bert_module": bert_module, task_head_name: head_module})

        # Operation with same name to all other tasks as it contains the shared bert_module
        op1 = Operation(
            name="bert_module", module_name="bert_module", inputs=[("_input_", task_data_name)]
        )

        # "Pass the output of op1 (the BERT module) as input to the head_module"
        op2 = Operation(
            name=task_head_name, module_name=task_head_name, inputs=["bert_module"]
        )

        op_sequence = [op1, op2]

        # Create the Task object, which includes the same name as that in dataloaders, all modules used,
        # and the sequence in which they are used.
        # Loss and scoring functions are added based on task type
        task_object = Task(
            name = task_formal_name,
            module_pool = module_pool,
            op_sequence = op_sequence,
            loss_func = task_type_function_mapping[task_type]["loss_function"],
            output_func = partial(F.softmax, dim=1),
            scorer = task_type_function_mapping[task_type]["scorer"],
        )

        # Add task to list of tasks
        tasks.append(task_object)

# Input list of tasks to MultitaskClassifier object to create model with architecture set for each task
model = MultitaskClassifier(tasks)

# Set out trainer settings - I.e. how the model will train
trainer_config = {
    "progress_bar": True,
    "n_epochs": 2,
    "lr": 0.02,
    "logging": True,
    "log_writer": "json",
    "checkpointing": True,
}

# Create trainer object using above settings
trainer = Trainer(**trainer_config)

# Train model using above settings on the datasets linked
trainer.fit(model, dataloaders)

# Output training stats of model
trainer.log_writer.write_log("output_statistics.json")

# Score model using test set and print
model_scores = model.score(dataloaders)
print(model_scores)
