# General_LM_Snorkel_MTL

## How to run:
1. Delete the `DELETE_ME_EXAMPLE*.tsv` files in the `data` directory
2. Copy your desired data files into the `data/Classification_Tasks` or `data/Tagging_Tasks` depending on the task type.
3. Make sure that all classification tasks have the desired text for analysis under the column headers `text_A` and the optional header of `text_B`, with the label of the text being under the header `label`.
4. Make sure that all tagging tasks have the text for tagging labelled under `text` and the **space separated tags** labelled under `tags`.
5. Run the `Snorkel_MTL.py` file with the following optional parameters `--max_seq_length` and `--batch_size` to detemine the size of the overall model and the batch size of all training data respectively.
6. Output will then be generated in a newly created `logs` folder.
