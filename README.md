# Grokking
1、The project contains three py files, the main program is “main.py”, which supports GPU and CPU.

2、Among them, the adjusted superparameters mainly include:

p：prime number p
K：the number of summands
Iters: optimization steps
Save_iters: Save the checkpoint every save_iters.
Optimizer: you can choose' AdamW',' SGD' and' RMSprop'.
Nn_model:' Transformer',' MLP' and' LSTM' can be selected.
training_fraction
batch_size
learning_rate
weight_decay
dropout

The specific settings of all parameters are introduced in detail in the paper.

3、The scale of the model and other parameters in the optimizer are not regarded as superparameters by default, and can be modified in the main program if necessary.
If you need to resume training from a checkpoint, you need to set “resume_from_checkpoint” to “True” and change “checkpoint_path”.
