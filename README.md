# pytorch-templates
My repository of templates for starting a pytorch project

Some notes about pytorch:

# Implement a `Module`
```
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))
```

Important functions:
`model.parameters()`

# Data
Example dataset:
```
import torch.utils.data as data

class Data(data.Dataset):
    def __init__(self, mydata):
        self.data = mydata

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

```
### Samplers
- `data.SequentialSampler(data_source)`
- `data.RandomSampler(data_source, replacement=False, num_samples=None)`
- `data.BatchSampler(sampler, batch_size, drop_last)`


# Linear Layers

`nn.Linear(in_features, out_features)`
- Input (N,...,in_features)
- Output (N,...,out_features)

`nn.Bilinear(in1_features, in2_features, out_features)`
- Input (N,...,in1_features),(N,...,in2_features)
- Output (N,...,out_features)

# Dropout
`nn.Dropout(p=0.5)`

# RNN

`nn.RNN()`
- input_size – The number of expected features in the input x
- hidden_size – The number of features in the hidden state h
- num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
- nonlinearity – The non-linearity to use. Can be either ‘tanh’ or ‘relu’. Default: ‘tanh’
- bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
- batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
- dropout – If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
- bidirectional – If True, becomes a bidirectional RNN. Default: False

### Input `input, h_0`

- (seq_len, batch, input_size)
- h_0 of shape (num_layers * num_directions, batch, hidden_size)

### Output `input, h_n`

- (seq_len, batch, num_directions * hidden_size)
- h_n (num_layers * num_directions, batch, hidden_size)

### Unpack directions
`output.view(seq_len, batch, num_directions, hidden_size)`

# LSTM

`nn.LSTM`
### Input `input, (h_0, c_0)`

- (seq_len, batch, input_size)
- h_0 of shape (num_layers * num_directions, batch, hidden_size)
- c_0 of shape (num_layers * num_directions, batch, hidden_size)

### Output `output, (h_n, c_n)`

- (seq_len, batch, num_directions * hidden_size)
- h_n (num_layers * num_directions, batch, hidden_size)
- c_n (num_layers * num_directions, batch, hidden_size)

# GRU
`nn.GRU`
- Same as RNN

# Embeddings

`nn.Embedding(num_embeddings, embedding_dim)`
- num_embeddings (int) – size of the dictionary of embeddings
- embedding_dim (int) – the size of each embedding vector
- padding_idx (int, optional) – If given, pads the output with the embedding vector at padding_idx (initialized to zeros) whenever it encounters the index.
- max_norm (float, optional) – If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm.
- norm_type (float, optional) – The p of the p-norm to compute for the max_norm option. Default 2.
- scale_grad_by_freq (boolean, optional) – If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default False.
- sparse (bool, optional) – If True, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for more details regarding sparse gradients.

### Input: 

LongTensor of arbitrary shape containing the indices to extract

### Output: 

(..., embedding_dim)

### Methods

`from_pretrained(embeddings, freeze=True, sparse=False)`


# Activations

- `nn.ReLU()`
- `nn.LeakyReLU(negative_slope=0.01)`
- `nn.Sigmoid()`
- `nn.Softmax()`
- `nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=4.0, head_bias=False)`


# Losses

- `nn.CrossEntropyLoss()`
- `nn.BCELoss()`
- `nn.BCEWithLogitsLoss()`

# Optimization

`import torch.optim as optim`

`SGD`, `Adagrad`, `Adam`, `Adadelta`, etc.

Example:
```
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

# Regularization

`nn.L1Loss()`
