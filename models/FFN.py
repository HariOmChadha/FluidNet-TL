import os
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, input_size, config):
        super(FFN, self).__init__()
        
        self.batch_norm_input = nn.BatchNorm1d(input_size)
        
        self.hidden_layer_1 = nn.Linear(input_size, 300, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(300)
        
        self.hidden_layer_2 = nn.Linear(300, 150, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(150)

        self.batch_norm_temp = nn.BatchNorm1d(5, momentum=0.01, eps=0.001)

        self.ln_A = nn.Linear(150, 1, bias=False)
        self.B = nn.Linear(150, 1, bias=False)

        self._initialize_weights()

        # Save model name and base path
        self.model_folder_path = os.path.join(config['save_folder'], config['name'])
        os.makedirs(self.model_folder_path, exist_ok=True)

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.hidden_layer_1.weight)
        nn.init.xavier_uniform_(self.hidden_layer_2.weight)
        nn.init.xavier_uniform_(self.ln_A.weight)
        nn.init.xavier_uniform_(self.B.weight)

    def forward(self, x, temperature):
        x = self.batch_norm_input(x)
        x = self.hidden_layer_1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)

        x = self.hidden_layer_2(x)
        x = self.batch_norm2(x)
        x = F.leaky_relu(x, negative_slope=0.1)

        #temperature = self.batch_norm_temp(temperature)

        ln_A = self.ln_A(x)
        B = self.B(x)

        output = ln_A + B / temperature
        
        return output, ln_A, B