import torch
from tqdm import tqdm


class Model(torch.nn.Module):
    
    def __init__(self, data):
        super().__init__()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.SGD(self.parameters(), lr=0.001)
        self.transformer = torch.nn.Transformer(
            nhead=16,
            num_encoder_layers=8,
            num_decoder_layers=8
        )

    def forward(self, data):
        pass

    def train_model(self):
        pass

    def test_model(self):
        pass

    

