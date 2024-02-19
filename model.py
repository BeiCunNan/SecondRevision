from torch import nn


class FNN(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.base_model.config.hidden_size, 100),
            nn.Linear(100, num_classes)
        )
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        cls_feats = raw_outputs.last_hidden_state[:, 0, :]
        predicts = self.fc(cls_feats)
        return predicts
