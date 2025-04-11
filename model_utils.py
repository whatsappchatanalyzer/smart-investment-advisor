import torch
from rnn_model import RNN

def load_model(path="rnn_stock_model.pth"):
    model = RNN(input_size=1, hidden_size=32, output_size=1)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict_trend(model, price_list):
    inputs = torch.tensor(price_list).float().unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        output = model(inputs)
    return output.item()
