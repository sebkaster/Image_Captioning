import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """ initialize decoder """
        
        # initialize superclass
        super().__init__()
        
        self.n_hidden_dim = hidden_size
        self.hidden_state = None
        
        # turns words into numerical vectors of specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm_cell = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, \
                                 num_layers =num_layers, bias=True, batch_first=True, \
                                 dropout=0, bidirectional=False)
        
        # maps the hidden state to our desired output
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        pass
    
    def forward(self, features, captions):
        """ forward propagation """
        # remove <end> tag
        captions = captions[:, :-1]
        
        # init the hidden and cell states to zeros
        self.hidden_state = self.init_hidden(features.size(0))

        # create embedded word vectors for each word in captions
        embeddings = self.word_embeddings(captions) 
        
        # stack features and captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)  
        
        # get output, and the new hidden state (h, c) from the lstm cell
        lstm_output, self.hidden_state = self.lstm_cell(embeddings, self.hidden_state)

        # put lstm_output through the fully-connected layer
        final_outputs = self.linear(lstm_output)

        return final_outputs
    
    def init_hidden(self, batch_size, num_layers=1):
        """ init hidden state to zeros at beginning of training """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (torch.zeros((num_layers, batch_size, self.n_hidden_dim), device=device), \
                torch.zeros((num_layers, batch_size, self.n_hidden_dim), device=device))

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # init
        output = []
        hidden = self.init_hidden(inputs.size(0))
    
        while True:
            lstm_output, hidden = self.lstm_cell(inputs, hidden) 
            outputs = self.linear(lstm_output)
            
            # predict the most likely next word
            outputs = outputs.squeeze(1)
            _, max_indice = torch.max(outputs, dim=1) 
            
            output.append(max_indice.cpu().numpy()[0].item()) # storing the word predicted
            
            if (max_indice == 1):
                # break if <end> detected
                break
            
            # embed the last predicted word to be the new input of the lstm
            inputs = self.word_embeddings(max_indice)
            inputs = inputs.unsqueeze(1)
            
        return output
