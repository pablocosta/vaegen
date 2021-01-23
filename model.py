import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var


class SenteceVAE(nn.Module):
    def __init__(self, vocabSize, embeddingsSize=50, recurrenceType="rnn", hiddenSize=80, wordDropout=0.5, embeddingDropout=0.1, latentSize=25, maxSequenceLenght=10, 
                 numLayers=1, biDirectional=True):
        
        self.vocabSize = vocabSize
        self.embeddingsSize = embeddingsSize
        self.recurrenceType = recurrenceType
        self.hiddenSize = hiddenSize
        self.wordDropout = wordDropout
        self.embeddingDropout = embeddingsSize
        self.latentSize = latentSize
        self.maxSequenceLenght = maxSequenceLenght
        self.numLayers = numLayers
        self.bidirectional= biDirectional
        
        if self.recurrenceType == "rnn":
            rnn = nn.RNN()
        elif self.recurrenceType == "gru":
            rnn = nn.GRU()
        elif self.recurrenceType == "lstm":
            renn = nn.LSTM()
        else:
            raise ValueError()
            
        self.embeddings = nn.Embedding(self.vocabSize, self.embeddingsSize)
        self.embDropout = nn.Dropout(p=self.embeddingDropout)

        self.encoder = rnn(self.embeddingsSize, self.hiddenSize, num_layers=self.numLayers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder = rnn(self.embeddingsSize, self.hiddenSize, num_layers=self.numLayers, bidirectional=self.bidirectional, batch_first=True)
        
        self.hiddenFactor = (2 if self.bidirectional else 1) * self.numLayers
        
        self.hidden2mean = nn.Linear(self.hiddenSize * self.hiddenFactor, latentSize)
        self.hidden2logv = nn.Linear(self.hiddenSize * self.hiddenFactor, latentSize)
        self.latent2hidden = nn.Linear(self.latentSize, self.hiddenSize * self.hiddenFactor)
        self.outputs2vocab = nn.Linear(self.hiddenSize * (2 if self.bidirectional else 1), self.vocabSize)
        
    def forward(self, inputSequence, lenght):
        
        batchSize = inputSequence.size(0)
        
        #sort batch
        sortedLenght, indices = torch.sort(lenght, descending=True)
        inputSequence = inputSequence[indices]
        
        #encoder part
        inputEmbeddings = self.embeddings(inputSequence)
        
        #pack sentences (for minibatch)
        packedInput = nn.utils.rnn.pack_padded_sequence(inputEmbeddings, sortedLenght.data.tolist(), batch_first=True)
        #rnn pass by
        _, hidden = self.encoder(packedInput)
        
        if self.bidirectional or self.numLayers > 1:
            # flatten hidden state
            hidden = hidden.view(batchSize, self.hiddenSize*self.hiddenFactor)
        else:
            hidden = hidden.squeeze()
        #reparametrization ~= sampling
        
        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = torch.randn([batchSize, self.latentSize]).cuda()
        z = z * std + mean 
        
        
        hidden = self.latent2hidden(z)
        
        if self.bidirectional or self.numLayers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hiddenFactor, batchSize, self.hiddenSize)
        else:
            hidden = hidden.squeeze()
            
        #feed the input to the decoder
        
        if self.wordDropout > 0:
            #then apply word dropout
            # randomly replace decoder input with <unk>
            prob = torch.rand(inputSequence.size())