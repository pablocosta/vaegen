import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import *


class SenteceVAE(nn.Module):
    def __init__(self, vocabSize, embeddingsSize=50, recurrenceType="rnn", hiddenSize=80, wordDropout=0.5, embeddingDropout=0.1, latentSize=25, maxSequenceLenght=10, 
                 numLayers=1, biDirectional=True):
        
        self.tensor = torch.cuda.FloatTensor 
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
            prob = torch.rand(inputSequence.size()).cuda()
            prob[(inputSequence.data - SOSToken) * (inputSequence.data - PADToken) == 0] = 1
            decoderInputSequence = inputSequence.clone()
            decoderInputSequence[prob < self.wordDropout] = UNKToken
            inputEmbeddings = self.embedding(decoderInputSequence)
        inputEmbeddings = self.embDropout(inputEmbeddings)
        packedInput = rnn_utils.pack_padded_sequence(inputEmbeddings
                                                    ,sortedLenght.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder(packedInput, hidden)
        # process outputs
        paddedOutputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        paddedOutputs = paddedOutputs.contiguous()
        #get original indices
        _,reversedIndex = torch.sort(indices)
        paddedOutputs = paddedOutputs[reversedIndex]
        b,s,_ = paddedOutputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(paddedOutputs.view(-1, paddedOutputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp, mean, logv, z
    
    
    def inference(self, n=4, z=None):
        
        if z is None:
            batchSize = n
            z = (torch.randn([batchSize, self.latentSize])).cuda()
        else:
            batchSize = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hiddenFactor, batchSize, self.hiddenSize)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequenceIdx = torch.arange(0, batchSize, out=self.tensor()).long()  # all idx of batch
        # all idx of batch which are still generating
        seqRunning = torch.arange(0, batchSize, out=self.tensor()).long()
        seqMask = torch.ones(batchSize, out=self.tensor()).bool()
        # idx of still generating sequences with respect to current loop
        runSequences = torch.arange(0, batchSize, out=self.tensor()).long()

        generations = self.tensor(batchSize, self.maxSequenceLenght).fill_(PADToken).long()

        t = 0
        while t < self.maxSequenceLenght and len(runSequences) > 0:

            if t == 0:
                inSequence = (torch.Tensor(batchSize).fill_(SOSToken).long()).cuda()

            inSequence = inSequence.unsqueeze(1)

            inpEmbedding = self.embedding(inSequence)

            output, hidden = self.decoder_rnn(inpEmbedding, hidden)

            logits = self.outputs2vocab(output)

            inSequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, inSequence, runSequences, t)

            # update gloabl running sequence
            seqMask[seqRunning] = (inSequence != EOSToken)
            seqRunning = sequenceIdx.masked_select(seqMask)

            # update local running sequences
            runMask = (inSequence != EOSToken).data
            runSequences = runSequences.masked_select(runMask)

            # prune input and hidden state according to local update
            if len(runSequences) > 0:
                inSequence = inSequence[runSequences]
                hidden = hidden[:, runSequences]

                runSequences = torch.arange(0, len(runSequences), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.reshape(-1)

        return sample

    def _save_sample(self, saveTo, sample, runSeqs, t):
        # select only still running
        runLatest = saveTo[runSeqs]
        # update token at position t
        runLatest[:,t] = sample.data
        # save back
        saveTo[runSeqs] = runLatest

        return saveTo