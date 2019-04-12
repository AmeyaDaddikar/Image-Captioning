import torch
from torch import nn
import torchvision

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim):
        """
        :param encoder_dim: feature size of encoded images = 2048
        :param decoder_dim: size of decoder's RNN 
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, decoder_dim)  # linear layer to transform encoded image
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        #print (encoder_out.size())
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, decoder_dim)
        decoder_hidden_copy=decoder_hidden.unsqueeze(2)  #(batch_size, decoder_dim,1)
        score = torch.bmm(att1,decoder_hidden_copy)  # (batch_size , num_pixels ,1)
        
        score = score.squeeze(2) # (batch_size , num_pixels)
        
        alpha = self.softmax(score)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha