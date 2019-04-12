import torch
from src.decoder_with_attention import DecoderWithAttention
from src.encoder import Encoder

import torchvision.transforms as transforms
from skimage import io, transform
import torch.nn.functional as F

from collections import Counter

# Model parameters
emb_dim = 512  # dimension of word embeddings
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)

word_freq = Counter()

words = [w for w in word_freq.keys()  if word_freq[w] > min_word_freq]
word_map = {k: v + 1 for v, k in enumerate(words)}
word_map['<unk>'] = len(word_map) + 1
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0


checkpoint = '../input/image-copy-2/checkpoint_copy.pt'

decoder = DecoderWithAttention(embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=len(word_map),
                                   dropout=dropout)
decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                     lr=decoder_lr)

encoder = Encoder()


# Move to GPU, if available
decoder = decoder.to(device)
encoder = encoder.to(device)

decoder.eval()
encoder.eval()
from scipy.misc import imread, imresize


if checkpoint is not None:
  checkpoint = torch.load(checkpoint)
  decoder.load_state_dict(checkpoint['decoder_state_dict'])
  decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_dict'])



def predict(img_path):
  sampled = []
  transform=transforms.Compose([transforms.Resize((256,256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
                                           ])
  
  image = transform(Image.open(image))

  image = torch.FloatTensor(image).to(device)
  enc_image = encoder(image)
  enc_image = enc_image.view(1, -1, 2048)
  print ('ENCODED IMG SIZE:' ,enc_image.size())
  h,c = decoder.init_hidden_state(enc_image) 
  pred = torch.LongTensor([[word_map['<start>']]]).to(device)
  ans=[]


  for t in range(50):
        
        embeddings = decoder.embedding(pred).squeeze(1)  
        attention_weighted_encoding, alpha = decoder.attention(enc_image,h)
        gate = decoder.sigmoid(decoder.f_beta(h))  
        #print ("BEFORE GATE")
        #print (attention_weighted_encoding.size())
        attention_weighted_encoding = gate * attention_weighted_encoding
        #print ("AFTER GATE")
        #print (embeddings.size())
        #print (attention_weighted_encoding.size())
        h, c = decoder.decode_step(
            torch.cat([embeddings, attention_weighted_encoding], dim=1),(h, c))  # (batch_size_t, decoder_dim)
        pt = decoder.fc(decoder.dropout(h))  # (batch_size_t, vocab_size)
        #print (pt.size())
        _,pred = pt.max(1)
        
        data = F.softmax(pt.squeeze(0), dim=0)
        
        sampled.append(pred.item())
        #print (pred)
        ans.append(pred.item())
        n = len(ans)
        if (ans[n-1] == word_map['<end>']):
            break
        
  generated_words = [rev_word_map[sampled[i]] for i in range(len(sampled))]
  #print (ans)
  return generated_words
  