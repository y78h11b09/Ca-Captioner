import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from torch.autograd import Variable
from torchvision import transforms
from build_vocab import Vocabulary
from backbone.backbonebase import Encoder
# from transformer.models import Transformer
from backbone.swin384 import SwinTransformer
from transformersepov.models3 import Transformer
from PIL import Image
import json
import torch.nn as nn
from data_loader import get_loader
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose(
    [transforms.RandomCrop(args.crop_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    args.src_vocab_size = len(vocab)
    args.tgt_vocab_size = len(vocab)
    # data_loader_eval = get_loader(args.image_dir_eval, args.caption_path_eval, vocab, transform, 1,
    #                               shuffle=False,
    #                               num_workers=args.num_workers)
    encoder = SwinTransformer(img_size=384,
                              embed_dim=192,
                              depths=[2, 2, 18, 2],
                              num_heads=[6, 12, 24, 48],
                              window_size=12,
                              num_classes=1000).to(device)
    print('load pretrained weights!')
    encoder.load_weights(
        './weights/swin_large_patch4_window12_384_22kto1k_no_head.pth'
    )
    # Freeze parameters
    for _name, _weight in encoder.named_parameters():
        _weight.requires_grad = False
    decoder = Transformer(n_layers_dec=3, n_layers_enc=9, d_k=64, d_v=64, d_model=1536, d_ff=2048, n_heads=8,
                          max_seq_len=50,
                          tgt_vocab_size=len(vocab), dropout=0.1).to(device)


    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    encoder.eval()
    decoder.eval()
    image_dir = args.image_dir
    images = os.listdir(image_dir)
    img_s = {}
    img_dict = {}
    prob_proj = nn.LogSoftmax(dim=-1)

    # flag=[]
    # total_step=len(data_loader_eval)
    for i, image in enumerate(images):
        img_path = os.path.join(image_dir, image)
        # img_dict[image] = str(img_path)
        with open(img_path, 'r+b') as f:
            with Image.open(f) as img:
                # resize image
                # print(img.shape)
                img = transform(img).unsqueeze(0)
                # print(img.shape)(1,3,224,224)
                img_dict[image] = img
    for img_key,img in img_dict.items():
            img_var = Variable(img)
            img_var = img_var.to(device)

            y = encoder(img_var)
            #enc_outputs=y.to(device)
            enc_outputs, _ = decoder.encode(y)
            beam_size=1
            k_prev_words = torch.LongTensor([[vocab.word2idx['<start>']]]* beam_size).to(device)
            top_k_scores = torch.zeros(beam_size, 1).to(device)
            complete_seqs = list()
            complete_seqs_scores = list()
            for step in range(args.max_decode_step):
                len_dec_seq = step+1
                dec_partial_inputs_len = torch.tensor([len_dec_seq]*beam_size).long().to(device)
                enc_output = enc_outputs.repeat(1, beam_size, 1).view(
                    beam_size, enc_outputs.size(1), enc_outputs.size(2))
                dec_out,_,_=decoder.decode(k_prev_words,dec_partial_inputs_len,enc_output)
                scores=decoder.tgt_proj(dec_out)
                scores=prob_proj(scores)

                for t in range(scores.size(0)):
                    scores[t,-1,:]+=top_k_scores[t,]
                if step==0:
                    top_k_scores, top_k_words = scores[0,-1,].topk(beam_size, 0, True, True)
                else:
                    scores=scores[:,-1,:]
                    top_k_scores, top_k_words = scores.reshape(-1).topk(beam_size, 0, True, True)
                prev_word_inds = top_k_words / len(vocab) # (s)
                next_word_inds = top_k_words % len(vocab)
                p = prev_word_inds.type(torch.LongTensor)
                prev_word_inds = p
                k_prev_words = torch.cat([k_prev_words[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != vocab.word2idx['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                if len(complete_inds) > 0:
                    complete_seqs.extend(k_prev_words[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                beam_size -= len(complete_inds)  # reduce beam length accordingly
                # Proceed with incomplete sequences
                if beam_size == 0:
                    break
                k_prev_words = k_prev_words[incomplete_inds]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            #print(complete_seqs_scores)
            sampled_re=[]
            m = complete_seqs_scores.index(max(complete_seqs_scores))
            #print(m)
            seq = complete_seqs[m]
            for word_id in seq:
                #if word_id not in {vocab.word2idx['<start>'], vocab.word2idx['<end>'], vocab.word2idx['<pad>'],vocab.word2idx['.']}:
                word = vocab.idx2word[word_id]
                sampled_re.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_re)
            img_s[img_key] = sentence
    json_dic = json.dumps(img_s, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=True)
    print(json_dic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
    parser.add_argument('--image', type=str,default='data/',
                        help='input image for generating caption')
    """
    parser.add_argument('--image_dir', type=str,default='data/image_datang_resized2',
                        help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='attnpov2(1 38.3 122)/encoder-1.ckpt',#default .pkl
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='attnpov2(1 38.3 122)/decoder-1.ckpt',#default.pkl
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--crop_size', type=int, default=384, help='size for center cropping images')

    # Model parameters (should be same as paramters in trainRT.py)
    parser.add_argument('--batch_size', type=int, default=1)  # yuan128
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--max_decode_step', type=int, default=100)
    args=parser.parse_args()
    main(args)
    #args = parser.parse_args()
    #main(args)
