import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import random
from build_vocab import Vocabulary
from data_loader import get_loader
from backbone.swin384 import SwinTransformer
from transformersepov.models3 import Transformer
from torchvision import transforms

# Device configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
# torch.backends.cudnn.benchmark = True

random.seed(12)
torch.manual_seed(12)
np.random.seed(12)


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose(
        [transforms.RandomCrop(args.crop_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader_train = get_loader(args.image_dir_train, args.caption_path_train, vocab, transform, args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)

    data_loader_test = get_loader(args.image_dir_test, args.caption_path_test, vocab, transform, args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers)
    # Build the models

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
    decoder = Transformer(n_layers_dec=3, n_layers_enc=9, d_k=64, d_v=64, d_model=1536, d_ff=2048, n_heads=8,max_seq_len=50,
                          tgt_vocab_size=len(vocab),dropout=0.1 ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # decoder_optimizer = Ranger(decoder.parameters(), lr=1e-3, weight_decay=1e-3)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(decoder_optimizer, T_0=5)
    #params = list(decoder.parameters()) + list(encoder.adp1.parameters()) +list(encoder.adp2.parameters())
    encoder.eval()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.000001)#,weight_decay=0.00000025)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20)

    # total_step_train = 20
    # total_step_val = 20

    total_step_train = len(data_loader_train)
    total_step_test = len(data_loader_test)

    loss_train_list = []
    loss_test_list = []
    #
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))



    for epoch in range(args.num_epochs):
        total_loss_train = 0.0
        total_loss_test  = 0.0
        for i, (images, captions, lengths) in enumerate(data_loader_train):
            # Set mini-batch train_src
            # if i == 2000:
            #     break
            images = images.to(device)
            enc_outputs = encoder(images)
            #print(len(enc_outputs))
            #enc_outputs = enc_outputs.to(device)
            encoded_outputs,_ = decoder.encode(enc_outputs)
            #enc_outputs = enc_outputs.to(device)
            # print(enc_outputs.shape)
            dec_inputs_len = torch.tensor(lengths).to(device)
            dec_inputs_len = (dec_inputs_len - 1).to(device)
            dec_inputs = (captions[:, :-1]).to(device)  # 除最后一行的所有行所有列
            # tgt_pos = map(lambda x: x.to(device))
            #print((enc_outputs.shape))#(2,196,2048)
            outputs, _ = decoder(enc_outputs, dec_inputs, dec_inputs_len)
            targets = (captions[:, 1:]).to(device)  # 从第2列的所有行所有列
            loss_train = criterion(outputs, targets.contiguous().view(-1))  # contiguous(),把tensor变成在内存中连续分布的形式
            optimizer.zero_grad()
            loss_train.backward()

            optimizer.step()
            # scheduler.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], train_Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, args.num_epochs, i,
                                                                                              total_step_train, loss_train.item(),
                                                                                              np.exp(loss_train.item())))


            total_loss_train += loss_train.item()
        #lr_scheduler.step()
        torch.save(decoder.state_dict(), os.path.join(args.model_path, 'decoder-{}.ckpt'.format(epoch + 1)))
        torch.save(encoder.state_dict(), os.path.join(args.model_path, 'encoder-{}.ckpt'.format(epoch + 1)))

        # for i, (images, captions, lengths) in enumerate(data_loader_test):
        #     # Set mini-batch train_src
        #     # if i == 20:
        #     #     break
        #     images = images.to(device)
        #     enc_outputs = encoder(images)
        #     #enc_outputs = enc_outputs.to(device)
        #     # print(enc_outputs.shape)
        #     dec_inputs_len = torch.tensor(lengths).to(device)
        #     dec_inputs_len = (dec_inputs_len - 1).to(device)
        #     dec_inputs = (captions[:, :-1]).to(device)  # 除最后一行的所有行所有列
        #     # tgt_pos = map(lambda x: x.to(device))
        #     outputs, _ = decoder(enc_outputs, dec_inputs, dec_inputs_len)
        #     targets = (captions[:, 1:]).to(device)  # 从第2列的所有行所有列
        #     loss_test = criterion(outputs, targets.contiguous().view(-1))  # contiguous(),把tensor变成在内存中连续分布的形式
        #
        #
        #     # Print log info
        #     if i % args.log_step == 0:
        #         print('Epoch [{}/{}], Step [{}/{}], val_Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, args.num_epochs, i,
        #                                                                                       total_step_test, loss_test.item(),
        #                                                                                       np.exp(loss_test.item())))
        #     # print(loss.item())
        #
        #     total_loss_test += loss_test.item()
        avg_loss_train = float('%.4f' % (total_loss_train / total_step_train))
        avg_loss_val = float('%.4f' % (total_loss_test / total_step_test))
        loss_train_list.append(avg_loss_train)
        loss_test_list.append(avg_loss_val)



    # 绘制loss vs epoch 图
    # x = np.arange(0,0+args.num_epochs,5).astype(dtype=np.str)
    x = np.arange(1, 1 + args.num_epochs).astype(dtype=np.str)
    # y = Loss_list[0:-1:5]
    y1 = loss_train_list
    y2 = loss_test_list

    plt.plot(x, y1, 'bo-',alpha=0.5, linewidth=1,label = 'train_loss')
    plt.plot(x, y2, 'ro-', alpha=0.5, linewidth=1, label='val_loss')
    plt.legend()
    plt.title('loss .vs. epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('./train_loss2.png')
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='attnpov201/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=384, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir_train', type=str, default='/media/huashuo/mydisk/cocodataset/coco_karpathy/train2014',
                        help='directory for resized images')

    parser.add_argument('--image_dir_test', type=str, default='/media/huashuo/mydisk/cocodataset/coco_karpathy/train2014',
                        help='directory for resized images')

    parser.add_argument('--caption_path_train', type=str, default='/media/huashuo/mydisk/cocodataset/coco_karpathy/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--caption_path_test', type=str,
                        default='/media/huashuo/mydisk/cocodataset/coco_karpathy/annotations/captions_test2014.json',
                        help='path for val annotation json file')
    parser.add_argument('--log_step', type=int, default=1, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=20)  # yuan128
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=0.000001)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--encoder_path', type=str, default='attnpov20/encoder-3.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='attnpov20/decoder-3.ckpt', help='path for trained decoder')
    args = parser.parse_args()
    print(args)
    main(args)
