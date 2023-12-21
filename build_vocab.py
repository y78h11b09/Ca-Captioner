import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()#create the counter, default value is 0;
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])#get  the caption 'sentence' of each row of coco train_src
        tokens = nltk.tokenize.word_tokenize(caption.lower())#sentence tokenizer words for each captions: ['a', 'panoramic', 'view', 'of', 'a', 'kitchen', 'and', 'all', 'of', 'its', 'appliances', '.']
        counter.update(tokens)#add tokens to counter: Counter({'a': 1, 'very': 1, 'clean': 1, 'and': 1, 'well': 1, 'decorated': 1, 'empty': 1, 'bathroom': 1})

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    #idx2word={0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>', 4: 'a', 5: 'very', 6: 'clean', 7: 'and',....}
    #word2idx ={'<pad>':0,'<start>':1,...}
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(json=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default= 'D:\\shuju\coco2014\\annotations\\captions_train2014.json',#原E:/datasets/COCO2014-2015/annotations/captions_train2014.json
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', #'./data/vocab.pkl'' is error
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)