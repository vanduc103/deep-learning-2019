import numpy as np
from six.moves import urllib
import os, sys, tarfile
import pandas as pd
import pickle 
import nltk
from PIL import Image
import random
from miscc.config import cfg

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict


def unzip(src_dir, data_dir):
    """
    Unzips the downloaded 'CUB_200_2011' dataset and move its location to data/birds/CUB_200_2011.
    
    Inputs:
    - src_dir: directory of zipped 'CUB_200_2011' dataset
    - data_dir: base directory of 'CUB_200_2011' dataset, which is data/
    """
    try:
        if src_dir.endswith('.zip'):
            print('unzipping ' + src_dir)
            with zipfile.ZipFile(src_dir) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dirpath)
        elif src_dir.endswith('.tgz') or src_dir.endswith('tar.gz'):
            print('unzipping ' + src_dir)
            tar = tarfile.open(src_dir)
            tar.extractall()
            tar.close()
        os.rename('CUB_200_2011', os.path.join(data_dir, 'CUB_200_2011'))
    except:
        raise('wrong format')
        
def download(url, dirpath):
    """
    Downloads 'CUB_200_2011.tgz' file to 'dirpath'.

    Inputs:
    - url: url of 'CUB_200_2011.tgz'
    - dirpath: directory of saving the downloaded tgz file
    """
    filepath = dirpath
    u = urllib.request.urlopen(url)
    f = open(filepath, 'wb')
    filesize = int(u.headers["Content-Length"])
    
    print("Downloading: %s Bytes: %s" % ("CUB-200-2011 (birds images)", filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)

        status = (("[{}  " + " ***progress: {:03.1f}% ]").format('=' * int(float(downloaded) / 
            filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status, end='')
        
        sys.stdout.flush()
    f.close()

class CUBDataset():
    def __init__(self, data_dir, split='train', imsize=256):
        url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
        
        self.imsize = imsize
        self.split = split
        
        self.current_dir = os.getcwd()
        print(f'self.current_dir:\n{self.current_dir}\n')
        self.data_dir = os.path.join(self.current_dir, data_dir)
        print(f'self.data_dir:\n{self.data_dir}\n')
        
        self.image_dir = os.path.join(self.data_dir, url.split('/')[-1])
        print(f'self.image_dir:\n{self.image_dir}\n')
        if os.path.exists(self.image_dir):
           print(f'Dataset already exists')
        else:
            download(url, self.image_dir)
            unzip(self.image_dir, self.data_dir)
        
        self.image_dir = os.path.join(self.image_dir.split('.tgz')[0], 'images')
        print(f'self.image_dir:\n{self.image_dir}\n')

        
        self.bbox = self.load_bbox()
        
        self.split_dir = os.path.join(self.data_dir, split)
        
        self.filenames, self.captions, self.captions_ids, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(data_dir, split)
        self.class_id = self.load_class_id(self.split_dir)
        
        self.images = self.generate_bboxed_image(self.split_dir, self.filenames)
            
    def load_bbox(self):
        """
        Loads the corresponding bounded box of each bird image, which is saved in a txt format.
        
        Outputs:
        - filename_bbox: directionary, e.g., {'001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111': [x-left, y-top, width, height], .....}
        """
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)

        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()

        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox

        return filename_bbox
    
    def load_filenames(self, data_dir, split):
        """
        Loads assigned filenames for 'train' and 'test'.
        
        Inputs:
        - data_dir: base directory of dataset. ('data/birds')
        - split: either 'train' or 'test'

        Outputs:
        - filenames: numpy array, e.g., ['001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111', '001.Black_footed_Albatross/Black_Footed_Albatross_0002_55', ....]
        """
        filepath = os.path.join(data_dir, split, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f, encoding='latin1')
        #print(f'Load filenames:\t{filepath}')
        return np.asarray(filenames)
    
    def load_class_id(self, data_dir):
        """
        Loads class ids of each image file.
        
        Inputs:
        - data_dir: directroy of either 'train' or 'test' dataset. ('data/birds/train' or 'data/birds/test')
        
        Outputs:
        - class_id: list, e.g., [1, 1, 1, 1, 1, ..... 2, 2, 2, ....] 
          1 means '001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111', 2 means '002.Laysan_Albatross/Laysan_Albatross_0001_545'
        """
        with open(data_dir + '/class_info.pickle', 'rb') as f:
            class_id = pickle.load(f, encoding='latin1')
        return class_id
   
    def load_captions(self, data_dir, filenames):
        """
        Loads 10 captions for each image of either 'train' or 'test' dataset and tokenizes each caption as words.

        Inputs:
        - data_dir: base directory of dataset ('data/birds')
        - filenames: filenames of either 'train' or 'test' dataset

        Outputs:
        - all_captions: list
        """
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text_c10/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        #print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == cfg.TEXT.CAPTIONS_PER_IMAGE:
                        break
                if cnt < cfg.TEXT.CAPTIONS_PER_IMAGE:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions
    
    def random_wrong_captions(self):
        # generate randomly selected captions for r-precision evaluation
        # size: [len(self.captions_ids) * cfg.WRONG_CAPTION, cfg.TEXT.WORDS_NUM]
        captions_ids_wrong = np.zeros(
            (len(self.captions_ids) * cfg.WRONG_CAPTION, cfg.TEXT.WORDS_NUM), dtype=int)

        for i in range(len(self.images) * cfg.TEXT.CAPTIONS_PER_IMAGE * cfg.WRONG_CAPTION):
            captions_ids_wrong[i] = self.captions_ids[random.randint(0, len(self.captions_ids) - 1)]

        return captions_ids_wrong
    
    def get_caption(self, word_ids):
        # a list of indices for a sentence
        sent_caption = np.asarray(word_ids).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def build_dictionary(self, train_captions, test_captions):
        """
        Loads both train and test captions and generate vocab. Based on the vocab, assigns id to each word.

        Inputs:
        - train_captions: list
        - test_captions: list

        Outputs:
        - train_captions_new: list, Ids of tokenized words of train captions
        - test_captions_new: list, Ids of tokenized words of test captions
        - ixtoword: dictionary, transforms id to word
        - wordtoix: dictionary, transforms word to id
        - len(ixtoword): length of vocab
        """
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            x, x_len = self.get_caption(rev)
            train_captions_new.append(np.squeeze(x, axis=1))

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            x, x_len = self.get_caption(rev)
            test_captions_new.append(np.squeeze(x, axis=1))

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]
    
    def load_text_data(self, data_dir, split):
        """
        Computes the backward pass for a convolutional layer.

        Inputs:
        - data_dir: base directory of dataset ('data/birds')
        - split: either 'train' or 'test'

        Outputs:
        - filenames: list, filenames of either 'train' or 'test' datset depending on 'split'
        - captions: list, tokenized words of either 'train' or 'test' datset
        - captions_ids: list, ids of tokenized words of either 'train' or 'test' datset
        - ixtoword: dictionary, id to word based on generated vocab
        - wordtoix: dictionary, word to id based on generated vocab
        - n_words: scalar, length of generated vocab
        """
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions_ids, test_captions_ids, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions_ids, test_captions_ids, train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions_ids, test_captions_ids = x[0], x[1]
                train_captions, test_captions = x[2], x[3]
                ixtoword, wordtoix = x[4], x[5]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            captions_ids = train_captions_ids
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            captions_ids = test_captions_ids
            filenames = test_names
        return filenames, captions, captions_ids, ixtoword, wordtoix, n_words

    
    def generate_bboxed_image(self, save_dir, filenames):
        """
        Generates bounded-boxed images 

        Inputs:
        - save_dir: directory of saving pre-processed images
        - filenames: filenames of image

        Outputs:
        - bboxed_images: np.array
        """
        bboxed_images = []
        for filename in filenames:
            bboxed_images.append(self.get_imgs(os.path.join(self.image_dir, filename+'.jpg'), self.imsize, self.bbox[filename]))
        bboxed_images = np.asarray(bboxed_images)
        return bboxed_images
    
    def get_imgs(self, img_path, imsize, bbox=None):
        """
        Crops and normalizes given image using corresponding bounded box information

        Inputs:
        - img_path: path of image
        - imsize: resizing size
        - bbox: corresponding bounded box

        Outputs:
        - img: np.array
        """
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])
            img = img.resize((imsize, imsize))

            img = np.asarray(img)
            img = img / (255. / 2.)
            img = img - 1.

        return img
