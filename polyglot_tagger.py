from __future__ import unicode_literals

import io
import re
import sys
import math
import string
import random
import pickle
from argparse import ArgumentParser
from collections import Counter, defaultdict

import dynet as dy
import numpy as np
from gensim.models.word2vec import KeyedVectors

class Meta:
    def __init__(self):
        self.c_dim = 32  # character-rnn input dimension
        self.add_words = 1  # additional lookup for missing/special words
        self.n_hidden = 64  # pos-mlp hidden layer dimension
        self.lstm_char_dim = 64  # char-LSTM output dimension
        self.lstm_word_dim = 128  # LSTM (word-char concatenated input) output dimension


class POSTagger():
    def __init__(self, model=None, meta=None):
        self.model = dy.Model()
        if model:
            self.meta = pickle.load(open('%s.meta' %model, 'rb'))
        else:
            self.meta = meta
        self.EWORDS_LOOKUP = self.model.add_lookup_parameters((self.meta.n_words_eng, self.meta.w_dim_eng))
        self.HWORDS_LOOKUP = self.model.add_lookup_parameters((self.meta.n_words_hin, self.meta.w_dim_hin))
        if not model:
            for word, V in ewvm.vocab.iteritems():
                self.EWORDS_LOOKUP.init_row(V.index+self.meta.add_words, ewvm.syn0[V.index])
            for word, V in hwvm.vocab.iteritems():
                self.HWORDS_LOOKUP.init_row(V.index+self.meta.add_words, hwvm.syn0[V.index])

        self.CHARS_LOOKUP = self.model.add_lookup_parameters((self.meta.n_chars, self.meta.c_dim))

        # MLP on top of biLSTM outputs 100 -> 32 -> ntags
        self.W1 = self.model.add_parameters((self.meta.n_hidden, self.meta.lstm_word_dim*2))
        self.W2 = self.model.add_parameters((self.meta.n_tags, self.meta.n_hidden))
        self.B1 = self.model.add_parameters(self.meta.n_hidden)
        self.B2 = self.model.add_parameters(self.meta.n_tags)

        # word-level LSTMs
        self.fwdRNN = dy.LSTMBuilder(1, self.meta.w_dim_eng, self.meta.lstm_word_dim, self.model) 
        self.bwdRNN = dy.LSTMBuilder(1, self.meta.w_dim_eng, self.meta.lstm_word_dim, self.model)

        # char-level LSTMs
        self.cfwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.w_dim_eng/2, self.model)
        self.cbwdRNN = dy.LSTMBuilder(1, self.meta.c_dim, self.meta.w_dim_eng/2, self.model)
        if model:
            self.model.populate('%s.dy' %model)

    def word_rep(self, word, f, b, lang='en'):
        if not self.eval and random.random() < 0.25:
            return self.HWORDS_LOOKUP[0] if lang=='hi' else self.EWORDS_LOOKUP[0]
        if lang == 'hi':
            idx = self.meta.hw2i.get(word.lower(), 0)
            if not idx:
                return self.char_rep(word, f, b)
            return self.HWORDS_LOOKUP[idx]
        elif lang == 'en':
            idx = self.meta.ew2i.get(word, self.meta.ew2i.get(word.lower(), 0))
            if not idx:
                return self.char_rep(word, f, b)
            return self.EWORDS_LOOKUP[idx]
    
    def char_rep(self, word, f, b):
        no_c_drop = False
        if self.eval or random.random()<0.9:
            no_c_drop = True
        bos, eos, unk = self.meta.c2i["bos"], self.meta.c2i["eos"], self.meta.c2i["unk"]
        char_ids = [bos] + [self.meta.c2i.get(c, unk) if no_c_drop else unk for c in w] + [eos]
        char_embs = [self.CHARS_LOOKUP[cid] for cid in char_ids]
        fw_exps = f.transduce(char_embs)
        bw_exps = b.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

    def enable_dropout(self):
        self.fwdRNN.set_dropout(0.3)
        self.bwdRNN.set_dropout(0.3)
        self.cfwdRNN.set_dropout(0.3)
        self.cbwdRNN.set_dropout(0.3)
        self.w1 = dy.dropout(self.w1, 0.3)
        self.b1 = dy.dropout(self.b1, 0.3)

    def disable_dropout(self):
        self.fwdRNN.disable_dropout()
        self.bwdRNN.disable_dropout()
        self.cfwdRNN.disable_dropout()
        self.cbwdRNN.disable_dropout()

    def build_tagging_graph(self, words, ltags):
        # parameters -> expressions
        self.w1 = dy.parameter(self.W1)
        self.b1 = dy.parameter(self.B1)
        self.w2 = dy.parameter(self.W2)
        self.b2 = dy.parameter(self.B2)

        # apply dropout
        if self.eval:
            self.disable_dropout()
        else:
            self.enable_dropout()

        # initialize the RNNs
        f_init = self.fwdRNN.initial_state()
        b_init = self.bwdRNN.initial_state()
    
        cf_init = self.cfwdRNN.initial_state()
        cb_init = self.cbwdRNN.initial_state()

        # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
        wembs = [self.word_rep(w, cf_init, cb_init, l) for w,l in zip(words, ltags)]
        #if not self.eval:
        #    wembs = [dy.block_dropout(x, 0.25) for x in wembs]
    
        # feed word vectors into biLSTM
        fw_exps = f_init.transduce(wembs)
        bw_exps = b_init.transduce(reversed(wembs))
    
        # biLSTM states
        bi_exps = [dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]
    
        # feed each biLSTM state to an MLP
        exps = []
        for xi in bi_exps:
            xh = self.w1 * xi
            xo = self.w2 * (self.meta.activation(xh) + self.b1) + self.b2
            exps.append(xo)
        return exps
    
    def sent_loss(self, words, tags, ltags):
        self.eval = False
        vecs = self.build_tagging_graph(words, ltags)
        for v,t in zip(vecs,tags):
            tid = self.meta.t2i[t]
            err = dy.pickneglogsoftmax(v, tid)
            self.loss.append(err)
    
    def tag_sent(self, words, ltags):
        self.eval = True
        dy.renew_cg()
        vecs = self.build_tagging_graph(words, ltags)
        vecs = [dy.softmax(v) for v in vecs]
        probs = [v.npvalue() for v in vecs]
        tags = []
        for prb in probs:
            tag = np.argmax(prb)
            tags.append(self.meta.i2t[tag])
        return zip(words, tags)

def read(fname, lang=None):
    data = []
    sent = []
    pid = 3 if args.ud else 4
    wid = 1 if lang =='dev' and not args.norm else 2
    fp = io.open(fname, encoding='utf-8')
    for i,line in enumerate(fp):
        line = line.split()
        if not line:
            data.append(sent)
            sent = []
        else:
            if lang == 'dev':
                try:
                    w,p,l = line
                except ValueError:
                    try:
                        w,p,l = line[wid], line[pid], line[8]
                    except Exception:
                        sys.stderr.write('Wrong file format\n')
                        sys.exit(1)
                l = l.split('|')[0]
                l = 'hi' if l=='hi' else 'en'
            else:
                l = 'hi' if lang == 'hi' else 'en'
                try:
                    w,p = line
                except ValueError:
                    try:
                        w,p = line[1], line[pid]
                    except Exception:
                        sys.stderr.write('Wrong file format\n')
                        sys.exit(1)
            sent.append((w,p,l))
    if sent: data.append(sent)
    return data

def eval(dev, ofp=None):
    good_sent = bad_sent = good = bad = 0.0
    gall, pall = [], []
    for sent in dev:
        words, golds, ltags = zip(*sent)
        tags = [t for w,t in tagger.tag_sent(words, ltags)]
        #pall.extend(tags)
        if list(tags) == list(golds): good_sent += 1
        else: bad_sent += 1
        for go,gu in zip(golds,tags):
            if go == gu: good += 1
            else: bad += 1
    #print(cr(gall, pall, digits=4))
    print(good/(good+bad), good_sent/(good_sent+bad_sent))
    return good/(good+bad)

def train_tagger(train):
    pr_acc = 0.0
    n_samples = len(train)
    num_tagged, cum_loss = 0, 0
    for ITER in xrange(args.iter):
        dy.renew_cg()
        tagger.loss = []
        random.shuffle(train)
        for i,s in enumerate(train, 1):
            if i % 500 == 0 or i == n_samples:   # print status
                trainer.status()
                print(cum_loss / num_tagged)
                cum_loss, num_tagged = 0, 0
            words, golds, ltags = zip(*s)
            tagger.sent_loss(words, golds, ltags)
            num_tagged += len(golds)
            if len(tagger.loss) > 50:
                batch_loss = dy.esum(tagger.loss)
                cum_loss += batch_loss.scalar_value()
	        batch_loss.backward()
                trainer.update()
                tagger.loss = []
                dy.renew_cg()
        if tagger.loss:
            batch_loss = dy.esum(tagger.loss)
            cum_loss += batch_loss.scalar_value()
	    batch_loss.backward()
            trainer.update()
            tagger.loss = []
            dy.renew_cg()
        print("epoch %r finished" % ITER)
        sys.stdout.flush()
        cum_acc = 0.0
        if args.edev:
            cum_acc += eval(edev)
        if args.hdev:
            cum_acc += eval(hdev)
        if args.cdev:
            cum_acc = eval(cdev)
        if cum_acc > pr_acc:
            pr_acc = cum_acc
            print('Save Point:: %d' %ITER)
            if args.save_model:
                tagger.model.save('%s.dy' %args.save_model)


def get_char_map(data):
    tags = set()
    meta.c2i = {'bos':0, 'eos':1, 'unk':2}
    cid = len(meta.c2i)
    for sent in data:
        for w,p,l in sent:
            tags.add(p)
            for c in w:
                if not meta.c2i.has_key(c):
                    meta.c2i[c] = cid
                    cid += 1
    meta.n_chars = len(meta.c2i)
    meta.n_tags = len(tags)
    meta.i2t = dict(enumerate(tags))
    meta.t2i = {t:i for i,t in meta.i2t.items()}

if __name__ == '__main__':
    parser = ArgumentParser(description="POS Tagger")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--dynet-gpu')
    parser.add_argument('--dynet-mem')
    parser.add_argument('--dynet-devices')
    parser.add_argument('--dynet-autobatch')
    parser.add_argument('--dynet-seed', dest='seed', type=int, default='127')
    parser.add_argument('--etrain', help='English CONLL/TNT Train file')
    parser.add_argument('--edev', help='English CONLL/TNT Dev/Test file')
    parser.add_argument('--htrain', help='Hindi CONLL/TNT Train file')
    parser.add_argument('--hdev', help='Hindi CONLL/TNT Dev/Test file')
    #parser.add_argument('--ctrain', help='Hindi-English CS CONLL/TNT Train file')
    parser.add_argument('--cdev', help='Hindi-English CS CONLL/TNT Dev/Test file')
    parser.add_argument('--hi-embds', dest='hembd', help='Pretrained Hindi word2vec Embeddings')
    parser.add_argument('--hi-limit', dest='hlimit', type=int, default=None,
                        help='load only top-n pretrained Hindi word vectors (default=all vectors)')
    parser.add_argument('--en-embds', dest='eembd', help='Pretrained English word2vec Embeddings')
    parser.add_argument('--en-limit', dest='elimit', type=int, default=None,
                        help='load only top-n pretrained English word vectors (default=all vectors)')
    parser.add_argument('--trainer', default='momsgd', help='Trainer [momsgd|adam|adadelta|adagrad]')
    parser.add_argument('--activation-fn', dest='act_fn', default='tanh', help='Activation function [tanh|rectify|logistic]')
    parser.add_argument('--ud', type=int, default=1, help='1 if UD treebank else 0')
    parser.add_argument('--iter', type=int, default=100, help='No. of Epochs')
    parser.add_argument('--bvec', type=int, help='1 if binary embedding file else 0')
    parser.add_argument('--norm', action='store_true', help='set if testing on normalized word forms (3rd column in UD format)')
    group.add_argument('--save-model', dest='save_model', help='Specify path to save model')
    group.add_argument('--load-model', dest='load_model', help='Load Pretrained Model')
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    meta = Meta()
    if args.edev:
        edev = read(args.edev, lang='en')
    if args.hdev:
        hdev = read(args.hdev, lang='hi')
    if args.cdev:
        cdev = read(args.cdev, 'dev')
    if not args.load_model: 
        train_e = read(args.etrain, 'en')
        train_h = read(args.htrain, 'hi')
        ewvm = KeyedVectors.load_word2vec_format(args.eembd, binary=args.bvec, limit=args.elimit)
        hwvm = KeyedVectors.load_word2vec_format(args.hembd, binary=args.bvec, limit=args.hlimit)
        meta.w_dim_eng = ewvm.syn0.shape[1]
        meta.n_words_eng = ewvm.syn0.shape[0]+meta.add_words
        meta.w_dim_hin = hwvm.syn0.shape[1]
        meta.n_words_hin = hwvm.syn0.shape[0]+meta.add_words
        
        get_char_map(train_e+train_h)
        meta.ew2i = {}
        for w in ewvm.vocab:
            meta.ew2i[w] = ewvm.vocab[w].index + meta.add_words
        meta.hw2i = {}
        for w in hwvm.vocab:
            meta.hw2i[w] = hwvm.vocab[w].index + meta.add_words
    
        trainers = {
            'momsgd'   : dy.MomentumSGDTrainer,
            'adam'     : dy.AdamTrainer,
            'simsgd'   : dy.SimpleSGDTrainer,
            'adagrad'  : dy.AdagradTrainer,
            'adadelta' : dy.AdadeltaTrainer
            }
        act_fn = {
            'sigmoid' : dy.logistic,
            'tanh'    : dy.tanh,
            'relu'    : dy.rectify,
            }
        meta.trainer = trainers[args.trainer]
        meta.activation = act_fn[args.act_fn] 

    if args.save_model:
        pickle.dump(meta, open('%s.meta' %args.save_model, 'wb'))
    if args.load_model:
        tagger = POSTagger(model=args.load_model)
        if args.cdev:
            eval(cdev) 
        if args.edev:
            eval(edev) 
        if args.hdev:
            eval(hdev) 
    else:
        tagger = POSTagger(meta=meta)
        trainer = meta.trainer(tagger.model)
        train_tagger(train_e+train_h)
