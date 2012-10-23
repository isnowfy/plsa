# -*- coding: utf-8 -*-

import math
import operator
import random

def cos_sim(p, q):
    sum0 = sum(map(lambda x:x*x, p))
    sum1 = sum(map(lambda x:x*x, q))
    sum2 = sum(map(lambda x:x[0]*x[1], zip(p, q)))
    return sum2/(sum0**0.5)/(sum1**0.5)

def _rand_mat(sizex, sizey):
    ret = []
    for i in xrange(sizex):
        ret.append([])
        for _ in xrange(sizey):
            ret[-1].append(random.random())
        norm = sum(ret[-1])
        for j in xrange(sizey):
            ret[-1][j] /= norm
    return ret

class Plsa:

    def __init__(self, corpus, topics=2):
        self.topics = topics
        self.corpus = corpus
        self.docs = len(corpus)
        self.each = map(sum, map(lambda x:x.values(), corpus))
        self.words = max(reduce(operator.add, map(lambda x:x.keys(), corpus)))+1
        self.likelihood = 0
        self.zw = _rand_mat(self.topics, self.words)
        self.dz = _rand_mat(self.docs, self.topics)
        self.dw_z = None
        self.beta = 0.8
        self._cal_p_dw()

    def _cal_p_dw(self):
        self.p_dw = []
        for d in xrange(self.docs):
            self.p_dw.append({})
            for w in self.corpus[d]:
                tmp = 0
                for z in xrange(self.topics):
                    tmp += (self.zw[z][w]*self.dz[d][z])**self.beta
                self.p_dw[-1][w] = tmp

    def _e_step(self):
        self.dw_z = []
        for d in xrange(self.docs):
            self.dw_z.append({})
            for w in self.corpus[d]:
                self.dw_z[-1][w] = []
                for z in xrange(self.topics):
                    self.dw_z[-1][w].append(((self.zw[z][w]*self.dz[d][z])**self.beta)/self.p_dw[d][w])

    def _m_step(self):
        for z in xrange(self.topics):
            self.zw[z] = [0]*self.words
            for d in xrange(self.docs):
                for w in self.corpus[d]:
                    self.zw[z][w] += self.corpus[d][w]*self.dw_z[d][w][z]
            norm = sum(self.zw[z])
            for w in xrange(self.words):
                self.zw[z][w] /= norm
        for d in xrange(self.docs):
            self.dz[d] = [0]*self.topics
            for z in xrange(self.topics):
                for w in self.corpus[d]:
                    self.dz[d][z] += self.corpus[d][w]*self.dw_z[d][w][z]
            for z in xrange(self.topics):
                self.dz[d][z] /= self.each[d]
        self._cal_p_dw()

    def _cal_likelihood(self):
        self.likelihood = 0
        for d in xrange(self.docs):
            for w in self.corpus[d]:
                self.likelihood += self.corpus[d][w]*math.log(self.p_dw[d][w])

    def train(self, max_iter=100):
        cur = 0
        for i in xrange(max_iter):
            print '%d iter' % i
            self._e_step()
            self._m_step()
            self._cal_likelihood()
            print 'likelihood %f ' % self.likelihood
            if cur != 0 and abs((self.likelihood-cur)/cur) < 1e-8:
                break
            cur = self.likelihood

    def inference(self, doc, max_iter=100):
        doc = dict(filter(lambda x:x[0]<self.words, doc.items()))
        words = sum(doc.values())
        ret = []
        for i in xrange(self.topics):
            ret.append(random.random())
        norm = sum(ret)
        for i in xrange(self.topics):
            ret[i] /= norm
        p_dw = {}
        for w in doc:
            p_dw[w] = 0
            for z in xrange(self.topics):
                p_dw[w] += ret[z]*self.zw[z][w]
        tmp = 0
        for _ in xrange(max_iter):
            # e setp
            dw_z = {}
            for w in doc:
                dw_z[w] = []
                for z in xrange(self.topics):
                    dw_z[w].append(self.zw[z][w]*ret[z]/p_dw[w])
            # m step
            ret = [0]*self.topics
            for z in xrange(self.topics):
                for w in doc:
                    ret[z] += doc[w]*dw_z[w][z]
            for z in xrange(self.topics):
                ret[z] /= words
            # cal likelihood
            p_dw = {}
            for w in doc:
                p_dw[w] = 0
                for z in xrange(self.topics):
                    p_dw[w] += ret[z]*self.zw[z][w]
            likelihood = 0
            for w in doc:
                likelihood += doc[w]*math.log(p_dw[w])
            if tmp != 0 and abs((likelihood-tmp)/tmp) < 1e-8:
                break
            tmp = likelihood
        return ret

    def post_prob_sim(self, docd, q):
        sim = 0
        for w in docd:
            tmp = 0
            for z in xrange(self.topics):
                tmp += self.zw[z][w]*q[z]
            sim += docd[w]*math.log(tmp)
        return sim

######### unittest #################################
import unittest

class TestPlsa(unittest.TestCase):

    def test_train(self):
        corpus = [{0:2,3:5},{0:5,2:1},{1:2,4:5}]
        p = Plsa(corpus)
        p.train()
        self.assertTrue(cos_sim(p.dz[0], p.dz[1])>cos_sim(p.dz[0], p.dz[2]))
        self.assertTrue(p.post_prob_sim(p.corpus[0], p.dz[1])>p.post_prob_sim(p.corpus[0], p.dz[2]))

    def test_inference(self):
        corpus = [{0:2,3:5},{0:5,2:1},{1:2,4:5}]
        p = Plsa(corpus)
        p.train()
        z = p.inference({0:4, 6:7})
        self.assertTrue(abs(cos_sim(p.dz[0], p.dz[1])-cos_sim(p.dz[0], z))<1e-8)

if __name__ == '__main__':
    unittest.main()
