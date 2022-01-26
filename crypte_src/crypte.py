#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:44:58 2021

@author: lovingmage
"""
import numpy as np
import random
import resource
import time
import phe.paillier as paillier
from dataclasses import dataclass
import crypte_src.util
import json
import crypte_src.provision as pro


class Crypte:
    def __init__(self, attr, db=None):
        self.attr = attr
        self.nattr = len(attr)
        self.dblen = sum(attr)
        self.pk = []
        if db == None:
            self.db = []
        else:
            self.db = db

    def set_pk(self, pubkey):
        self.pk = pubkey

    def fn_set_pk(self, pk_fname):
        # TODO: Implement from fnmae
        self.pk = pk_fname

    def insert(self, enc_vec):
        if len(enc_vec) == self.dblen:
            self.db.append(enc_vec)
        else:
            raise ValueError('Record size not match.\n')

    # @param attr_pred - i-th attribute
    # @param val_pred_1 - value range start
    # @param val_pred_2 - value range end.
    # Example
    #       Point query: filter(2,3,3), SELECT * From Tb WHERE 2nd_attr = value-3
    #       Range query: filter(2,1,3), SELECT * FROM Tb WHERE 2nd_attr in (value-1, value-3)
    def filter(self, attr_pred, val_pred_1, val_pred_2):
        if attr_pred > self.nattr:
            raise ValueError('Non existing attrbute')
        else:
            if val_pred_1 > self.attr[attr_pred - 1] or val_pred_2 > self.attr[attr_pred - 1]:
                raise ValueError('Attribute value out of index')
            if val_pred_1 > val_pred_2:
                raise ValueError('Wrong value order')
            if val_pred_1 <= 0 or val_pred_1 <= 0:
                raise ValueError('Wrong attribute value')

        idxbase = sum(self.attr[:attr_pred - 1])
        idx1 = idxbase + val_pred_1 - 1
        idx2 = idxbase + val_pred_2
        # print(idx1, idx2)
        if (val_pred_1 == val_pred_2):
            return [elem[idx1] for elem in self.db]
        # return [elem[idx1:idx2] for elem in self.db]
        else:
            res = []
            for elem in self.db:
                res += elem[idx1:idx2]
            return res

    def count(self, v):
        return pro.lab_sum_vector(v)
            

