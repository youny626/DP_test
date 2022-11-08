#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 30 19:26:47 2021

@author: lovingmage
"""


import unittest
import resource
import phe.paillier as paillier
import crypte.provision as pro
import crypte.core as cte
from crypte.core import Cdata
from crypte.core import CSP, AS

def test_q1(obj, attr_pred, val_pred_1, val_pred_2):
    encfilter = cte.filter(obj, attr_pred, val_pred_1, val_pred_2)
    #encfilter = x.filter(2,3,4)
    encnt = encfilter.count()
    return encnt
    
def test_q2(obj, attr_1, attr_2, pk):
    cosproduct = cte.cosprod(obj, attr_1, attr_2, pk)
    return cosproduct    
    
def test_q3(obj, attr_pred):
    encnt = obj.group_by(attr_pred)
    return encnt

class TestClient(unittest.TestCase):
    ''' Test of filter query '''
    def test_filter(self):
        data = [1,2,3,7,5,6,7,8,9]
        attr = [2,5,2]
        
        # set up CSP and derive keys
        cs = CSP()
        cs.reg_eps(1.0)
        pk, sk = cs.derive_key()
        
        #set up AS, pk and init cdata
        x = Cdata(attr=[2,5,2])
        x.set_pk(pk)
        c = pro.lab_encrypt_vector(pk, data)
        x.insert(c) 
        a = AS()
        a.set_key(pk)
        a.load_data(x)
        
        # simulate client upload cdata
        a.insert_to_db(c)
        a.insert_to_db(c)
        a.insert_to_db(c)
        a.insert_to_db(c)
        
        
        c = a.execute(test_q1, 2, 3, 4)
        #c = analytics_server.laplace_distort(20, 0.5, 1, c)
        self.assertEqual(55, cs.reveal_clear(c))
        
        
    ''' Test of DP filter query '''
    def test_laplace_filter(self):
        data = [1,2,3,7,5,6,7,8,9]
        attr = [2,5,2]
        
        # set up CSP and derive keys
        cs = CSP()
        cs.reg_eps(1.0)
        pk, sk = cs.derive_key()
        
        #set up AS, pk and init cdata
        x = Cdata(attr=[2,5,2])
        x.set_pk(pk)
        c = pro.lab_encrypt_vector(pk, data)
        x.insert(c) 
        a = AS()
        a.set_key(pk)
        a.load_data(x)
        
        # simulate client upload cdata
        a.insert_to_db(c)
        a.insert_to_db(c)
        a.insert_to_db(c)
        a.insert_to_db(c)
        
        
        c = a.execute(test_q1, 2, 3, 4)
        print("\n")
        #print("True counting output is", cs.reveal_clear(c))
        sens = 1
        eps = 0.05
        c = a.laplace_distort(sens, eps, 1, c)
        print("DP counting output is", cs.reveal_noisy(c, sens, eps))
        #self.assertEqual(55, cs.reveal_clear(c))
        
    ''' Test of group by query '''
    def test_group_by(self):
        data = [1,2,3,7,5,6,7,8,9]
        attr = [2,5,2]
        
        # set up CSP and derive keys
        cs = CSP()
        cs.reg_eps(1.0)
        pk, sk = cs.derive_key()
        
        #set up AS, pk and init cdata
        x = Cdata(attr=[2,5,2])
        x.set_pk(pk)
        c = pro.lab_encrypt_vector(pk, data)
        x.insert(c) 
        a = AS()
        a.set_key(pk)
        a.load_data(x)
        
        # simulate client upload cdata
        a.insert_to_db(c)
        a.insert_to_db(c)
        a.insert_to_db(c)
        a.insert_to_db(c)
        
        
        c = a.execute(test_q3, 2)
        self.assertEqual([15, 35, 25, 30, 35], cs.reveal_clear_vector(c))
        
    ''' Test of group by query '''
    def test_laplace_group_by(self):
        data = [1,2,3,7,5,6,7,8,9]
        attr = [2,5,2]
        
        # set up CSP and derive keys
        cs = CSP()
        cs.reg_eps(1.0)
        pk, sk = cs.derive_key()
        
        #set up AS, pk and init cdata
        x = Cdata(attr=[2,5,2])
        x.set_pk(pk)
        c = pro.lab_encrypt_vector(pk, data)
        x.insert(c) 
        a = AS()
        a.set_key(pk)
        a.load_data(x)
        
        # simulate client upload cdata
        a.insert_to_db(c)
        a.insert_to_db(c)
        a.insert_to_db(c)
        a.insert_to_db(c)
        
        
        c = a.execute(test_q3, 2)
        print("\n")
        #print("True counting output is", cs.reveal_clear(c))
        sens = 1
        eps = 0.05
        c = a.laplace_distort(sens, eps, 5, c)
        print("DP group by output is", cs.reveal_noisy_vector(c, sens, eps, 5))
        #self.assertEqual([15, 35, 25, 30, 35], cs.reveal_clear_vector(c))
    
    ''' Test of generating cros product table'''
    def test_cosprod(self):
        data = [1,2,3,7,5,6,7,8,9]
        attr = [2,5,2]
        
        # set up CSP and derive keys
        cs = CSP()
        cs.reg_eps(1.0)
        pk, sk = cs.derive_key()
        
        #set up AS, pk and init cdata
        x = Cdata(attr=[2,5,2])
        x.set_pk(pk)
        c = pro.lab_encrypt_vector(pk, data)
        x.insert(c) 
        a = AS()
        a.set_key(pk)
        a.load_data(x)
        
        # simulate client upload cdata
        a.insert_to_db(c)
        a.insert_to_db(c)

        #
        c = a.execute(test_q2, 1, 2, pk)
        g_truth = [3, 7, 5, 6, 7, 6, 14, 10, 12, 14]
        #print(cosproduct.get_data()[0][0])
        self.assertEqual(g_truth, cs.reveal_clear_mult_vector(c.get_data()[0]))


    def test(self):
        import numpy as np

        # 1 0 1 0
        # 0 1 0 1
        # 1 1 0 0
        # 0 0 1 1
        # x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        # y = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
        x = [1, 0]
        y = [0, 1]
        print(np.cross(x, y))
        
    ''' Test of multi-dimensional filter query'''
    def test_cosprod_filter(self):
        attr = [2, 2]
        
        # set up CSP and derive keys
        cs = CSP()
        cs.reg_eps(1.0)
        pk, sk = cs.derive_key()
        
        #set up AS, pk and init cdata
        x = Cdata(attr=attr)
        x.set_pk(pk)
        data = [1, 0, 0, 1]
        c = pro.lab_encrypt_vector(pk, data)
        x.insert(c) 
        a = AS()
        a.set_key(pk)
        a.load_data(x)
        
        # simulate client upload cdata
        data = [1, 0, 1, 0]
        c = pro.lab_encrypt_vector(pk, data)
        a.insert_to_db(c)
        # analytics_server.insert_to_db(c)
        print(x.get_data())
        print(len(x.get_data()), len(x.get_data()[0]), len(x.get_data()[0][0]))
            
        c = a.execute(test_q2, 1, 2, pk)
        # g_truth = [1, ]
        #print(cosproduct.get_data()[0][0])
        # print(len(c.get_data()))
        # print(len(c.get_data()[0]))
        self.assertEqual([0, 1, 0, 0], cs.reveal_clear_mult_vector(c.get_data()[0]))
        self.assertEqual([1, 0, 0, 0], cs.reveal_clear_mult_vector(c.get_data()[1]))

        # attr1 = 2, attr2 = 1
        # so the index (1-based) to filter for the cross product is 2
        print(c.get_attr())
        print(c.get_data())
        c.set_data(cs.re_encrypt_mult(c.get_data()))
        print(len(c.get_data()), len(c.get_data()[0]), len(c.get_data()[0][0]))


        a = AS()
        a.set_key(pk)
        a.load_data(c) # load_data will change the current cdata to the cdata to load

        encrypted_count = a.execute(test_q1, 1, 1, 2)
        # c.set_data(c.get_data()[0])
        # encfilter = cte.filter(c, 1, 2, 2)
        # encrypted_count = encfilter.count()
        print(encrypted_count)
        print("True counting output is", cs.reveal_clear(encrypted_count))


        sens = 1
        eps = 1
        c = a.laplace_distort(sens, eps, 1, encrypted_count)
        print("DP counting output is", cs.reveal_noisy(c, sens, eps))

        # print(len(c.get_data()))
        
        
if __name__ == '__main__':
    unittest.main()
