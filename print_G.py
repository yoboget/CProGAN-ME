# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:05:26 2019

@author: yboge
"""

import pickle



# Import official CelebA-HQ networks.
with open('karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)
print(G)