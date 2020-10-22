# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:02:02 2019

@author: Isaac
"""

#1
def rps():
    a1 = input("Player 1 input your choice: ")
    a2 = input("Player 2 input your choice: ")
    if a1 == "rock" and a2 == "paper":
        print("Player 1 wins!")
    if a1 == "rock" and a2 == "scissors":
        print("Player 2 wins!")
        
        
#1.1
def rps1():
    "rock" > "scissors"
    "scissors" > "paper"
    "paper" > "rock"
    a1 = input("Player 1 input your choice: ")
    a2 = input("Player 2 input your choice: ")
    if a1 > a2 :
        print("\nPlayer 1 wins")
    else: 
        print("\nPlayer 2 wins")
        
#2
def cc():
    cyp = input(str(("Input the message you wish to encode: "))
    chr("97")