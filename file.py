# -*- coding: utf-8 -*-

#1
def list():
    list = ["A","B","C","D"] #Defines the list, uses square brackets as it's a list
    for letter in list: #Simple for loop
        print (letter) #Print the thing that I specified in previous line
#2
def age():
    a1 = input("Input the age of User 1: ") #Gaining an input from a user
    a2 = input("Input the age of User 2: ") #Same as above
    if a1 > a2: #Cheeky if loop, probably could be done cleaner at some point in the future
        print(a1)
    else:
        print(a2)


#2.5
def age1(a1, a2):
    if a1 > a2: #Cheeky if loop, probably could be done cleaner at some point in the future
        print(a1)
    elif a1 == a2:
        print("They are the same age")
    else:
        print(a2)
        
        
#3.1
def bcks():
    string = input("Input a word: ")
    back = ""
    back = back + string[-1]
    print(back)
    length = len(string)
    print (length)
    
    
#3.5    
def bcks1():
    string = input("Input a word: ")
    back = ""
    for letter in string:
        back = back + string[-1]
        string = string[:-1]
    print(back)
    
    
#4
def val():
    try:
        n = int(input("Input a number between 5 and 10: "))
        if n <= 10 and n >= 5:
            print("Well done")
        else:
            print("Not in range")
    except:
        print("Wrong input")
        

        
        
        
        
        
        
    
    
    

        
