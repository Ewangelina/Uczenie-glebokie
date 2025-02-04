import json

file = open('pantadeusz.txt', 'r', encoding="utf-8")
output = open('pairs_single.csv', 'w', encoding="utf-8")

string = ""
correct_string = ""


for i in range(99):    
    char = file.read(1).lower()
    if char == "\n":
        char = " "
        
    string = string + char

while 1:
    char = file.read(1).lower()        
    if not char: 
        break
    if char == "\n":
        char = " "

    correct_string = char

    out = string + "*" + correct_string + "\n"
    output.write(out)
    output.close()
    output = open('pairs_single.csv', 'a', encoding="utf-8")
    string = string[1:] + correct_string
    
    

file.close()
output.close()

print("END")
