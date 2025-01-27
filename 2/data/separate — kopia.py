import json

file = open('pantadeusz.txt', 'r', encoding="utf-8")
train_out = open('train_data.txt', 'w', encoding="utf-8")
test_out = open('test_data.txt', 'w', encoding="utf-8")
val_out = open('val_data.txt', 'w', encoding="utf-8")

string = ""
correct_string = ""
full_letters = 445641
train_size = int(445641 * 6 / 10)
test_size = int(445641 * 2 / 10)

for i in range(train_size):
    char = file.read(1).lower()
    if not char == "\n":
        train_out.write(char)
    else:
        train_out.write(" ")

for i in range(test_size):
    char = file.read(1).lower()
    if not char == "\n":
        test_out.write(char)
    else:
        test_out.write(" ")

while True:
    char = file.read(1).lower()
    if not char: 
        break
    if not char == "\n":
        val_out.write(char)
    else:
        val_out.write(" ")

file.close()
train_out.close()
test_out.close()
val_out.close()

print("END")
