import pickle

class MyEncoderDecoder:
    def __init__(self):
        self.encoder = pickle.load(open(".\\data\\encode.sav", 'rb'))
        self.decoder = pickle.load(open(".\\data\\decode.sav", 'rb'))

    def encode(self, string):
        ret = []
        for i in range(len(string)):
            ret.append(self.encoder[string[i]])

        return ret

    def decode(self, array):
        string = ""

        for el in array:
            string = string + str(self.decoder[str(int(el))])

        return string
        

def create_dictionary():
    #d = pickle.load(open(".\\disct.sav", 'rb'))
    #print(d)
    #exit(0)
    
    file = open('pantadeusz.txt', 'r', encoding="utf-8")
    first_dict = {"k": 0}
    second_dict = {"0": "k"}
    i = 1
    while 1:
        char = file.read(1).lower()

        if char == "\n":
            char = " "
            
        if not char:
            pickle.dump(first_dict, open(".\\encode.sav", 'wb'))
            print(first_dict)
            pickle.dump(second_dict, open(".\\decode.sav", 'wb'))
            print(second_dict)
            break
        
        if char not in first_dict:
            first_dict.update({char: i})
            second_dict.update({str(i): char})
            i = i + 1
            

