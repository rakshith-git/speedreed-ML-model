import simple_model
import json 

with open(r'tr3.json', 'r') as file:
    data = json.load(file)
    
sents = []
multipliers = []
for i in range(0, len(data)):
    sents.append(data[i]["sentence"])
    multipliers.append(data[i]["multipliers"])
    
simple_model.train_and_test(sents, multipliers)
    