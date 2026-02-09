import json

def load_data(filepath=r'C:\Users\Anna\Desktop\MLA_100k.jsonlines'):
    listings = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            listings.append(json.loads(line))
    return listings

