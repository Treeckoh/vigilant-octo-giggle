from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
import requests
import json
import copy
import bs4
import re
import os



class GenData:
    def __init__(
        self,
        generation
    ):
        self.STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
        self.generation = generation
        if f'{generation}.json' in os.listdir():
            with open(f'{generation}.json') as json_file:
                self.generation_data = json.load(json_file)
        else:
            self.get_gen_pokemon()
        self.preprocess_gen_pokemon()
    def get_gen_pokemon(self):
        gen_lst = {
            'gen_1':'rb',
            'gen_2':'gs',
            'gen_3':'rs',
            'gen_4':'dp',
            'gen_5':'bw',
            'gen_6':'xy',
            'gen_7':'sm',
            'gen_8':'ss',
            'gen_9':'sv'
        }

        base_url = 'https://www.smogon.com/dex/'

        query_url = base_url + gen_lst[self.generation] + '/' + 'pokemon'

        pokedex_req = requests.get(query_url)

        soup = bs4.BeautifulSoup(pokedex_req.text)

        soup_str = str(soup).replace('\n', '')

        pokemon_lst = re.findall(r'(\{"pokemon":[\s\S]*)\],"formats"', soup_str)

        pokemon_dict = json.loads(pokemon_lst[0] + ']}')

        pokemon_dict['pokemon'] = [pokemon for pokemon in pokemon_dict['pokemon'] if pokemon['isNonstandard']!='CAP']

        special_pokemon = [pokemon for pokemon in pokemon_dict['pokemon'] if not pokemon['oob']]
        dex_pokemon = [pokemon for pokemon in pokemon_dict['pokemon'] if pokemon['oob']]

        dex_pokemon = sorted(dex_pokemon, key=lambda x: x['oob']['dex_number'])
        self.generation_data = {
            'dex_pokemon': dex_pokemon, 
            'special_pokemon': special_pokemon
        }
        with open(f'{self.generation}.json', 'w') as f:
            json.dump(self.generation_data, f)
            
    def preprocess_gen_pokemon(self):
        df = pd.DataFrame(self.generation_data['dex_pokemon'])
        df.drop(labels=['formats', 'isNonstandard', 'oob'], axis=1, inplace=True)

        types_mlb = MultiLabelBinarizer()
        types_mlb.fit(df['types'])
        df2 = pd.DataFrame(data=types_mlb.transform(df['types']), columns=types_mlb.classes_)
        df = pd.concat([df, df2], axis=1)

        abilities_mlb = MultiLabelBinarizer()
        abilities_mlb.fit(df['abilities'])
        df2 = pd.DataFrame(data=abilities_mlb.transform(df['abilities']), columns=abilities_mlb.classes_)
        df = pd.concat([df, df2], axis=1)

        df.set_index('name', inplace=True)

        df.drop(labels=['types', 'abilities'], axis=1, inplace=True)

        max_height, max_weight = max(df['height']), max(df['weight'])
        df['weight'] = df['weight'] / max_weight
        df['height'] = df['height'] / max_height

        total_stats_data = copy.deepcopy(df['hp'])
        total_stats_data.rename('total_stats', inplace=True)
        for stat in self.STATS[1:]:
            total_stats_data += df[stat]
        df = pd.concat([df, pd.DataFrame(total_stats_data)], axis=1)

        for stat in self.STATS:
            df[stat] = df[stat] / df['total_stats']

        stat_max = max(df['total_stats'])
        df['total_stats'] = df['total_stats'] / stat_max

        self.df = df
        self.types_binarizer = types_mlb
        self.abilities_binarizer = abilities_mlb
        self.max_height = max_height
        self.max_weight = max_weight
        self.stat_max = stat_max
    
    def create_train_test(self):
        self.X = self.df.iloc[:, 6:]
        self.y = self.df.iloc[:, :6]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

if __name__ == '__main__':
    test_scrape = GenData(generation='gen_3')
    test_scrape.create_train_test()
    print(f'Shape of training data: {test_scrape.X_train.shape}, shape of testing data: {test_scrape.X_test.shape}')
    