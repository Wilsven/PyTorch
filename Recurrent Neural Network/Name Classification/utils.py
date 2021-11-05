# data: https://download.pytorch.org/tutorial/data.zip
import io
import os
import unicodedata
import string
import glob

import torch
import random

# alphabets (lower) + alphabets (upper) + " ,.;'"
ALL_LETTERS = string.ascii_letters + " ,.;'"
N_LETTERS = len(ALL_LETTERS)

# Turn a Unicode string to plain ASCII, source: https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) 
        if unicodedata.category(c) != 'Mn' 
        and c in ALL_LETTERS
    )

def load_data():
    # Build category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []
    
    def find_path(path):
        return glob.glob(path)
        
    # Read file and split into lines
    def read_lines(filename):
        # Open the file, strip the text, remove newline
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]
    
    for filename in find_path('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0] # get the language category
        all_categories.append(category)
        
        lines = read_lines(filename)
        category_lines[category] = lines
        
    return category_lines, all_categories

"""
To represent a single letter, we use a “one-hot vector” of 
size <1 x N_LETTERS>. A one-hot vector is filled with 0s
except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.
To make a word we join a bunch of those into a
2D matrix <line_length x 1 x N_LETTERS>.
That extra 1 dimension is because PyTorch assumes
everything is in batches - we’re just using a batch size of 1 here.
"""

# Find letter index from ALL_LETTERS, e.g. "a" = 0
def letter_to_index(letter):
    return ALL_LETTERS.find(letter) 

# For demonstration, turn a letter into a <1 x N_LETTERS> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS) # size of (1, N_LETTERS)
    index = letter_to_index(letter)
    tensor[0][index] = 1
    return tensor


# Same logic and concept as previous function
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        index = letter_to_index(letter)
        tensor[i][0][index] = 1
    return tensor


def random_training_example(category_lines, all_categories):
    
    def random_choice(a):
        random_idx = random.randint(0, len(a)-1)
        return a[random_idx]
    
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def main():
    print(ALL_LETTERS)
    print(unicode_to_ascii('Ślusàrski'))
    
    category_lines, all_categories = load_data()
    print(category_lines['Italian'][:5])
    
    print(letter_to_tensor('J')) # [1, 57]
    print(line_to_tensor('Jones').size()) # [5, 1, 57]
    
# if __name__ == '__main__':
#     main()      