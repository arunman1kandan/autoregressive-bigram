"""
Module: bigram_name_generator.py

This script reads a list of names from a text file, computes bigram statistics (i.e., the frequency of two-character sequences),
and uses those statistics to generate new names character by character. It employs PyTorch tensors for efficient counting and sampling.

Usage:
    - Ensure 'names.txt' is in the same directory, containing one name per line.
    - Run the script: python bigram_name_generator.py
    - The script will print a randomly generated name based on learned bigram probabilities.

Author: Arun Manikandan
Date: 2025-05-24
"""

import pprint  # For pretty-printing Python data structures
import matplotlib.pyplot as plt  # For visualization (heatmap)
import torch  # PyTorch for tensor operations and sampling

# Step 1: Load the dataset of names
# ---------------------------------
# The dataset contains baby names, one per line. We read them into a list.
words = open("./names.txt", "r", encoding="utf-8").read().splitlines()

# Step 2: Identify unique characters
# ----------------------------------
# To create a mapping from characters to indices (and vice versa), we collect all unique characters from the dataset.
# These will serve as the vocabulary for our bigram model.
characters = sorted(list(set("".join(words))))

# Step 3: Initialize bigram count matrix
# --------------------------------------
# We define a matrix that will hold the count of how many times each character is followed by another.
# We add a special character '^' to denote the start and end of a name.
# The matrix is of size (vocab_size x vocab_size), where vocab_size includes this special token.
vocab_size = len(characters) + 1  # +1 for the special token '^'
bigram_counts = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

# We create two mappings:
# stoi: maps character to index, where '^' is 0
# itos: maps index back to character
stoi = { s: i+1 for i, s in enumerate(characters) }  # Characters get indices from 1
stoi['^'] = 0
itos = { i: s for s, i in stoi.items() }

# Step 4: Count all bigrams in the dataset
# ----------------------------------------
# For each name, we add a '^' at the beginning and end to represent start and end boundaries.
# We then count how many times each character pair (bigram) appears in the dataset.
for name in words:
    padded_name = ['^'] + list(name) + ['^']
    for ch1, ch2 in zip(padded_name, padded_name[1:]):
        ix1 = stoi[ch1]  # index of first character
        ix2 = stoi[ch2]  # index of second character
        bigram_counts[ix1][ix2] += 1  # increment the count for this bigram

# Optional Visualization: Uncomment the following to see the bigram frequency matrix
# -----------------------------------------------------------------------------------
# plt.figure(figsize=(16, 16))
# plt.imshow(bigram_counts, cmap='Blues')
# for i in range(vocab_size):
#     for j in range(vocab_size):
#         bigram = itos[i] + itos[j]
#         plt.text(j, i, bigram, ha="center", va="bottom", color="gray")
#         plt.text(j, i, bigram_counts[i, j].item(), ha="center", va="top", color="gray")
# plt.axis("off")
# plt.show()

# Step 5: Normalize counts to probabilities
# -----------------------------------------
# We convert the raw counts into probabilities by dividing each row by the sum of that row.
# This gives us a probability distribution over the next character, given the current one.
bigram_counts = bigram_counts.float()  # Convert to float for division
normalized_bigram_counts = bigram_counts / bigram_counts.sum(1, keepdim=True)

# Step 6: Generate names by sampling from the bigram model
# --------------------------------------------------------
# We start from the special '^' token (index 0) and sample characters until we reach '^' again (end of name).
sampled_index = 0  # Start from '^'

while True:
    # Get the probability distribution for the next character
    probs = normalized_bigram_counts[sampled_index]
    # Sample one character index based on this distribution
    sampled_index = torch.multinomial(probs, 1, replacement=True).item()

    # If the sampled character is the end symbol, we finish the name
    if itos[sampled_index] == '^':
        print()  # Move to a new line
        break

    # Otherwise, print the character (stay on the same line)
    print(itos[sampled_index], end="")

# End of script
