# 🧠 Bigram Name Generator

This project generates realistic-looking names using a simple **character-level bigram model** trained on a dataset of real names.  
It builds a transition matrix from character to character, then **samples new names** based on these probabilities.

---

## 📁 Project Structure

- `bigram_name_generator.py` — Main script that learns from data and generates names.
- `names.txt` — A text file containing training names, one per line.
- `README.md` — You're reading it.

---

## 🚀 How It Works (Step-by-Step Explanation)

### 1. Load the Dataset

The script starts by reading all the names from `names.txt`:

```python
words = open("./names.txt", "r", encoding="utf-8").read().splitlines()
```

Each word is a name like `emma`, `oliver`, etc.

---

### 2. Build Character Vocabulary

All unique characters from the dataset are collected to build a vocabulary.

```python
characters = sorted(list(set("".join(words))))
```

A special symbol `^` is added to represent both **start and end** of names, making the total size = 27 (assuming 26 lowercase letters).

---

### 3. Create Index Mappings

```python
stoi = { s:i+1 for i, s in enumerate(characters) }
stoi['^'] = 0  # Start/End token

itos = { i:s for s, i in stoi.items() }
```

These maps are used to convert between characters and integer indices.

---

### 4. Count Bigram Frequencies

```python
bigram_counts = torch.zeros((27, 27), dtype=torch.int32)

for name in words:
    padded_name = ["^"] + list(name) + ["^"]
    for ch1, ch2 in zip(padded_name, padded_name[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        bigram_counts[ix1][ix2] += 1
```

Each character pair (bigram) like `^e`, `em`, `ma`, `a^` is counted in a matrix.

---

### 5. Normalize Counts into Probabilities

```python
bigram_counts = bigram_counts.float()
normalized_bigram_counts = bigram_counts / bigram_counts.sum(1, keepdim=True)
```

Each row in the matrix now represents a **probability distribution** over the next character.

---

### 6. Generate Names via Sampling

```python
sampled_index = 0  # Start token

while True:
    bigram_index = normalized_bigram_counts[sampled_index]
    sampled_index = torch.multinomial(bigram_index, 1, replacement=True).item()
    
    print(itos[sampled_index], end="") if itos[sampled_index] != "^" else print()
    
    if sampled_index == 0:
        break
```

This loop:
- Starts with `^`
- Samples the next character based on the learned distribution
- Stops when it samples `^` again (end of name)

---

### 🔍 Optional: Visualize Bigram Matrix

Uncomment this section in your script to view a heatmap:

```python
plt.figure(figsize=(16,16))
plt.imshow(bigram_counts, cmap='Blues')

for i in range(27):
    for j in range(27):
        bigram = itos[i] + itos[j]
        plt.text(j, i, bigram, ha="center", va="bottom", color="gray")
        plt.text(j, i, bigram_counts[i,j].item(), ha="center", va="top", color="gray")

plt.axis("off")
plt.show()
```

---

## 🛠️ Requirements

- Python 3.6+
- PyTorch
- Matplotlib (only for visualization)

Install dependencies with:

```bash
pip install torch matplotlib
```

---

## 📄 How to Run

1. Add your own name list to `names.txt` (one per line).
2. Run the script:

```bash
python main.py
```

3. It will print randomly generated names based on your training data.

---

## 🔠 Example Output

```
mia
noah
ava
liam
```

---

## 🧠 Concepts Covered

- N-gram modeling (bigram)
- Character-level text generation
- Probability matrices
- Markov chains
- Sampling from distributions

---

## 👨‍💻 Author

**Arun Manikandan**  
Built with ❤️ and PyTorch  
🗕 Date: 2025-05-24

---

## 📜 License

MIT License – feel free to use, modify, and learn from this project.
