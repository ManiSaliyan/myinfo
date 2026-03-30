const express = require('express');
const app = express();
const PORT = 3000;
const cors = require('cors');
app.use(cors());
// Example GET route: /get-letter?filename=example.jpg
const info1 = `
!pip install gensim scipy

import gensim.downloader as api
from scipy.spatial.distance import cosine

print("Loading Word2Vec model...")
model = api.load("word2vec-google-news-300")
print("Model loaded successfully.\n")

vector = model['king']

print("First 10 dimensions of 'king' vector:")
print(vector[:10], "\n")

print("Top 10 words most similar to 'king':")
for word, similarity in model.most_similar('king'):
    print(f"{word}: {similarity:.4f}")
print()

result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print("Analogy - 'king' - 'man' + 'woman' ≈ ?")
print(f"Result: {result[0][0]} (Similarity: {result[0][1]:.4f})\n")

print("Analogy - 'paris' + 'italy' - 'france' ≈ ?")
for word, similarity in model.most_similar(positive=['paris', 'italy'], negative=['france']):
    print(f"{word}: {similarity:.4f}")
print()

print("Analogy - 'walking' + 'swimming' - 'walk' ≈ ?")
for word, similarity in model.most_similar(positive=['walking', 'swimming'], negative=['walk']):
    print(f"{word}: {similarity:.4f}")
print()

similarity = 1 - cosine(model['king'], model['queen'])
print(f"Cosine similarity between 'king' and 'queen': {similarity:.4f}")

def explore_word_relationships(word1, word2, word3):
    print(f"Relationship between '{word1}', '{word2}', and '{word3}':")
    result = model.most_similar(positive=[word2, word3], negative=[word1], topn=1)
    print(f"Result: {result[0][0]} (Similarity: {result[0][1]:.4f})\n")
    return result

explore_word_relationships("king", "man", "woman")
explore_word_relationships("paris", "france", "germany")
explore_word_relationships("apple", "fruit", "carrot")

def analyze_similarity(word1, word2):
    similarity = 1 - cosine(model[word1], model[word2])
    print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")

analyze_similarity("cat", "dog")
analyze_similarity("computer", "keyboard")
analyze_similarity("music", "art")

def find_most_similar(word):
    similar_words = model.most_similar(word, topn=5)
    print(f"Top 5 similar words to '{word}':")
    for similar_word in similar_words:
        print(f"{similar_word[0]}: {similar_word[1]:.4f}")

find_most_similar("happy")
print("\n")
find_most_similar("sad")
print("\n")
find_most_similar("technology")
`;
const info2 = `
!pip install gensim numpy matplotlib scikit-learn

import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

print("Loading pre-trained word vectors...")
wv = api.load("word2vec-google-news-300")
print("Model loaded successfully.")

try:
    vec = wv["king"] - wv["man"] + wv["woman"]
    sims = [(word, sim) for word, sim in wv.similar_by_vector(vec, topn=10) if word not in {"king","man","woman"}]
    print("\nWord Relationship: king - man + woman => Most similar words:")
    for word, sim in sims[:5]:
        print(f"{word}: {sim:.4f}")
except KeyError as e:
    print(f"Error: {e} not in vocabulary")

words = ["king", "man", "woman", "queen", "prince", "princess", "royal", "throne"]
words += [w for w, _ in sims[:5]]
vectors = np.array([wv[w] for w in words])

pca = PCA(n_components=2)
reduced_pca = pca.fit_transform(vectors)
plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    plt.scatter(*reduced_pca[i], color='blue')
    plt.text(reduced_pca[i,0]+0.02, reduced_pca[i,1]+0.02, word, fontsize=12)
plt.title("Word Embeddings Visualization (PCA)")
plt.grid(True)
plt.show()

tsne = TSNE(n_components=2, random_state=42, perplexity=3)
reduced_tsne = tsne.fit_transform(vectors)
plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    plt.scatter(*reduced_tsne[i], color='blue')
    plt.text(reduced_tsne[i,0]+0.02, reduced_tsne[i,1]+0.02, word, fontsize=12)
plt.title("Word Embeddings Visualization (t-SNE)")
plt.grid(True)
plt.show()

import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

model = api.load("word2vec-google-news-300")

words = ['computer', 'internet', 'software', 'hardware', 'keyboard', 'mouse', 'server', 'network', 'programming', 'database']
vectors = [model[word] for word in words]

pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

input_word = 'computer'
similar_words = model.most_similar(input_word, topn=5)

print(f"Top 5 words similar to '{input_word}':")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")

plt.figure(figsize=(8, 6))
for i, word in enumerate(words):
    plt.scatter(reduced[i, 0], reduced[i, 1])
    plt.annotate(word, (reduced[i, 0], reduced[i, 1]))
plt.title("PCA Visualization of Technology Word Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)

plt.show()
`;
const info3 = `
!pip install gensim matplotlib scikit-learn

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from gensim.models import Word2Vec

medical_corpus = [
"The patient was diagnosed with diabetes and hypertension.",
"MRI scans reveal abnormalities in the brain tissue.",
"The treatment involves antibiotics and regular monitoring.",
"Symptoms include fever, fatigue, and muscle pain.",
"The vaccine is effective against several viral infections.",
"Doctors recommend physical therapy for recovery.",
"The clinical trial results were published in the journal.",
"The surgeon performed a minimally invasive procedure.",
"The prescription includes pain relievers and anti-inflammatory drugs.",
"The diagnosis confirmed a rare genetic disorder."
]

processed_corpus = [sentence.lower().split() for sentence in medical_corpus]


model = Word2Vec(sentences=processed_corpus, vector_size=100, window=5, min_count=1,workers=4, epochs=50)


words = list(model.wv.index_to_key) # List of words in the vocabulary
embeddings = np.array([model.wv[word] for word in words]) # Word embeddings for each word

tsne = TSNE(n_components=2, random_state=42, perplexity=5)
tsne_result = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], color="blue")

for i, word in enumerate(words):
    plt.text(tsne_result[i, 0] + 0.02, tsne_result[i, 1] + 0.02, word, fontsize=12)
plt.title("Word Embeddings Visualization (Medical Domain)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()


def find_similar_words(input_word, top_n=5):
    try:
        similar_words = model.wv.most_similar(input_word, topn=top_n)
        print(f"Words similar to '{input_word}':")
        for word, similarity in similar_words:
            print(f" {word} ({similarity:.2f})")
    except KeyError:
        print(f"'{input_word}' not found in vocabulary.")

find_similar_words("treatment")
find_similar_words("vaccine")`;
const info4 = `
!pip install gensim matplotlib scikit-learn

import gensim.downloader as api
from transformers import pipeline
import nltk
import string
from nltk.tokenize import word_tokenize

nltk.download('punkt')

print("Loading pre-trained word vectors...")
word_vectors = api.load("glove-wiki-gigaword-100")

import gensim.downloader as api
from transformers import pipeline
import nltk
import string
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
print("Loading pre-trained word vectors...")
word_vectors = api.load("glove-wiki-gigaword-100")

def replace_keyword_in_prompt(prompt, keyword, word_vectors, topn=1):
    """
 Replace only the specified keyword in the prompt with its most similar word.
 Args:
 prompt (str): The original input prompt.
 keyword (str): The word to be replaced with a similar word.
 word_vectors (gensim.models.KeyedVectors): Pre-trained word embeddings.
 topn (int): Number of top similar words to consider (default: 1).
 Returns:
    str: The enriched prompt with the keyword replaced.
    """
    words = word_tokenize(prompt)
    enriched_words = []
    for word in words:
        cleaned_word = word.lower().strip(string.punctuation)
        if cleaned_word == keyword.lower():
        try:
            similar_words = word_vectors.most_similar(cleaned_word, topn=topn)
            if similar_words:
                replacement_word = similar_words[0][0]
                print(f"Replacing '{word}' → '{replacement_word}'")
                enriched_words.append(replacement_word)
                continue
        except KeyError:
            print(f"'{keyword}' not found in the vocabulary. Using original word.")
        enriched_words.append(word)
    enriched_prompt = " ".join(enriched_words)
    print(f"\n Enriched Prompt: {enriched_prompt}")
    return enriched_prompt

print("\nLoading GPT-2 model...")
generator = pipeline("text-generation", model="gpt2")

def generate_response(prompt, max_length=100):
  try:
    response = generator(prompt, max_length=max_length, num_return_sequences=1)
    return response[0]['generated_text']
  except Exception as e:
    print(f"Error generating response: {e}")
    return None

original_prompt = "Who is king."
print(f"\n Original Prompt: {original_prompt}")
key_term = "king"
enriched_prompt = replace_keyword_in_prompt(original_prompt, key_term, word_vectors)

print("\nGenerating response for the original prompt...")
original_response = generate_response(original_prompt)
print("\nOriginal Prompt Response:")
print(original_response)
print("\nGenerating response for the enriched prompt...")
enriched_response = generate_response(enriched_prompt)
print("\nEnriched Prompt Response:")
print(enriched_response)

print("\n🔍 Comparison of Responses:")
print("Original Prompt Response Length:", len(original_response))
print("Enriched Prompt Response Length:", len(enriched_response))
print("Original Prompt Sentence Count:", original_response.count("."))
print("Enriched Prompt Sentence Count:", enriched_response.count("."))`;
const info5 = `
def is_safe(board, row, col):
    for i in range(row):
        if board[i] == col:
            return False
    for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
        if board[i] == j:
            return False
    for i, j in zip(range(row-1, -1, -1), range(col+1, 8)):
        if board[i] == j:
            return False
    return True

def solve_queens_util(board, row):
    if row >= 8:
        return True
    for col in range(8):
        if is_safe(board, row, col):
            board[row] = col
            if solve_queens_util(board, row + 1):
                return True
            board[row] = -1
    return False

def solve_queens():
    board = [-1] * 8
    if not solve_queens_util(board, 0):
        print("Solution does not exist")
        return False
    print("Solution:")
    for i in range(8):
        for j in range(8):
            if board[i] == j:
                print("Q", end=" ")
            else:
                print(".", end=" ")
        print()
    return True

solve_queens()
`;
const info6 = `
import numpy as np

def tsp_nearest_neighbor(distances):
    num_cities = distances.shape[0]
    visited = [False] * num_cities
    tour = []
    current_city = 0
    tour.append(current_city)
    visited[current_city] = True

    for _ in range(num_cities - 1):
        nearest_city = None
        nearest_distance = float('inf')
        for next_city in range(num_cities):
            if not visited[next_city] and distances[current_city, next_city] < nearest_distance:
                nearest_city = next_city
                nearest_distance = distances[current_city, next_city]
        current_city = nearest_city
        tour.append(current_city)
        visited[current_city] = True

    tour.append(tour[0])  # Return to starting city
    return tour

if __name__ == "__main__":
    distances = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    tour = tsp_nearest_neighbor(distances)
    print("Tour:", tour)
`;
const info7 = `
class KnowledgeBase:
    def __init__(self):
        self.known_facts = set()
        self.inference_rules = []

    def add_fact(self, fact):
        self.known_facts.add(fact)

    def add_rule(self, condition, result):
        self.inference_rules.append((condition, result))

    def forward_chaining(self, target):
        derived_facts = set()
        to_process = list(self.known_facts)

        while to_process:
            current = to_process.pop(0)
            if current == target:
                return True

            for condition, result in self.inference_rules:
                if condition in derived_facts:
                    if result not in derived_facts and result not in to_process:
                        to_process.append(result)

            derived_facts.add(current)

        return False

if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.add_fact("A")
    kb.add_fact("B")
    kb.add_rule("A", "C")
    kb.add_rule("B", "C")
    kb.add_rule("C", "D")
    
    target_goal = "D"
    if kb.forward_chaining(target_goal):
        print(f"The goal '{target_goal}' is reachable.")
    else:
        print(f"The goal '{target_goal}' is not reachable.")
`;
const info8 = `
class Statement:
    def __init__(self, predicate_name, parameters):
        self.predicate_name = predicate_name
        self.parameters = parameters

    def __eq__(self, other):
        return isinstance(other, Statement) and self.predicate_name == other.predicate_name and self.parameters == other.parameters

    def __hash__(self):
        return hash((self.predicate_name, tuple(self.parameters)))

    def __str__(self):
        return f"{self.predicate_name}({', '.join(self.parameters)})"

    def __lt__(self, other):
        if not isinstance(other, Statement):
            return NotImplemented
        if self.predicate_name < other.predicate_name:
            return True
        elif self.predicate_name == other.predicate_name:
            return self.parameters < other.parameters
        else:
            return False

class Rule:
    def __init__(self, statements):
        self.statements = set(statements)

    def __eq__(self, other):
        return isinstance(other, Rule) and self.statements == other.statements

    def __hash__(self):
        return hash(tuple(sorted(self.statements)))

    def __str__(self):
        return " | ".join(str(stmt) for stmt in self.statements)

def apply_resolution(rule1, rule2):
    new_rules = set()
    for stmt1 in rule1.statements:
        for stmt2 in rule2.statements:
            if stmt1.predicate_name == stmt2.predicate_name and stmt1.parameters != stmt2.parameters:
                merged_statements = (rule1.statements | rule2.statements) - {stmt1, stmt2}
                new_rules.add(Rule(merged_statements))
    return new_rules

def resolution_process(knowledge_base, goal):
    pending_rules = list(knowledge_base)
    while pending_rules:
        current = pending_rules.pop(0)
        for existing in list(knowledge_base):
            if current != existing:
                new_generated = apply_resolution(current, existing)
                for new_rule in new_generated:
                    if new_rule not in knowledge_base:
                        pending_rules.append(new_rule)
                        knowledge_base.add(new_rule)
                    if not new_rule.statements:
                        return True
                    if goal in new_rule.statements:
                        return True
    return False

if __name__ == "__main__":
    kb = {
        Rule({Statement("P", ["a", "b"]), Statement("Q", ["a"])}),
        Rule({Statement("P", ["x", "y"])}),
        Rule({Statement("Q", ["y"]), Statement("R", ["y"])}),
        Rule({Statement("R", ["z"])}),
    }

    target = Statement("R", ["a"])
    found = resolution_process(kb, target)

    if found:
        print("Query is satisfiable.")
    else:
        print("Query is unsatisfiable.")
`;
const info9 = `
class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'

    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    def make_move(self, position):
        if self.board[position] == ' ':
            self.board[position] = self.current_player
            if self.check_winner(position):
                print(f"Player {self.current_player} wins!")
                return True
            elif ' ' not in self.board:
                print("It's a tie!")
                return True
            else:
                self.current_player = 'O' if self.current_player == 'X' else 'X'
                return False
        else:
            print("That position is already taken!")
            return False

    def check_winner(self, position):
        row_index = position // 3
        col_index = position % 3
        # Check row
        if all(self.board[row_index*3 + i] == self.current_player for i in range(3)):
            return True
        # Check column
        if all(self.board[col_index + i*3] == self.current_player for i in range(3)):
            return True
        # Check diagonal
        if row_index == col_index and all(self.board[i*3 + i] == self.current_player for i in range(3)):
            return True
        # Check anti-diagonal
        if row_index + col_index == 2 and all(self.board[i*3 + (2-i)] == self.current_player for i in range(3)):
            return True
        return False

def main():
    game = TicTacToe()
    while True:
        game.print_board()
        position = int(input(f"Player {game.current_player}, enter your position (0-8): "))
        if game.make_move(position):
            game.print_board()
            break

if __name__ == "__main__":
    main()
`;
app.get('/info1', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info1);   // return the Python code
});
app.get('/info2', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info2);   // return the Python code
});
app.get('/info3', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info3);   // return the Python code
});
app.get('/info4', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info4);   // return the Python code
});
app.get('/info5', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info5);   // return the Python code
});
app.get('/info6', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info6);   // return the Python code
});
app.get('/info7', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info7);   // return the Python code
});
app.get('/info8', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info8);   // return the Python code
});
app.get('/info9', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(info9);   // return the Python code
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
module.exports = app;
