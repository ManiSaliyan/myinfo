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
5]import gensim.downloader as api
import random
model = api.load("glove-wiki-gigaword-100")
def generate_similar_words(seed_word, topn=10):
    if seed_word in model:
        return [word for word, _ in model.most_similar(seed_word, topn=topn)]
    else:
        return []
def create_paragraph(seed_word):
    similar_words = generate_similar_words(seed_word, topn=10)
    if not similar_words:
        return f"No similar words found for '{seed_word}'."
    random.shuffle(similar_words)
    selected_words = similar_words[:5]
    paragraph = f"In a world defined by {seed_word}, "
    paragraph += f"people found themselves surrounded by concepts like {', '.join(selected_words[:-1])}, and {selected_words[-1]}. "
    paragraph += f"These ideas shaped the way they thought, acted, and dreamed. "
    paragraph += f"Every step forward in their journey reflected the essence of '{seed_word}', "
    paragraph += f"bringing them closer to understanding the true meaning of {selected_words[0]}."
    return paragraph
seed = "freedom"
print(create_paragraph(seed))
`;
const info6 = `
pip install transformers torch

from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

input_sentences = [
 "The new phone I bought is absolutely amazing!",
 "Worst customer service ever. I'm never coming back.",
 "The experience was average, nothing special.",
 "Fast delivery and the packaging was perfect.",
 "The product broke within two days. Very disappointed."
]

results = sentiment_pipeline(input_sentences)

print("Sentiment Analysis Results:\n")
for sentence, result in zip(input_sentences, results):
 print(f"Input Sentence: {sentence}")
 print(f"Predicted Sentiment: {result['label']}, Confidence Score: {result['score']:.2f}\n")
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

const ml1= `
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/student/Downloads/cust_data.csv')

df.info()
df.head()

num_col= 'age'
mean = df[num_col].mean()
median = df[num_col].median()
mode = df[num_col].mode()[0]
std_dev = df[num_col].std()
variance = df[num_col].var()
data_range = df[num_col].max() - df[num_col].min()
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")
print(f"Range: {data_range}")

plt.figure(figsize=(4, 3))
sns.histplot(df[num_col], bins=10, kde=True, color='blue')
plt.title(f'Histogram of {num_col}')
plt.xlabel(num_col)
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(4,3))
sns.boxplot(x=df[num_col], color='green')
plt.title(f'Boxplot of {num_col}')
plt.show()

Q1 = df[num_col].quantile(0.25)
Q3 = df[num_col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df[num_col] < lower_bound) | (df[num_col] > upper_bound)][num_col]
print(f"Number of outliers detected: {len(outliers)}")

cat_col= 'gender'
counts = df[cat_col].value_counts()
print(counts)

plt.figure(figsize=(8,4))
sns.barplot(x=counts.index, y=counts.values)
plt.title(f"Bar chart of {cat_col}")
plt.xlabel(cat_col)
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(5,5))
plt.pie(counts, labels = counts.index, autopct='%1.1f%%')
plt.title(f"Pie chart of {cat_col}")
plt.show()
`;
const ml2= `
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
df = data.frame
x_col = 'sepal_length'
y_col = 'petal_length'
data.describe()

correlation = df[[x_col,y_col]].corr('pearson')
print("Pearson Correlation Coefficient:\n", correlation)

covariance = df[[x_col,y_col]].cov()
print("Covariance Matrix:\n", covariance)

plt.figure(figsize=(8, 5))
plt.scatter(df[x_col], df[y_col])
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title(f"Scatter Plot of {x_col} vs {y_col}")
plt.show()

data_co = df.iloc[:, :-1]
covariance_matrix = data_co.cov()
correlation_matrix = data_co.corr()
print("Covariance Matrix:\n", covariance_matrix)
print("\n Correlation Matrix:\n", correlation_matrix)

plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
`;
const ml3= `
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

df_pca = pd.DataFrame(X_pca, columns=['PC1','PC2'])
df_pca['Species'] = y

plt.figure(figsize=(6,4))
colors = ['brown','hotpink','purple']
for i, color in zip(np.unique(y), colors): plt.scatter(df_pca.loc[df_pca['Species']==i,'PC1'],df_pca.loc[df_pca['Species']==i,'PC2'],c=color,label=iris.target_names[i])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Iris Dataset")
plt.legend()
plt.show()
`;
const ml4= `
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

def cls_knn(X_train, X_test, y_train, y_test, k_values, weighted=False):
   results = {}
   for k in k_values:
       if weighted:
           knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
       else:
           knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
       knn.fit(X_train, y_train)
       y_pred = knn.predict(X_test)
       accuracy = accuracy_score(y_test, y_pred)
       f1 = f1_score(y_test, y_pred, average='weighted')  # Weighted F1-score for multi-class
       results[k] = {'accuracy': accuracy, 'f1_score': f1}
   return results

k_values = [1, 3, 5]

print("Regular k-NN Results:")
regular_knn = cls_knn(X_train, X_test, y_train, y_test, k_values, weighted=False)
for k, metrics in regular_knn.items():
   print(f"k={k}: Accuracy={metrics['accuracy']:.4f}, F1-Score={metrics['f1_score']:.4f}")

print("\nWeighted k-NN Results:")
weighted_knn = cls_knn(X_train, X_test, y_train, y_test, k_values, weighted=True)
for k, metrics in weighted_knn.items():
   print(f"k={k}: Accuracy={metrics['accuracy']:.4f}, F1-Score={metrics['f1_score']:.4f}")

print("\nComparison of Regular k-NN and Weighted k-NN:")
for k in k_values:
   regular_acc = regular_knn[k]['accuracy']
   weighted_acc = weighted_knn[k]['accuracy']
   print(f"k={k}: Regular k-NN Accuracy={regular_acc:.4f}, Weighted k-NN Accuracy={weighted_acc:.4f}")
`;
const ml6= `
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

np.random.seed(42)

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
y = y + 10 * np.sin(X[:, 0] * 2)

plt.scatter(X, y, color='blue', label='Data Points')
plt.title("Synthetic Dataset")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.legend()
plt.show()

def locally_weighted_regression(X, y, query_point, tau=0.1):
    weights = np.exp(-np.sum((X - query_point) ** 2, axis=1) / (2 * tau ** 2))
    X_bias = np.c_[np.ones(X.shape[0]), X]
    W = np.diag(weights)
    theta = np.linalg.inv(X_bias.T @ W @ X_bias) @ (X_bias.T @ W @ y)
    query_point_bias = np.array([1, query_point[0]])
    y_pred = query_point_bias @ theta
    return y_pred

def predict_lwr(X_train, y_train, X_test, tau=0.1):
    y_pred = np.zeros(X_test.shape[0])
    for i, query_point in enumerate(X_test):
        y_pred[i] = locally_weighted_regression(X_train, y_train, query_point, tau)
    return y_pred

X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
tau = 0.1
y_pred = predict_lwr(X, y, X_test, tau)

plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X_test, y_pred, color='red', label='LWR Fit')
plt.title(f"Locally Weighted Regression (tau={tau})")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.legend()
plt.show()

mse = mean_squared_error(y, predict_lwr(X, y, X, tau))
print(f"Mean Squared Error (MSE) on Training Data: {mse:.4f}")
`;

const ml7 = `
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

boston_df = pd.read_csv("boston_housing_data.csv")

print("Linear Regression on Boston Housing Dataset")

X = boston_df[['RM']]
y = boston_df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

LR_model = LinearRegression()
LR_model.fit(X_train, y_train)
y_pred = LR_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

plt.scatter(X_test, y_test, color='green', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('House Price (MEDV)')
plt.title('Linear Regression on Boston Housing Dataset')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

auto_df = pd.read_csv("auto-mpg.csv")
print("Polynomial Regression on Auto MPG Dataset")

auto_df['horsepower'] = auto_df['horsepower'].replace('?', np.nan).astype(float)
auto_df.dropna(inplace=True)

X = auto_df[['horsepower']]
y = auto_df['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

PR_model = LinearRegression()
PR_model.fit(X_train_poly, y_train)
y_pred = PR_model.predict(X_test_poly)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

plt.scatter(X_test, y_test, color='purple', label='Actual')
sorted_indices = X_test.squeeze().argsort()
plt.plot(X_test.iloc[sorted_indices], y_pred[sorted_indices], color='red', label='Predicted')
plt.xlabel('Horsepower')
plt.ylabel('MPG (Miles Per Gallon)')
plt.title('Polynomial Regression on Auto MPG Dataset')
plt.legend()
plt.show()
`;

const ml8 = `
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = sns.load_dataset('titanic')

print(data.head())
print(data.info())

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

data = data[features + ['survived']].dropna()

data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['embarked'] = data['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

X = data[features]
y = data['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_split=2, random_state=42)
clf.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=features, class_names=['Not Survived', 'Survived'], filled=True)
plt.title("Decision Tree for Titanic Dataset")
plt.show()

y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
`;

const ml9 = `
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Naive Bayes classifier: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.show()
`;

const ml10 = `
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y_true = data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
silhouette_avg = silhouette_score(X_scaled, y_kmeans)
ari_score = adjusted_rand_score(y_true, y_kmeans)
print(f"Silhouette Score: {silhouette_avg:.3f}")
print(f"Adjusted Rand Index: {ari_score:.3f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 4))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_kmeans, palette="coolwarm", s=60)

plt.title('K-Means Clustering Result (PCA-reduced data)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_true, palette="Set2", s=60)
plt.title('True Labels (PCA-reduced data)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

plt.legend(title="Actual Class")

plt.grid(True)

plt.show()
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
app.get('/ml1', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml1);   // return the Python code
});
app.get('/ml2', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml2);   // return the Python code
});
app.get('/ml3', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml3);   // return the Python code
});
app.get('/ml4', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml4);
});
app.get('/ml6', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml6);   // return the Python code
});
app.get('/ml7', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml7);   // return the Python code
});
app.get('/ml8', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml8);   // return the Python code
});
app.get('/ml9', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml9);   // return the Python code
});
app.get('/ml10', (req, res) => {
  res.type('text/plain'); // set content type as plain text
  res.send(ml10);   // return the Python code
});
app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
module.exports = app;
