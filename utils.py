import networkx as nx
import numpy as np
import math
from collections import Counter

def normalize(arr):
    a_norm = np.nan_to_num((arr/arr.sum(axis=1)[:,None]).round(7))
    return a_norm

def create_transition_matrix(counter):
    # Create a list of the unique words in the Counter object.
    samples = list(set([word for pair in counter for word in pair]))
    # Create a numpy array with dimensions (number of words) x (number of words).
    transition_matrix = np.zeros((len(samples), len(samples)), dtype=np.float64)
    # Loop through the pairs of words in the Counter object and update the adjacency matrix.
    for (word1, word2), count in counter.items():
        i = samples.index(word1)
        j = samples.index(word2)
        transition_matrix[i, j] = count

    # Normalize the rows of the transition matrix to sum to 1.
    transition_matrix = normalize(transition_matrix)
    return transition_matrix, samples


# def create_transition_matrix(counter):
#     # Create a list of the unique words in the Counter object.
#     samples = list(set([word for pair in counter for word in pair]))
#     # Create a numpy array with dimensions (number of words) x (number of words).
#     transition_matrix = torch.zeros((len(samples), len(samples)), dtype=torch.float64)
#     # print("MATRIX DIMENSION: ",(len(words), len(words)) )
#     # Loop through the pairs of words in the Counter object and update the adjacency matrix.
#     for (word1, word2), count in counter.items():
#         i = samples.index(word1)
#         j = samples.index(word2)
#         transition_matrix[i, j] = count

#     # print(transition_matrix.sum(axis=1), transition_matrix.sum(axis=0))
#     transition_matrix = normalize(transition_matrix, p=1.0, dim = 1)
#     return transition_matrix, samples

def plot_transition_matrix(transition_matrix, counter):
    import matplotlib.pyplot as plt
    # Use imshow to plot the adjacency matrix as an image.
    plt.imshow(transition_matrix)
    samples = list(set([word for pair in counter for word in pair]))
    # Loop through the elements of the adjacency matrix and add a text label for each element.
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            plt.text(j, i, round(transition_matrix[i, j].item(), 2), ha="center", va="center", color="w")

    # Show the plot.
    plt.xticks(range(len(samples)), samples, rotation=45)
    plt.yticks(range(len(samples)), samples)

    # Show the plot.
    plt.show()

def infer_probabilities(tensor, samples):
        matrix = tensor[0]
        probabilities = {}
        for i in range(0, len(matrix)):
            if matrix[i] != 0.0:
                probabilities[samples[i]] = matrix[i]
        return probabilities

# def infer_probabilities(tensor, samples):
#         matrix = tensor.numpy()[0]
#         probabilities = {}
#         for i in range(0, len(matrix)):
#             if matrix[i] != 0.0:
#                 probabilities[samples[i]] = matrix[i]
#         return probabilities

def generate_probability_rank_list(samples, transition_matrix, input):
    # Create a one-hot vector of size 1*4
    size = len(samples)
    one_hot_vector = np.zeros((1, size), dtype=np.float64)
    # index = 3
    index = [i for i,x in enumerate(samples) if x == input]
    if len(index) > 0:
        one_hot_vector[0][index] = 1.0
    print("--- inferring for : ", index)
    print("SAMPLES: ", samples)
    print("INFERENCE VECTOR:", infer_probabilities(one_hot_vector, samples))
    # Multiply the one-hot vector with the transition matrix
    result = np.matmul(one_hot_vector, transition_matrix)

    # Print the resulting vector
    print("TESTING:", result, samples)
    print("INFERENCE RESULT:", infer_probabilities(result, samples))
    print("--- inference ended")
    return infer_probabilities(result, samples)

# def generate_probability_rank_list(samples, transition_matrix, input):
#     # Create a one-hot vector of size 1*4
#     size = len(samples)
#     one_hot_vector = torch.zeros((1, size), dtype=torch.float64)
#     # index = 3
#     index = [i for i,x in enumerate(samples) if x == input]
#     if len(index) > 0:
#         one_hot_vector[0][index] = 1.0
#     print("--- inferring for : ", index)
#     print("SAMPLES: ", samples)
#     print("INFERENCE VECTOR:", infer_probabilities(one_hot_vector, samples))
#     # Multiply the one-hot vector with the transition matrix
#     result = torch.matmul(one_hot_vector, transition_matrix)

#     # Print the resulting vector
#     print("INFERENCE RESULT:", infer_probabilities(result, samples))
#     print("--- inference ended")
#     return infer_probabilities(result, samples)

# define a function that takes a dictionary as input
def sort_dict_by_value(d):
    # create an empty list to store the keys
    keys = []
    
    # iterate over the dictionary items
    for key, value in d.items():
        # append the key to the list
        keys.append(key)
    
    # sort the keys by value
    keys.sort(key=lambda x: d[x])
    
    # return the sorted keys
    return keys



def get_descendants_with_counts(graph : nx.DiGraph, inference_string):
    sample_list = []
    descendants = [v for u,v in list(graph.out_edges(inference_string))]
    for descendant in descendants:
        if len(list(nx.all_simple_paths(graph, source=inference_string, target=descendant))) == 1:
            # sample_count = current_refactoring[descendant]['count']
            print("DESCENDANT:", descendant)
            sample_count = graph.nodes[descendant]['count']
            for i in range(0, sample_count):
                sample_list.append(descendant)
    return sample_list

def get_counter(samples, window_size=1):
    counter = {}
    # print("GET COUNTER:", samples)
    for item in samples:
        for i, char in enumerate(item):
            window = item[i:i+window_size]
            selected = window
            if i+window_size < len(item):
                output = item[i+window_size:]
                pair = (selected, output)
                if pair in counter.keys():
                    counter[pair] += 1
                else:
                    counter[pair] = 1
    return counter



def get_entropy(data, unit='natural'):
    base = {
        'shannon' : 2.,
        'natural' : math.exp(1),
        'hartley' : 10.
    }

    if len(data) <= 1:
        return 0

    counts = Counter()

    for d in data:
        for char in d:
            counts[char] += 1

    ent = 0

    probs = [float(c) / len(data) for c in counts.values()]
    for p in probs:
        if p > 0.:
            ent -= p * math.log(p, base[unit])

    return ent

def write_csv(write_array, file):
    import csv    
    with open(file, 'w') as csvfile:    
        fieldnames = write_array[0].keys()    
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()    
        for item in write_array:
            writer.writerow(item)    
        
    print("Writing complete")  

def format_and_write_csv(graph: nx.DiGraph):
     
    edges_data = []
    node_metadata = []
    for edge in graph.edges():
        source_node = edge[0]
        target_node = edge[1]
        edge = {
            "source" : source_node,
            "target" : target_node,
            "color" : "black"
        }
        edges_data.append(edge)
    for node, data in graph.nodes(data=True):
        node_item = {
            "id" : node,
            "count" : data['count'],
            "r_level" : data['r_level']
        }
        node_metadata.append(node_item)
    write_csv(edges_data, "edges.csv")
    write_csv(node_metadata, "nodes.csv")



