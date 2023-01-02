# 1. browser just represents the edges, the path through the edges creates the query to the lookup table
# 2. on inference, predicts sequence using markovian kernels, creates query, fetches info
# 3. on learning, find closes match abstraction, sends  

import networkx as nx
from plastic_network import PlasticNetworkConstructor
import os
import time
import graphviz
from utils import get_descendants_with_counts
import warnings

VISUALIZE_ACTIVATIONS = False

def get_prefix_matches(string_to_match, r_level_items):
    prefix_count = {}
    for item in r_level_items:
        common_prefix = os.path.commonprefix([item, string_to_match])
        if common_prefix in prefix_count.keys():
            prefix_count[common_prefix]['count'] += 1
            prefix_count[common_prefix]['matches'].append(item)
        else:
            prefix_count[common_prefix] = {
                'count' : 1,
                'matches' : [item]
            }
    return prefix_count

class PlasticPrefixNetwork(PlasticNetworkConstructor):
    def __init__(self, samples=[]):
        self.graph = self.init_samples(samples)
        
    def save(self, filename="graph.gpickle"):
        nx.write_gpickle(self.graph, filename)
    
    def load(self, filename="graph.gpickle"):
        self.graph = nx.read_gpickle(filename)

    def init_samples(self, samples):
        self.graph = nx.DiGraph()
        self.learn_multiple(samples)
        return self.graph

    def learn_multiple(self,samples):
        if len(samples)>0:
            for sample in samples:
                self.learn(sample)

    def r_lift_signal(self, node):
        print("LIFTING R at: ", node)
        self.graph.nodes[node]['r_level'] += 1
        if VISUALIZE_ACTIVATIONS:
            self.visualize_activation(highlight_edges=self.graph.out_edges(node), highlight_nodes=[node], highlight_colour='darkgreen', with_counts=True, with_r_levels=True)
        for u,v in self.graph.out_edges(node):
            self.r_lift_signal(v)

    def potentiate(self, node):
        print("POTENTIATING at: ", node)
        self.graph.nodes[node]['count'] += 1
        if VISUALIZE_ACTIVATIONS:
            self.visualize_activation(highlight_edges=self.graph.in_edges(node),highlight_nodes=[node], highlight_colour='darkblue', with_counts=True, with_r_levels=True)
        for u,v in self.graph.in_edges(node):
            self.potentiate(u)

    def add_node(self, node, **kwargs):
        print("adding node : ", node, " with data ", kwargs)
        if not self.graph.has_node(node):
            self.graph.add_node(node, **kwargs)

    def add_edge(self, source, target, **kwargs):
        print("adding edge : ", source, " -> ", target, " with data ", kwargs)
        if not self.graph.has_edge(source, target) and source != target:
            self.graph.add_edge(source, target, **kwargs)
    
    def remove_edge(self, source, target):
        print("removing edge : ", source, " -> ", target)
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)

    def select_match_string(self, prefix_count):
        # selected_key = list(prefix_count)[0]
        # select longest matched prefix
        selected_key = list(sorted(prefix_count.keys(), key=len))[-1]
        return selected_key, prefix_count[selected_key]['matches']

    def check_all_substrings(self, match_string, strings):
        substrings = []
        for string in strings:
            if string in match_string:
                substrings.append(string)

        return substrings

    def insert_at_level(self, teach_string, match_string, match_nodes, below_nodes, r_level):
        print("insert %s  at r_level %s ..." % (match_string, r_level))
        edges_to_remove = []
        for node in below_nodes:
            for u,v in self.graph.out_edges(node):
                if match_string in v:
                    edges_to_remove.append((u,v))
        for u,v in edges_to_remove:
            self.remove_edge(u,v)
        exists = self.graph.has_node(match_string)
        self.add_node(match_string, count=1, r_level=r_level)
        self.add_node(teach_string, count=0, r_level=r_level + 1)
        self.add_edge(match_string, teach_string)
        for node in match_nodes:
            self.add_edge(match_string, node)

        

        for node in below_nodes:
            self.add_edge(node, match_string)
        if not exists:
            self.r_lift_signal(match_string)
        self.potentiate(teach_string)

    def recursive_insert(self, prefix_count, teach_string, current_r_level, refactorings):
        match_string, match_nodes = self.select_match_string(prefix_count)
        print("CHECKING : ", match_string)
        below_level = current_r_level - 1
        if below_level in refactorings.keys():
            print("one level down does exist, hence finding out if should insert there or not")
            below_substrings = self.check_all_substrings(match_string, refactorings[below_level])
            if len(below_substrings) > 0:
                self.insert_at_level(teach_string, match_string, match_nodes, below_substrings, current_r_level-1)
            elif len(below_substrings) == 0:
                if match_string in refactorings[below_level]:
                    self.insert_at_level(teach_string, match_string, match_nodes, [], current_r_level-1)
                elif match_string not in refactorings[below_level]:
                    next_level_prefix_count = get_prefix_matches(teach_string, refactorings[below_level])
                    self.recursive_insert(next_level_prefix_count, teach_string, below_level, refactorings)
        elif below_level not in refactorings.keys():
            print("one level down doesn't exist, hence inserting at one level down.")
            self.insert_at_level(teach_string, match_string, match_nodes, [], current_r_level-1)

    def learn(self, teach_string):
        # ALGO STARTS HERE
        print("LEARNING: ", teach_string)
        refactorings = self.get_refactorings()
        # select nth r_level to start with, in this case the last one
        if len(refactorings) == 0:
            self.graph.add_node(teach_string, count=1, r_level=0)
        else:
            # find matches at give r_level
            start_r_level = max(refactorings.keys())
            current_r_level = start_r_level
            prefix_count = get_prefix_matches(teach_string, refactorings[current_r_level])
            print("PREFIX COUNT : ", prefix_count)
            self.recursive_insert(prefix_count, teach_string,current_r_level, refactorings)
        print("---END---")

    def infer(self, inference_string : str, visualize=False, stop_token="_"):
        print("REFACTORINGS:")
        # refactorings_w_counts = get_refactorings_with_counts(refactorings, G)
        refactorings = self.get_refactorings()
        first_r_level = 0
        step_counter = 0
        current_inference_string = inference_string
        print("STEP", step_counter, " -- ", current_inference_string)
        step_counter += 1
        start_token_found = False
        # # find the right starting refactoring  r-level to start
        for r_level in refactorings:
            if current_inference_string in refactorings[r_level]:
                first_r_level = r_level
                print("FOUND! ", first_r_level, refactorings[r_level], current_inference_string)
                start_token_found = True
                break
            if start_token_found:
                break
        print("FIRST R LEVEL: ", refactorings[first_r_level])
        if start_token_found == True:
            print("STEP", step_counter, " -- ", current_inference_string)
            step_counter += 1
            transition_samples = []
            if current_inference_string in refactorings[first_r_level]:
                print("EXACT MATCH FOUND. FORWARDING.")
                pass
            else:
                for x in refactorings[first_r_level]:
                    transition_samples += [x] * self.graph.nodes[x]['count']
                # make this a function
                next_inference_string, end_loop = self.get_next_inference_string(transition_samples, current_inference_string)
                if visualize:
                    self.visualize_activation(highlight_edges=[(current_inference_string, next_inference_string)], highlight_colour="blue", output_folder="view/inference")
                current_inference_string = next_inference_string

            while not current_inference_string.endswith(stop_token):
                print("STEP", step_counter, " -- ", current_inference_string)
                step_counter += 1
                transition_samples = get_descendants_with_counts(self.graph, current_inference_string)
                next_inference_string, end_loop = self.get_next_inference_string(transition_samples, current_inference_string)
                if visualize:
                    self.visualize_activation(highlight_edges=[(current_inference_string, next_inference_string)], highlight_colour="blue", output_folder="view/inference")
                current_inference_string = next_inference_string
            # print("NEXTMOST INFERENCE STRING: ", inference_string)
            # transition_samples = get_descendants_with_counts(G, common_substrings_count, inference_string)
            # inference_string, end_loop = get_next_inference_string(transition_samples, inference_string)
            print("END RESULT: ", current_inference_string)
        else:
            print("Insufficient data : Could not find start token.")
    
    def get_node_with_info(self, node, with_counts, with_r_levels):
        content = node
        if with_counts:
            try:
                content += " (c" + str(self.graph.nodes[node]['count']) + ")"
            except Exception as e:
                warnings.warn("Error occurred at node" + node + " " + str(e))
        if with_r_levels:
            try:
                content += " (r" + str(self.graph.nodes[node]['r_level']) + ")"
            except Exception as e:
                warnings.warn("Error occurred at node " + node + " " + str(e))

        return content
