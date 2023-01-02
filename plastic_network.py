from enlist_refactorings_graph import create_refactoring_graph, parse_seq_idx
import networkx as nx
import numpy as np
import graphviz
import time
import warnings
from utils import create_transition_matrix, plot_transition_matrix, generate_probability_rank_list, sort_dict_by_value, get_descendants_with_counts, get_counter, get_entropy


class PlasticNetworkConstructor():
    def __init__(self, samples):
        self.graph = create_refactoring_graph(samples)

    def save(self, filename="graph.gpickle"):
        nx.write_gpickle(self.graph, filename)
    
    def load(self, filename="graph.gpickle"):
        self.graph = nx.read_gpickle(filename)

    def get_refactorings_by_r_level(self, r_level):
        return [x for x,y in self.graph.nodes(data=True) if y['r_level']==r_level]

    def get_refactorings(self):
        refactorings = {}
        if self.graph:
            all_r_levels = nx.get_node_attributes(self.graph, 'r_level')
            
            for key in all_r_levels:
                if all_r_levels[key] in refactorings.keys():
                    refactorings[all_r_levels[key]] += [key]
                else:
                    refactorings[all_r_levels[key]] = [key]
        return refactorings

    def get_ranked_result(self, samples, current_inference_string, till_end=False):
        inf_string_len = len(current_inference_string)
        counter = get_counter(samples, inf_string_len)
        # print("COUNTER:", counter)
        transition_matrix, samples = create_transition_matrix(counter)
        plot_transition_matrix(transition_matrix, counter)
        rank_list = generate_probability_rank_list(samples, transition_matrix, current_inference_string)
        print(sort_dict_by_value(rank_list))
        selected = sort_dict_by_value(rank_list)[-1]
        return selected, rank_list

    def get_next_inference_string(self, transition_samples, current_inference_string):
        selected, ranked_list = self.get_ranked_result(transition_samples, current_inference_string, till_end=True)
        print("get_next_inference_string: ", selected, ranked_list)
        # next_inference_string = inference_string + selected
        next_inference_string = current_inference_string + selected
        return next_inference_string, True

    def infer(self, inference_string : str, visualize=False):
        print("REFACTORINGS:")
        # refactorings_w_counts = get_refactorings_with_counts(refactorings, G)
        refactorings = self.get_refactorings()
        first_r_level = 0
        step_counter = 0
        current_inference_string = inference_string if inference_string != "" else ">"
        print("STEP", step_counter, " -- ", current_inference_string)
        step_counter += 1
        start_token_found = False
        # # find the right starting refactoring  r-level to start
        for r_level in refactorings:
            for lvl_item in refactorings[r_level]:
                if current_inference_string in lvl_item:
                    first_r_level = r_level
                    print("FOUND! ",first_r_level, lvl_item, current_inference_string)
                    start_token_found = True
                    break
            if start_token_found:
                break
        # first_r_level = 1
        # start_token_found = True
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

            while not current_inference_string.endswith("_"):
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

    def rank_and_find_substrings(self, search_term, r_level_obj):
        found = False
        found_items = []
        for item in r_level_obj:
            match_strings = parse_seq_idx(search_term, item)
            print("MATCH: ", search_term, item, match_strings)
            if len(match_strings) > 0:
                found = True
                # ignore if last character is _, trivial matching
                match_strings = [x for x in match_strings if x != "_"]
                found_items.append((item, match_strings))
        return found_items, found

    def potentiate(self, from_abstraction, visualize=True, in_window=False):
        def increment_count(node):
            self.graph.nodes[node]['count'] += 1
            in_edges = self.graph.in_edges(node)
            print("IN_EDGES: ", node, in_edges)
            if len(in_edges) > 0:
                if visualize:
                    if in_window:
                        self.visualize_activation(highlight_edges=in_edges, highlight_colour="red", output_folder="view/potentiation", with_counts=True, with_r_levels=True)
                    else:
                        self.visualize_activation(highlight_edges=in_edges, highlight_colour="red", output_folder="", with_counts=True, with_r_levels=True)
            for edge in in_edges:
                print("INCREMENTING:", edge[0])
                if edge[0] == node:
                    warnings.warn(f"WARNING : Node {node} looping on itself.")
                increment_count(edge[0])
        return increment_count(from_abstraction)

    def learn(self, sample):
        token_teach_string = ">" + sample + "_"
        did_find = False
        print('REFACTORING:')
        refactorings = self.get_refactorings()
        current_r_level = len(refactorings.keys()) - 1
        # current_r_level = len(refactorings.keys())
        while did_find == False or current_r_level < 0:
            current_r_level_nodes = refactorings[current_r_level]
            print("START FROM: ", current_r_level_nodes)
            found_isomorphisms, did_find = self.rank_and_find_substrings(token_teach_string, current_r_level_nodes)
            print("STUFF: ", found_isomorphisms, did_find)

        print("FOUND ISOMORPHISMS : ", found_isomorphisms)
        for found_item, match_strings in found_isomorphisms:
            for match_string in match_strings:
                # check if abstraction already exists
                if (match_string, found_item) in self.graph.in_edges(found_item):
                    print("ERROR IFFIN:", match_string, found_item)
                    pass
                else:
                    # here, a new abstraction needs to be created
                    # print here
                    print("ADDING NEW ABSTRACTION", (match_string, found_item, token_teach_string))
                    new_abstraction = match_string
                    print("TESTING:", new_abstraction, found_item)
                    if new_abstraction != found_item:
                        found_item_incoming_nodes = [u for u,v in self.graph.in_edges(found_item)]
                        self.graph.remove_edges_from([(u,v) for u,v in self.graph.in_edges(found_item)])
                        self.graph.add_edge(new_abstraction, found_item)
                        self.graph.add_edge(new_abstraction, token_teach_string)
                        self.graph.nodes[token_teach_string]['count'] = 0
                        # this count should be come 2 after potentiation
                        self.graph.nodes[new_abstraction]['count'] = 1
                        self.graph.nodes[new_abstraction]['r_level'] = current_r_level
                        #stitch incoming nodes to new abstraction
                        self.graph.add_edges_from([(u, new_abstraction) for u in found_item_incoming_nodes if u in new_abstraction])
                    else:
                        print("ERROR MATCHING:", new_abstraction, found_item)

                    ## potentiate
                    self.potentiate(token_teach_string)
        
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

    def get_graphviz_obj(self, highlight_edges=[], highlight_nodes=[], highlight_colour='blue', output_folder=None, with_counts=False, with_r_levels=False):

        timestamp = time.time()
        w = graphviz.Digraph("random" + str(timestamp), format='png')
        w.attr(size="100,500")
        for edge in self.graph.edges():
            source_node = self.get_node_with_info(edge[0], with_counts, with_r_levels)
            target_node = self.get_node_with_info(edge[1], with_counts, with_r_levels)
            start_time = time.time()
            if edge in highlight_edges:
                # print("FOUND MATCH: ", edge, highlight_edges)
                
                w.edge(source_node, target_node, color=highlight_colour)
            else:
                # print("ADDING: ", source_node, target_node)
                w.edge(source_node, target_node)
            print("adding: %s -> %s | time: --- %s seconds ---" % (source_node,target_node, time.time() - start_time))
        for node in highlight_nodes:
            highlighted_node = self.get_node_with_info(node, with_counts, with_r_levels)
            w.node(highlighted_node, style='filled', fillcolor = highlight_colour, color = highlight_colour, fontcolor='white')
        if output_folder:
            w.render(directory=output_folder, view=True)
        return w
        
    def visualize(self, with_counts=False, with_r_levels=False):
        w = self.get_graphviz_obj(with_counts=with_counts, with_r_levels=with_r_levels)
        w
        return w
    def visualize_activation(self, highlight_edges=[], highlight_nodes=[], highlight_colour="blue", output_folder="view/test", with_counts=False, with_r_levels=False):
        print("VISUALIZING:", highlight_edges)
        w = self.get_graphviz_obj(highlight_edges=highlight_edges, highlight_nodes=highlight_nodes, highlight_colour=highlight_colour, output_folder=output_folder, with_counts=with_counts, with_r_levels=with_r_levels)
        w
        return w

