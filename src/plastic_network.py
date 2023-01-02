import networkx as nx
import time
import warnings
from utils import create_transition_matrix, plot_transition_matrix, generate_probability_rank_list, sort_dict_by_value, get_descendants_with_counts, get_counter, get_entropy, format_and_write_csv

class PlasticNetworkConstructor():
    def __init__(self):
        self.graph = nx.DiGraph()

    def save(self, filename="graph.gpickle"):
        """
        Save the graph to pickle file.
        """
        nx.write_gpickle(self.graph, filename)
    
    def load(self, filename="graph.gpickle"):
        """
        Load a graph from a pickle file.
        """
        self.graph = nx.read_gpickle(filename)

    def add_node(self, node, **kwargs):
        print("adding node : ", node, " with data ", kwargs)
        if not self.graph.has_node(node):
            self.graph.add_node(node, **kwargs)

    def remove_node(self, node):
        print("removing node : ", node)
        if self.graph.has_node(node):
            self.graph.remove_node(node)

    def add_edge(self, source, target, **kwargs):
        print("adding edge : ", source, " -> ", target, " with data ", kwargs)
        if not self.graph.has_edge(source, target) and source != target:
            self.graph.add_edge(source, target, **kwargs)
    
    def remove_edge(self, source, target):
        print("removing edge : ", source, " -> ", target)
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)

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

    def get_ranked_result(self, samples, current_inference_string):
        inf_string_len = len(current_inference_string)
        counter = get_counter(samples, inf_string_len)
        # print("COUNTER:", counter)
        transition_matrix, samples = create_transition_matrix(counter)
        plot_transition_matrix(transition_matrix, counter)
        rank_list = generate_probability_rank_list(samples, transition_matrix, current_inference_string)
        # print(sort_dict_by_value(rank_list))
        selected = sort_dict_by_value(rank_list)[-1]
        return selected, rank_list

    def get_next_inference_string(self, transition_samples, current_inference_string):
        print("Current inference string: ", current_inference_string)
        selected, ranked_list = self.get_ranked_result(transition_samples, current_inference_string)
        print("Selecting : ", selected, " with p ", ranked_list[selected])
        # next_inference_string = inference_string + selected
        next_inference_string = current_inference_string + selected
        print("Next inference string:", next_inference_string)
        return next_inference_string, True
        
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
        import graphviz
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
            # print("adding: %s -> %s | time: --- %s seconds ---" % (source_node,target_node, time.time() - start_time))
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

    def format_and_write_csv(self):
        format_and_write_csv(self.graph)

