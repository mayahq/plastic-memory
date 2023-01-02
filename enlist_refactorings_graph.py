import pylcs
import numpy as np
import networkx as nx
import time

def parse_seq_idx(phrase1, phrase2):
        data = pylcs.lcs_sequence_idx(phrase1, phrase2)
        sequence_indices = [x for x in np.split(data, np.where(np.diff(data) != 1)[0]+1) if not np.array_equal(x, [-1]) and not np.any(x == -1)]
        sequences = []
        for index in sequence_indices:
            string = "".join(phrase2[x] for x in index)
            sequences.append(string)
        return sequences

def create_refactoring_graph(samples):
        print("Creating refactoring graph...")
        G=nx.DiGraph()
        # get all possible common substrings
        samples_length = len(samples)
        print("Total samples length : %s" % (samples_length))
        common_substrings_count = {}
        # add the samples already there
        print("Calculating counts...")
        count_start_time = time.time()
        for sample in samples:
            if sample in common_substrings_count.keys():
                common_substrings_count[sample]["count"] += 1
            else:
                common_substrings_count[sample] = {
                    "count" : 1
                }
        count_end_time = time.time()
        print("Count time in s : %s" % (count_end_time - count_start_time))
        # print("samples: ", samples)
        print("Finding isomorphisms...")
        isomorphism_start_time = time.time()
        for i, _ in enumerate(samples):
            for j in range(i+1, len(samples)):
                # print("---")
                substrings = parse_seq_idx(samples[i], samples[j])
                # print("parsing: ", samples[i], samples[j])
                # print("substrings: ", substrings)
                substrings = [x for x in substrings if len(x) > 1]
                # print(words[i], words[j], " | ED | ", substrings)
                for string in substrings:
                    # print("string:", string, common_substrings_count)
                    if string in common_substrings_count.keys():
                        common_substrings_count[string]["count"] += 1
                    else:
                        common_substrings_count[string] = {
                            "count" : 2
                        }
            if i % 500 == 0:
                print("Status: %f / %f done. Section time %f" % (i, samples_length, time.time()-isomorphism_start_time))
                isomorphism_start_time = time.time()
        substring_list = list(common_substrings_count)
        # print('substrings:', substring_list)
        # if len(substring_list) > 0:
        #     max_abstraction_len = len(sorted(substring_list, key=len)[-1])
        # else:
        #     max_abstraction_len = 0
        # print("Common Substrings:", common_substrings_count)
        print("Creating graph from edges...")
        graph_start_time = time.time()
        includes_list = []
        for i in range(0, len(substring_list)):
            for j in range(i, len(substring_list)):
                # tuple = ({substring_list[i] : common_substrings_count[substring_list[i]]["count"]}, {substring_list[j] : common_substrings_count[substring_list[j]]["count"]})
                tuple = (substring_list[i], substring_list[j])
                if substring_list[i] == substring_list[j]:
                    pass
                elif substring_list[i] in substring_list[j]:
                    tuple = (substring_list[i], substring_list[j])
                    includes_list.append(tuple)
                elif substring_list[j] in substring_list[i]: 
                    tuple = (substring_list[j], substring_list[i])
                    includes_list.append(tuple)
        permutations = includes_list
        # print(permutations)
        G.add_edges_from(includes_list)
        # add the counts
        for node in G.nodes():
            G.nodes[node]['count'] = common_substrings_count[node]['count']
        print("Graph creation time : ", time.time() - graph_start_time)
        print("Calculating orthogonal r levels (refactoring levels)...")
        r_level_start_time = time.time()
        def filter_graphs(graph : nx.DiGraph):
            base_level_nodes = [x for x in graph.nodes() if len(nx.ancestors(graph, x)) == 0]
            levels = {
                0 : [],
                1 : base_level_nodes
            }
            # print(base_level_nodes)
            i = 0
            def make_tree(i, levels_dict):
                descendants = []
                if len(levels_dict[i]) > 0:
                    for base_node in levels_dict[i]:
                        imm_descs = []
                        for descendant in list(nx.descendants(graph, base_node)):
                            if len(list(nx.all_simple_paths(G, source=base_node, target=descendant))) == 1:
                                imm_descs.append(descendant)
                                
                        descendants.append(imm_descs)
                    levels_dict[i+1] = list(set(sum(descendants,start = [])))
                    # print("D : ", levels_dict)
                    return make_tree(i + 1, levels_dict)
                else:
                    return levels_dict
            levels = make_tree(1, levels)
            return levels
        if len(permutations) > 0:
            levels_dict = filter_graphs(G)
        else:
            levels_dict = {
                1 : list(common_substrings_count)
            }
        # print(nx.dfs_tree(G))
        # import matplotlib.pyplot as plt
        # nx.draw(G, with_labels=True)  # networkx draw()
        # plt.draw()
        refactorings = {}
        # add this as a control set
        refactorings[0] = {}
        for j, key in enumerate(levels_dict.keys()):
            refactoring = {}
            for i, item in enumerate(levels_dict[key]):
                refactoring[item] = i
                G.nodes[item]['r_level'] = key
            if len(refactoring.keys()) > 0 or key == 0:
                refactorings[key] = refactoring
        print("r_levels calculated in s:", time.time() - r_level_start_time)
        
        return G