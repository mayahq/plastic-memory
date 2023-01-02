# Plastic Memory Constructor

Demonstrating and visualizing hierarchial learning, automated abstraction &amp; plasticity.

Here is a [demo](https://www.youtube.com/watch?v=OOYJupTzCLI) of it in action, learning the first 40 names in the [makemore](https://github.com/karpathy/makemore) dataset.

## Get started

1. Install the requirements in `requirements.txt`. The basic implementation only requires `numpy` and `networkx` but you can optionally install `graphviz` to visualize graphs by following instructions [here](https://graphviz.readthedocs.io/en/stable/).
2. Follow the instructions in [`notebooks/01_introduction.ipynb`](notebooks/01_introduction.ipynb).

## About

Plastic memory is an experimental learning technique that learns by arranging information in a intelligence system as a hierarchical abstraction tree, and the process of learning as the dual mechanism of _non-local refactoring_ i.e. finding similarities and abstracting them out in the tree and _local synaptic firings_ i.e. strengthening / weakening pathways through the tree.

The process is analogous to the refactoring of a software tree, which starts off with low-level standard libraries, and gradually builds higher-level concepts or _abstractions_ by combining lower-level concepts together. The process exhibits the following properties :

## Features :

1. **Continuous learning** : No gradient descent - weights get continuously updated via local "synaptic firings" as it exposed to samples.
2. **Sample efficiency** : Dimensionality reduction via finding similarities/isomorphisms and refactoring them into a hierarchical tree
3. **Growing concept hierarchies** : Refactoring process creates abstract hierarchical concept libraries (demo [here](https://youtu.be/ONSVN4-Hua0) shows this for 32k first names from the makemore dataset, takes ~2 mins to train. note how the library grows from 1 r_level to 9 r_levels)
4. **Curriculum learning** : The order in which it is shown information plays a big role, simpler concepts shown first accelerate creating higher-level concepts later. As far as I see (subject to more testing), ideal information order is that where cumulative info entropy with every additional sample increases at a constant rate
5. **Emergent complexity** : Localized, so each neuron knows only about the ones surrounding it. Only the process to find isomorphisms + refactoring them is non-local, and the storage of information itself is non-local. It could be that the firing of a neural pathway is merely a query to global lookup table of states, which could explain how we perceive reality.
6. **Sparse networks** : During inference and learning, only certain neurons are fired along the decision tree, all are not active at all times
