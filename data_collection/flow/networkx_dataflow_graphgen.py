"""
NetworkX Dataflow graph generator
developed based on Pyverilog/examples/graphgen.py https://github.com/PyHDI/Pyverilog/blob/develop/examples/example_graphgen.py
"""

from __future__ import absolute_import
from __future__ import print_function
import sys
import os
import pygraphviz as pgv
import networkx as nx
import pickle as pkl
from optparse import OptionParser

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyverilog
from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer
from pyverilog.dataflow.optimizer import VerilogDataflowOptimizer
from pyverilog.dataflow.graphgen import VerilogGraphGenerator


def main():
    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    INFO = "Graph generator from dataflow"
    VERSION = pyverilog.__version__
    USAGE = "Usage: python example_graphgen.py -t TOPMODULE -s TARGETSIGNAL file ..."

    def showVersion():
        print(INFO)
        print(VERSION)
        print(USAGE)
        sys.exit()

    optparser = OptionParser()
    optparser.add_option("-v", "--version", action="store_true", dest="showversion",
                         default=False, help="Show the version")
    optparser.add_option("-I", "--include", dest="include", action="append",
                         default=[], help="Include path")
    optparser.add_option("-D", dest="define", action="append",
                         default=[], help="Macro Definition")
    optparser.add_option("-t", "--top", dest="topmodule",
                         default="TOP", help="Top module, Default=TOP")
    optparser.add_option("--nobind", action="store_true", dest="nobind",
                         default=False, help="No binding traversal, Default=False")
    optparser.add_option("--noreorder", action="store_true", dest="noreorder",
                         default=False, help="No reordering of binding dataflow, Default=False")
    optparser.add_option("-s", "--search", dest="searchtarget", action="append",
                         default=[], help="Search Target Signal")
    optparser.add_option("-o", "--output", dest="outputfile",
                         default="out.png", help="Graph file name, Default=out.png")
    optparser.add_option("--identical", action="store_true", dest="identical",
                         default=False, help="# Identical Laef, Default=False")
    optparser.add_option("--walk", action="store_true", dest="walk",
                         default=False, help="Walk contineous signals, Default=False")
    optparser.add_option("--step", dest="step", type='int',
                         default=1, help="# Search Steps, Default=1")
    optparser.add_option("--reorder", action="store_true", dest="reorder",
                         default=False, help="Reorder the contineous tree, Default=False")
    optparser.add_option("--delay", action="store_true", dest="delay",
                         default=False, help="Inset Delay Node to walk Regs, Default=False")
    (options, args) = optparser.parse_args()

    filelist = args
    if options.showversion:
        showVersion()

    for f in filelist:
        if not os.path.exists(f):
            raise IOError("file not found: " + f)

    if len(filelist) == 0:
        showVersion()

    analyzer = VerilogDataflowAnalyzer(filelist, options.topmodule,
                                       noreorder=options.noreorder,
                                       nobind=options.nobind,
                                       preprocess_include=options.include,
                                       preprocess_define=options.define)
    analyzer.generate()

    directives = analyzer.get_directives()
    terms = analyzer.getTerms()
    binddict = analyzer.getBinddict()

    optimizer = VerilogDataflowOptimizer(terms, binddict)

    optimizer.resolveConstant()
    resolved_terms = optimizer.getResolvedTerms()
    resolved_binddict = optimizer.getResolvedBinddict()
    constlist = optimizer.getConstlist()

    G = {}
    signals = []
    for signal, _ in resolved_binddict.items():
        graphgen = VerilogGraphGenerator(options.topmodule, terms, binddict,
                                         resolved_terms, resolved_binddict, constlist, options.outputfile)
        graphgen.generate(str(signal), walk=options.walk, identical=options.identical,
                          step=options.step, do_reorder=options.reorder, delay=options.delay)
        graph = nx.nx_agraph.from_agraph(graphgen.graph)
        delete_color_attribute(graph)
        G[signal] = graph
        signals.append(signal)

    sys.stdout = original_stdout
    pkl.dump(create_module_graph(signals, G), sys.stdout.buffer)

def delete_color_attribute(graph):
    for node in graph.nodes:
        if 'color' in graph.nodes[node]:
            del graph.nodes[node]['color']
    for u, v in graph.edges:
        if 'color' in graph.edges[u, v]:
            del graph.edges[u, v]['color']

def create_module_graph(signals, G):
    # Initialize unique ID counters
    op_id_counter = 0
    signal_id_counter = 0

    # Function to process each signal graph
    def process_signal_graph(G_signal, signal_name):
        signal_name = repr(signal_name)
        nonlocal op_id_counter, signal_id_counter
        # Create a copy to avoid modifying the original graph
        G_signal = G_signal.copy()

        # Mapping from old node names to new node names
        node_mapping = {}

        # Step 1: Update the node whose label is the current signal
        current_signal_node = None
        for node, data in G_signal.nodes(data=True):
            if data.get('label') == signal_name:
                current_signal_node = node
                break

        if current_signal_node is not None:
            new_node_name = signal_name.replace(".", "_")
            data = G_signal.nodes[current_signal_node]
            data.pop('label', None)

            if new_node_name in G_signal and new_node_name != current_signal_node:
                # Merge nodes if the new node name already exists
                merge_nodes_two(G_signal, current_signal_node, new_node_name)
                # No need to update node_mapping since new_node_name already exists
                node_mapping[new_node_name] = f"signal_{signal_id_counter}"
                signal_id_counter += 1
                data['label'] = new_node_name
            else:
                node_mapping[current_signal_node] = f"signal_{signal_id_counter}"
                signal_id_counter += 1
                data['label'] = new_node_name
        else:
            print(f"Warning: Node with label '{signal_name}' not found in the graph.")
            return None

        # Step 2: Rename nodes with labels to "op_n"
        for node, data in G_signal.nodes(data=True):
            if node in node_mapping:
                continue  # Skip nodes already renamed

            if 'label' in data:
                new_node_name = f'op_{op_id_counter}'
                op_id_counter += 1
                node_mapping[node] = new_node_name
            # Nodes without labels will be handled in Step 3

        # Step 3: Rename nodes without labels to "signal_n"
        for node, data in G_signal.nodes(data=True):
            if node in node_mapping:
                continue  # Node already renamed

            new_node_name = f'signal_{signal_id_counter}'
            signal_id_counter += 1
            node_mapping[node] = new_node_name
            # Use original node name as label
            data['label'] = node.replace(".", "_")

        # Apply the node renamings
        G_signal = nx.relabel_nodes(G_signal, node_mapping)

        return G_signal

    def merge_nodes_two(G, source, target):
        """Merge 'source' node into 'target' node in graph G."""
        # Redirect edges from source to target
        in_edges = list(G.in_edges(source, data=True))
        out_edges = list(G.out_edges(source, data=True))

        # Add edges to the target node
        for u, v, data in in_edges:
            if v == source:
                v = target
            G.add_edge(u, target, **data)
        for u, v, data in out_edges:
            if u == source:
                u = target
            G.add_edge(target, v, **data)

        # Remove the source node
        G.remove_node(source)

    def merge_nodes(G, nodes_to_merge):
        """Merge nodes in 'nodes_to_merge' list into a single node."""
        if not nodes_to_merge:
            return

        # The first node in the list will be the main node
        main_node = nodes_to_merge[0]
        for duplicate_node in nodes_to_merge[1:]:
            # Redirect edges from duplicate_node to main_node
            in_edges = list(G.in_edges(duplicate_node, data=True))
            out_edges = list(G.out_edges(duplicate_node, data=True))

            # Add edges to the main node
            for u, v, data in in_edges:
                if v == duplicate_node:
                    v = main_node
                G.add_edge(u, main_node, **data)
            for u, v, data in out_edges:
                if u == duplicate_node:
                    u = main_node
                G.add_edge(main_node, v, **data)

            # Remove the duplicate node
            G.remove_node(duplicate_node)

    # Process each signal graph and collect transformed graphs
    transformed_graphs = []
    for signal in signals:
        # Apply the transformations
        G_transformed = process_signal_graph(G[signal], signal)
        if G_transformed is None:
            return None
        transformed_graphs.append(G_transformed)

    # Step 4: Merge all transformed signal graphs into one module graph
    module_graph = nx.DiGraph()
    for G_transformed in transformed_graphs:
        module_graph = nx.compose(module_graph, G_transformed)

    # Merge signal nodes that have the same label
    label_to_nodes = {}
    for node, data in module_graph.nodes(data=True):
        if node.startswith("op_"):
            continue
        label = data.get('label')
        if label:
            label_to_nodes.setdefault(label, []).append(node)

    # Merge nodes with the same label
    for nodes in label_to_nodes.values():
        if len(nodes) > 1:
            merge_nodes(module_graph, nodes)

    return module_graph

if __name__ == '__main__':
    main()