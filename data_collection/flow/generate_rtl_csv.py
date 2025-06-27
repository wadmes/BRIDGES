"""
data generation step 7
collect all metadata related to rtl and generate a csv file
"""
import os
import csv
import json
import typer
import pickle as pkl
from pathlib import Path
from tqdm import tqdm

def get_file_length(file_path):
    with open(file_path, 'r') as file:
        return len(file.read())

def extract_dataflow_graph_info(G):
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    out_degree_distribution = {}
    in_degree_distribution = {}
    type_distribution = {}
    for node, data in G.nodes(data=True):
        out_degree = G.out_degree(node)
        in_degree = G.in_degree(node)
        out_degree_distribution[out_degree] = out_degree_distribution.get(out_degree, 0) + 1
        in_degree_distribution[in_degree] = in_degree_distribution.get(in_degree, 0) + 1
        if node.startswith("op_"):
            node_type = data['label']
        else:
            assert(node.startswith("signal_"))
            node_type = 'Signal'
        type_distribution[node_type] = type_distribution.get(node_type, 0) + 1
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "out_degree_distribution": out_degree_distribution,
        "in_degree_distribution": in_degree_distribution,
        "type_distribution": type_distribution,
    }

def main(
    data_dir: Path = typer.Option(..., help="Base directory where all relevent outputs are stored."),
):
    with open(data_dir / "rtl_data/rtl.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        header = [
            "rtl_id", "module_number", "module_name_list", "dataflow_status", "synthesis_status",
            "GPT_4o_label", "Llama3_70b_label", "consistent_label", "verilog_file_length", "#dataflow_node", "#dataflow_edge",
            "dataflow_node_type_distribution", "dataflow_node_in_degree_distribution", "dataflow_node_out_degree_distribution"
        ]
        writer.writerow(header)
        with open(data_dir / "rtl_data/rtl.json", "r") as file:
            data = json.load(file)
        for idx, v in tqdm(data.items()):
            rtl_id = int(idx)
            module_names = list(v['name_mapping'].keys())
            dataflow_status = 1 if v['dataflow_status'] else 0
            synthesis_status = 1 if v['synthesis_status'] else 0
            verilog_file_length = get_file_length(data_dir / f"synthesis/{idx}/rtl.sv")
            total_nodes = 0
            total_edges = 0
            total_type_distribution = {}
            total_in_degree_distribution = {}
            total_out_degree_distribution = {}
            for module in module_names:
                if os.path.isfile(data_dir / f"rtl_data/dataflow_graph/{rtl_id}_{module}.pkl"):
                    with open(data_dir / f"rtl_data/dataflow_graph/{rtl_id}_{module}.pkl", "rb") as graph_file:
                        graph = pkl.load(graph_file)
                        graph_info = extract_dataflow_graph_info(graph)
                        total_nodes += graph_info["node_count"]
                        total_edges += graph_info["edge_count"]
                        for key, value in graph_info["type_distribution"].items():
                            total_type_distribution[key] = total_type_distribution.get(key, 0) + value
                        for key, value in graph_info["in_degree_distribution"].items():
                            total_in_degree_distribution[key] = total_in_degree_distribution.get(key, 0) + value
                        for key, value in graph_info["out_degree_distribution"].items():
                            total_out_degree_distribution[key] = total_out_degree_distribution.get(key, 0) + value
            row = [
                rtl_id, len(module_names), module_names, dataflow_status, synthesis_status,
                v['GPT_4o_label'], v['Llama3_70b_label'], v['consistent_label'], verilog_file_length,
                total_nodes, total_edges, total_type_distribution, total_in_degree_distribution, total_out_degree_distribution
            ]
            writer.writerow(row)

if __name__ == "__main__":
    typer.run(main)