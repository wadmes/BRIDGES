"""
data generation step 3
This script performs Verilog RTL analysis using PyVerilog tools. It analyzes Verilog files for syntax correctness, 
extracts module-level dataflow information, and generates dataflow graphs for each module. Results, including 
successes and failures for syntax and dataflow analysis, are logged and saved incrementally for fault tolerance.
"""
import re
import os
import typer
import subprocess
import pickle as pkl
from tqdm import tqdm
from pathlib import Path

def analyze_syntax(
    verilog_file: Path,
    output_dir: Path,
    parser: Path,
    analysis_time_limit: int,
) -> int:
    try:
        result = subprocess.run(
            ["python3", str(parser), str(verilog_file)],
            capture_output=True,
            text=True,
            check=True,
            timeout=analysis_time_limit,
        )
        if not result.stdout:
            return 3
        with open(output_dir / "syntax.txt", 'w') as f:
            f.write(result.stdout)
        return 0 
    except subprocess.TimeoutExpired:
        return 1
    except subprocess.SubprocessError:
        return 2
    
def analyze_dataflow(
    module: str,
    verilog_file: Path,
    output_dir: Path,
    dataflow_analyzer: Path,
    analysis_time_limit: int,
) -> int:
    try:
        result = subprocess.run(
            ['python3', str(dataflow_analyzer), '-t', module, str(verilog_file)],
            capture_output=True,
            text=True,
            check=True,
            timeout=analysis_time_limit,
        )
        if not result.stdout:
            return 3
        with open(output_dir / f"{module}.txt", 'w') as f:
            f.write(result.stdout)
        return 0
    except subprocess.TimeoutExpired:
        return 1
    except subprocess.SubprocessError:
        return 2

def generate_dataflow_graph(
    module: str,
    verilog_file: Path,
    output_dir: Path,
    dataflow_graph_generator: Path,
    analysis_time_limit: int,
) -> int:
    try:
        result = subprocess.run(
            ["python3", str(dataflow_graph_generator), "-t", module, str(verilog_file)],
            capture_output=True,
            check=True,
            timeout=analysis_time_limit,
        )
        with open(output_dir / f"{module}.pkl", 'wb') as f:
            pkl.dump(pkl.loads(result.stdout), f)
        return 0
    except subprocess.TimeoutExpired:
        os.remove(output_dir / f"{module}.txt")
        return 1
    except subprocess.SubprocessError:
        os.remove(output_dir / f"{module}.txt")
        return 2
    
def analyze_rtl(
    idx: int,
    verilog_file: Path,
    output_dir: Path,
    parser: Path,
    dataflow_analyzer: Path,
    dataflow_graph_generator: Path,
    analysis_time_limit: int,
) -> tuple[bool, list, list]:
    output_dir.mkdir(parents=True, exist_ok=True)
    syntax_status = analyze_syntax(verilog_file, output_dir, parser, analysis_time_limit)
    with open(verilog_file, 'r') as f:
        code = f.read()
    modules = re.findall(r'(?:^module\s+)(\w+)(?:\s*#\s*\(.*?\))?\s*(?:\(|;)', code, re.MULTILINE)
    dataflow_success, dataflow_fail = [], []
    for module in modules:
        if analyze_dataflow(module, verilog_file, output_dir, dataflow_analyzer, analysis_time_limit) != 0:
            dataflow_fail.append((idx, module))
            continue
        if generate_dataflow_graph(module, verilog_file, output_dir, dataflow_graph_generator, analysis_time_limit) != 0:
            dataflow_fail.append((idx, module))
            continue
        dataflow_success.append((idx, module))
    return (syntax_status == 0, dataflow_success, dataflow_fail)

def main(
    num_data: int = typer.Option(..., help="Number of data points."),
    synthesis_dir: Path = typer.Option(..., help="Synthesis result directory."),
    data_dir: Path = typer.Option(..., help="Base directory containing synthesis results and output folders."),
    parser: Path = typer.Option(..., help="PyVerilog syntax analyzer script."),
    dataflow_analyzer: Path = typer.Option(..., help="PyVerilog dataflow analyzer script."),
    dataflow_graph_generator: Path = typer.Option(..., help="PyVerilog dataflow graph generator."),
    analysis_time_limit: int = typer.Option(30, help="PyVerilog analysis time limit in sec.")
):
    output_dir = data_dir / "pyverilog"
    output_dir.mkdir(parents=True, exist_ok=True)
    syntax_success, syntax_fail, dataflow_success, dataflow_fail = [], [], [], []
    for idx in tqdm(range(num_data)):
        result = analyze_rtl(idx, synthesis_dir / f"{idx}" / "rtl.v", output_dir / f"{idx}", parser, dataflow_analyzer, dataflow_graph_generator, analysis_time_limit)
        if result[0]:
            syntax_success.append(idx)
        else:
            syntax_fail.append(idx)
        dataflow_success += result[1]
        dataflow_fail += result[2]
        with open(output_dir / "pyverilog_analysis.pkl", 'wb') as f:
            pkl.dump((syntax_success, syntax_fail, dataflow_success, dataflow_fail), f)

if __name__ == "__main__":
    typer.run(main)