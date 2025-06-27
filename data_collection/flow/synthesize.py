"""
data generation step 2
This script performs RTL synthesis in parallel using a custom iterator to process RTL datasets. It dynamically loads 
a user-defined iterator function to parse RTL data, writes RTL code and related information to output directories, 
and runs synthesis efforts with configurable time limits. Results (success, timeout, or failure) are logged and saved 
incrementally for fault tolerance. The synthesis tasks are distributed across multiple processes for efficiency.
"""
import os
import typer
import subprocess
import pickle as pkl
import importlib.util
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
from itertools import product
from typing import Tuple, Callable

DEFAULT_SYNTHESIS_TIMELIMIT = 3600
DEFAULT_NUM_PROCESSES = 1

def load_custom_iterator(iterator_file: Path) -> Callable:
    """Dynamically load a iterator function from a given file, function name should be the same as the file name"""
    func_name = iterator_file.stem
    spec = importlib.util.spec_from_file_location("custom_iterator_module", iterator_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, func_name, None)
    if not callable(func):
        raise ValueError(f"{func_name} is not a callable function in {iterator_file}.")
    return func

def log(log_path: Path, msg: str) -> None:
    """Utility function to log messages to a file."""
    with open(log_path, 'a') as f:
        f.write(msg)
        f.write("\n\n")

def synthesize_rtl(rtl_id: int, output_dir: Path, synthesis_timelimit: int, synthesis_lib: Path) -> Tuple[int, int]:
    """Synthesize the RTL and handle different synthesis efforts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "synthesis_log.txt"

    efforts = ("low", "medium", "high")
    launch_cmds = ("genus", "-no_gui", "-abort_on_error", "-log", "genus", "-execute")
    syn_cmds = (
        f"set_db / .library {synthesis_lib}",
        "read_hdl -v ../../rtl.sv",
        "elaborate",
        "bitblast_all_ports",
        'update_names -hnet -restricted {[ ] .} -replace_str "_"',
        'update_names -inst -restricted {[ ] .} -replace_str "_"',
        "set_db hdl_bus_wire_naming_style %s__%d",
        "syn_generic", "syn_map", "syn_opt",
        "ungroup -all -flatten -force",
        'redirect syn.v "write_hdl -generic"',
        "report_power > power_report.txt",
        "report_area -summary > area_report.txt",
        "exit",
    )

    for generic_effort, map_effort, opt_effort in product(efforts, repeat=3):
        working_dir = output_dir / f"{generic_effort}_{map_effort}_{opt_effort}"
        working_dir.mkdir(exist_ok=True)
        effort_cmds = (
            f"set_db syn_generic_effort {generic_effort}",
            f"set_db syn_map_effort {map_effort}",
            f"set_db syn_opt_effort {opt_effort}",
        )
        genus_cmds = effort_cmds + syn_cmds

        with open(working_dir / "cmd.tcl", "w") as f:
            f.write("\n".join(genus_cmds))

        try:
            subprocess.run(
                launch_cmds + (";\n".join(genus_cmds),),
                cwd=working_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                timeout=synthesis_timelimit,
            )
            log(log_file, f"Effort ({generic_effort}, {map_effort}, {opt_effort}) synthesized successfully!")
        except subprocess.TimeoutExpired:
            log(log_file, f"Effort ({generic_effort}, {map_effort}, {opt_effort}) timed out. Aborting...")
            return (rtl_id, 1)
        except subprocess.CalledProcessError:
            log(log_file, f"Effort ({generic_effort}, {map_effort}, {opt_effort}) failed. Aborting...")
            return (rtl_id, 2)
        except Exception as e:
            log(log_file, f"Unexpected error: {e}. Aborting...")
            return (rtl_id, 3)

    return (rtl_id, 0)

def process_rtl(args: Tuple[int, str, str, Path, int, Path]) -> Tuple[int, int]:
    idx, prompt, code, output_dir, synthesis_timelimit, synthesis_lib, prompt_type = args

    rtl_output_dir = output_dir / str(idx)
    rtl_output_dir.mkdir(parents=True, exist_ok=True)

    if prompt_type:
        with open(rtl_output_dir / "instruction.txt", 'w') as f:
            f.write(prompt)
    else:
        with open(rtl_output_dir / "description.txt", 'w') as f:
            f.write(prompt)
    with open(rtl_output_dir / "rtl.sv", 'w') as f:
        f.write(code)
    
    return synthesize_rtl(idx, rtl_output_dir / "syn", synthesis_timelimit, synthesis_lib)

def main(
    iterator_file: Path = typer.Option(..., help="Path to the custom iterator function for loading dataset."),
    dataset_path: Path = typer.Option(..., help="Path to the RTL dataset."),
    data_dir: Path = typer.Option(..., help="Base directory where all relevent outputs, including synthesis results, will be saved."),
    synthesis_lib: Path = typer.Option(..., help="Path to the Genus synthesis standard cell lib."),
    synthesis_timelimit: int = typer.Option(DEFAULT_SYNTHESIS_TIMELIMIT, help="Time limit (s) for each synthesis process."),
    num_processes: int = typer.Option(DEFAULT_NUM_PROCESSES, help="Number of parallel processes."),
    prompt_type: bool = typer.Option(True, "--instruction/--description", help="Whether the RTL's related prompt is instruction (--instruction) or description (--description).")
):
    """
    Main function for RTL synthesis using multiprocessing.

    This function dynamically loads a user-defined iterator function to parse RTL data, distributes
    the synthesis tasks across multiple processes, and logs results periodically. It ensures fault-tolerant
    handling by saving intermediate results and reporting synthesis success, timeout, or failure statistics.

    Args:
        iterator_file (Path): Path to the custom iterator function for loading dataset.
        dataset_path (Path): Path to the RTL dataset.
        data_dir (Path): Base directory where all relevent outputs, including synthesis results, will be saved.
        synthesis_lib (Path): Path to the Genus synthesis standard cell lib.
        synthesis_timelimit (int): Time limit (s) for each synthesis process.
        num_processes (int): Number of parallel processes.
        prompt_type (bool): Flag indicating if the RTL's related information is an instruction or description.
    """
    output_dir = data_dir / 'synthesis'
    output_dir.mkdir(parents=True, exist_ok=True)

    rtl_itr_func = load_custom_iterator(iterator_file)
    rtl_itr = list(rtl_itr_func(dataset_path))

    lock = mp.Lock()
    results = []
    tmp_results_pkl_log = output_dir / "tmp_results_pkl_log.pkl"
    with mp.Pool(num_processes) as pool:
        result_itr = pool.imap_unordered(
            process_rtl, 
            ((idx, prompt, code, output_dir, synthesis_timelimit, synthesis_lib, prompt_type) for idx, prompt, code in rtl_itr)
        )
        for idx, result in enumerate(tqdm(result_itr, desc="Synthesizing RTLs", total=len(rtl_itr))):
            results.append(result)
            if (idx + 1) % 100 == 0:
                with lock:
                    with open(tmp_results_pkl_log, 'wb') as f:
                        pkl.dump(results, f)
        with lock:
            with open(tmp_results_pkl_log, 'wb') as f:
                pkl.dump(results, f)

    success, timeout, fail = [], [], []
    for idx, result_code in results:
        if result_code == 0:
            success.append(idx)
        elif result_code == 1:
            timeout.append(idx)
        else:
            fail.append(idx)

    print(f"Success: {len(success)} RTLs")
    print(f"Timeout: {len(timeout)} RTLs")
    print(f"Failure: {len(fail)} RTLs")

    with open(output_dir / "synthesis_result.pkl", "wb") as f:
        pkl.dump((success, timeout, fail), f)

    with lock:
        if tmp_results_pkl_log.exists():
            os.remove(tmp_results_pkl_log)

if __name__ == "__main__":
    typer.run(main)
