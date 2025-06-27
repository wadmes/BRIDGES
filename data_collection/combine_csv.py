"""
combine the two netlist CSV files of RTLCoder and MGVerilog
RTLCoder keeps the original netlist_id
MGverilog uses netlist_id + 1000000 as the new id
"""
import pandas as pd

MGverilog = pd.read_csv('MGVerilog.csv')
RTLCoder = pd.read_csv('RTLCoder.csv')
MGverilog['id'] = MGverilog['id'] + 1000000
result = pd.concat([MGverilog, RTLCoder], ignore_index=True)
result.to_csv('BRIDGES.csv', index=False)