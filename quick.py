import pickle

with open('/storage/yangzou/gf/vlsi_data/tmp/design_info.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)