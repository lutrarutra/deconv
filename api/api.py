from core import deconV as dV

def load_files(sc_path, bulk_path):
    return dV.read_data(sc_path), dV.read_data(bulk_path, is_bulk=True)