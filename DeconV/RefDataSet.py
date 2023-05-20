import torch

class RefDataSet(torch.utils.data.Dataset):
    def __init__(self, adata, label_key, layer="counts", device="cpu") -> None:
        super().__init__()
        self.sc_counts = torch.tensor(adata.layers[layer], dtype=torch.float32, device=device).round()
        self.labels = torch.tensor(adata.obs[label_key].cat.codes.values, dtype=torch.long, device=device)

    def __len__(self):
        return len(self.sc_counts)

    def __getitem__(self, idx):
        return self.sc_counts[idx], self.labels[idx], idx