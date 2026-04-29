"""
Dataset for DeepPrime PE6 inference.
"""
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Maps PE type name → read-count column names used for filtering
AVG_PAIR = {
    "PEmax": ("HEK-M-1-7D-UAR+total_read_counts", "HEK-M2-1-7D-UAR+total_read_counts"),
    "PEmaxdRNaseH": ("HEK-M-2-7D-UAR+total_read_counts", "HEK-M2-2-7D-UAR+total_read_counts"),
    "PE6a(+PEmaxCas9)": ("HEK-M-3-7D-UAR+total_read_counts", "HEK-M2-3-7D-UAR+total_read_counts"),
    "PE6b(+PEmaxCas9)": ("HEK-M-4-7D-UAR+total_read_counts", "HEK-M2-4-7D-UAR+total_read_counts"),
    "PE6c(+PEmaxCas9)": ("HEK-M-5-7D-UAR+total_read_counts", "HEK-M2-5-7D-UAR+total_read_counts"),
    "PE6d(+PEmaxCas9)": ("HEK-M-6-7D-UAR+total_read_counts", "HEK-M2-6-7D-UAR+total_read_counts"),
    "PE6e(+dRNaseH)": ("HEK-M-7-7D-UAR+total_read_counts",),
    "PE6f(+dRNaseH)": ("HEK-M-8-7D-UAR+total_read_counts",),
    "PE6g(+dRNaseH)": ("HEK-M-9-7D-UAR+total_read_counts",),
}

BIOFEATURE_COLS = [
    "PBS_len", "RTT_len", "RT-PBS_len", "Edit_pos", "Edit_len", "RHA_len",
    "type_sub", "type_ins", "type_del",
    "Tm1_PBS", "Tm2_RTT_cTarget_sameLength", "Tm3_RTT_cTarget_replaced",
    "Tm4_cDNA_PAM-oppositeTarget", "Tm5_RTT_cDNA", "deltaTm_Tm4-Tm2",
    "GC_count_PBS", "GC_count_RTT", "GC_count_RT-PBS",
    "GC_contents_PBS", "GC_contents_RTT", "GC_contents_RT-PBS",
    "MFE_RT-PBS-polyT", "MFE_Spacer", "DeepSpCas9_score",
]


class PE6DeepPrimeDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        datafilter: Dict[str, List[str]],
        read_count_filter: int = 0,
        norm_mean: pd.Series = None,
        norm_std: pd.Series = None,
    ):
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        assert len(datafilter["PE_types"]) < 2, "Only one PE type is supported"
        pe_type = datafilter["PE_types"][0]

        # No row filtering — use all rows
        filtered_data = data.copy()

        label_cols = datafilter["PE_types"]
        self.labels = torch.from_numpy(
            filtered_data[[*label_cols, "type_sub", "type_ins", "type_del"]].values
        )

        self.data = filtered_data.drop(columns=label_cols)
        self.annotations = filtered_data["ID"]

        # One-hot encode sequences
        self.data["Target_encoded"] = self.data["Target"].apply(self._one_hot)
        self.data["Masked_EditSeq_encoded"] = self.data["Masked_EditSeq"].apply(self._one_hot)

        self.genetic_features = np.stack(
            self.data[["Target_encoded", "Masked_EditSeq_encoded"]]
            .apply(lambda x: np.stack([x["Target_encoded"], x["Masked_EditSeq_encoded"]], axis=0), axis=1)
            .to_numpy(),
            axis=0,
        )
        self.genetic_features = 2 * self.genetic_features - 1  # scale to [-1, 1]

        self.biofeatures = self._normalize_biofeatures(self.data[BIOFEATURE_COLS])

        self._len = len(self.data)
        del self.data

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        g = torch.from_numpy(self.genetic_features[idx].copy()).permute(2, 0, 1)
        b = torch.from_numpy(self.biofeatures[idx].copy())
        label = self.labels[idx].clone().detach()
        annot = self.annotations.iloc[idx]
        return ((g, b), label), annot

    def _one_hot(self, sequence: str):
        sequence = str(sequence).upper()
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "X": 4}
        map_seq = [mapping[c] for c in sequence]
        arr = np.eye(5)[map_seq]
        return np.delete(arr, -1, axis=1)

    def _normalize_biofeatures(self, biofeatures: pd.DataFrame) -> np.ndarray:
        if self.norm_mean is not None:
            is_scalar = isinstance(self.norm_mean, (float, int))
            if isinstance(self.norm_mean, pd.Series) and len(self.norm_mean) == 1:
                self.norm_mean = self.norm_mean.iloc[0]
                if self.norm_std is not None and isinstance(self.norm_std, pd.Series):
                    self.norm_std = self.norm_std.iloc[0]
                is_scalar = True

            if not is_scalar and isinstance(self.norm_mean, (pd.Series, pd.DataFrame)):
                # Rename to match DeepPrime original feature names used in norm_mean index
                rename_map = {
                    "PBS_len": "PBSlen", "RTT_len": "RTlen", "RT-PBS_len": "RT-PBSlen",
                    "Tm1_PBS": "Tm1", "Tm2_RTT_cTarget_sameLength": "Tm2",
                    "Tm3_RTT_cTarget_replaced": "Tm2new", "Tm5_RTT_cDNA": "Tm3",
                    "Tm4_cDNA_PAM-oppositeTarget": "Tm4", "deltaTm_Tm4-Tm2": "TmD",
                    "GC_count_PBS": "nGCcnt1", "GC_count_RTT": "nGCcnt2",
                    "GC_count_RT-PBS": "nGCcnt3", "GC_contents_PBS": "fGCcont1",
                    "GC_contents_RTT": "fGCcont2", "GC_contents_RT-PBS": "fGCcont3",
                    "MFE_RT-PBS-polyT": "MFE3", "MFE_Spacer": "MFE4",
                }
                biofeatures = biofeatures.rename(columns=rename_map)
                biofeatures = biofeatures[self.norm_mean.index]

        norm_mean = self.norm_mean if self.norm_mean is not None else biofeatures.mean()
        norm_std = self.norm_std if self.norm_std is not None else biofeatures.std()
        norm_bf = (biofeatures - norm_mean) / norm_std
        return norm_bf.fillna(0).values
