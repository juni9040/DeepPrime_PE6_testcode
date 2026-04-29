"""
Bio-feature preprocessing for DeepPrime PE6.

Computes the 24 bio-features needed by the model from raw pegRNA design parameters:
  - Tm1~Tm5 (melting temperatures)
  - GC counts / contents
  - MFE (minimum free energy, RNA secondary structure)
  - DeepSpCas9 score
  - 74nt target sequences (Target, Masked_EditSeq)
  - Edit length/position/type features

Required packages:
  pip install biopython  genet  ViennaRNA
  (ViennaRNA provides the `RNA` Python module)

Input CSV required columns:
  REF_ID, WideTargetSequence, OligoSequence_fixed_length, Guide,
  PBS, RTT, Edit_type, Edit_len (or Edit length), Edit_pos (or Edit position),
  PBS_len (or PBS_length), RTT_len (or RT_length), leading G,
  <pe_type_column> (e.g. PEmaxdRNaseH)
"""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
import re

import pandas as pd
import numpy as np
from Bio.Seq import Seq, reverse_complement, transcribe
from Bio.SeqUtils import MeltingTemp as mt
from Bio.SeqUtils import gc_fraction as gc


# ---------------------------------------------------------------------------
# Dataclasses (mirrors src/data/components/data_model/biofeatures.py)
# ---------------------------------------------------------------------------

@dataclass
class TmSequences:
    Tm1_PBS_seq: str
    Tm2_RTT_cTarget_sameLength_seq: str
    Tm3_RTT_cTarget_replaced_seq: str
    Tm4_cDNA_PAM_oppositeTarget_seq: Optional[Tuple[str, str]]
    Tm5_RTT_cDNA_seq: str


@dataclass
class TmData:
    Tm1_PBS: float
    Tm2_RTT_cTarget_sameLength: float
    Tm3_RTT_cTarget_replaced: float
    Tm4_cDNA_PAM_oppositeTarget: float
    Tm5_RTT_cDNA: float
    deltaTm_Tm4_Tm2: float


@dataclass
class PegRNAExtensionData:
    GC_count_PBS: int
    GC_count_RTT: int
    GC_count_RT_PBS: int
    GC_contents_PBS: float
    GC_contents_RTT: float
    GC_contents_RT_PBS: float


@dataclass
class MFEData:
    MFE_RT_PBS_polyT: float
    MFE_Spacer: float


@dataclass
class EditTypeClass:
    type_sub: int
    type_ins: int
    type_del: int


@dataclass
class TargetSequenceData:
    wild_type_sequence: str
    deepspcas9_guide_30: str
    prime_edited_sequence: str
    edit_position: int


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UPSTREAM_LENGTH_TO_NICK = 21
DOWNSTREAM_IDX_TO_NICK = 53

# Final column name mapping (from internal names → model-expected names)
RENAME_MAP_FINAL = {
    "RT_PBS": "RT-PBS",
    "PBS_length": "PBS_len",
    "RT_length": "RTT_len",
    "RT_PBS_len": "RT-PBS_len",
    "Edit position": "Edit_pos",
    "Edit length": "Edit_len",
    "wild_type_sequence": "Target",
    "prime_edited_sequence": "Masked_EditSeq",
    "Tm1_PBS": "Tm1_PBS",
    "Tm2_RTT_cTarget_sameLength": "Tm2_RTT_cTarget_sameLength",
    "Tm3_RTT_cTarget_replaced": "Tm3_RTT_cTarget_replaced",
    "Tm4_cDNA_PAM_oppositeTarget": "Tm4_cDNA_PAM-oppositeTarget",
    "Tm5_RTT_cDNA": "Tm5_RTT_cDNA",
    "deltaTm_Tm4_Tm2": "deltaTm_Tm4-Tm2",
    "GC_count_PBS": "GC_count_PBS",
    "GC_count_RTT": "GC_count_RTT",
    "GC_count_RT_PBS": "GC_count_RT-PBS",
    "GC_contents_PBS": "GC_contents_PBS",
    "GC_contents_RTT": "GC_contents_RTT",
    "GC_contents_RT_PBS": "GC_contents_RT-PBS",
    "MFE_RT_PBS_polyT": "MFE_RT-PBS-polyT",
    "MFE_Spacer": "MFE_Spacer",
    "REF_ID": "ID",
}


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

def find_guide_indices(wide_seq: str, guide: str) -> Tuple[int, int]:
    results = [(m.start(), m.end() - 1) for m in re.finditer(re.escape(guide), wide_seq)]
    assert len(results) == 1, f"Expected exactly one match of guide in WideTargetSequence, got {len(results)}"
    return results[0]


def determine_seqs(alt_type, alt_len, wt_seq, pbs_seq, rt_seq, nick_index) -> TmSequences:
    tm1_pbs = transcribe(pbs_seq)
    tm2_rtt_ctarget = wt_seq[nick_index: nick_index + len(rt_seq)]

    if alt_type.lower().startswith("sub"):
        tm2new = tm2_rtt_ctarget
        tm3_anti = reverse_complement(tm2_rtt_ctarget)
    elif alt_type.lower().startswith("ins"):
        tm2new = wt_seq[nick_index: nick_index + len(rt_seq) - alt_len]
        tm3_anti = reverse_complement(tm2new)
    elif alt_type.lower().startswith("del"):
        tm2new = wt_seq[nick_index: nick_index + len(rt_seq) + alt_len]
        tm3_anti = reverse_complement(tm2new)
    else:
        raise ValueError(f"Invalid edit type: {alt_type}")

    return TmSequences(
        Tm1_PBS_seq=tm1_pbs,
        Tm2_RTT_cTarget_sameLength_seq=tm2_rtt_ctarget,
        Tm3_RTT_cTarget_replaced_seq=tm2new,
        Tm4_cDNA_PAM_oppositeTarget_seq=(reverse_complement(rt_seq), tm3_anti),
        Tm5_RTT_cDNA_seq=transcribe(rt_seq),
    )


def determine_tm(seq_data: TmSequences) -> TmData:
    def _tm(seq, nn_table):
        return mt.Tm_NN(seq=Seq(seq), nn_table=nn_table)

    def _tm4(seq_pairs):
        seq1, seq2 = seq_pairs
        fTm4 = 0.0
        for s1, s2 in zip(str(seq1), str(seq2)):
            try:
                fTm4 = mt.Tm_NN(seq=Seq(s1), c_seq=Seq(s2), nn_table=mt.DNA_NN3)
            except ValueError:
                fTm4 = 0
        return fTm4

    tm1 = _tm(seq_data.Tm1_PBS_seq, mt.R_DNA_NN1)
    tm2 = _tm(seq_data.Tm2_RTT_cTarget_sameLength_seq, mt.DNA_NN3)
    tm3 = _tm(seq_data.Tm3_RTT_cTarget_replaced_seq, mt.DNA_NN3)
    try:
        tm4 = _tm4(seq_data.Tm4_cDNA_PAM_oppositeTarget_seq) if seq_data.Tm4_cDNA_PAM_oppositeTarget_seq else 0.0
    except ValueError:
        tm4 = 0.0
    tm5 = _tm(seq_data.Tm5_RTT_cDNA_seq, mt.R_DNA_NN1)

    return TmData(
        Tm1_PBS=tm1, Tm2_RTT_cTarget_sameLength=tm2, Tm3_RTT_cTarget_replaced=tm3,
        Tm4_cDNA_PAM_oppositeTarget=tm4, Tm5_RTT_cDNA=tm5, deltaTm_Tm4_Tm2=tm4 - tm2,
    )


def determine_gc(pbs_seq, rt_seq) -> PegRNAExtensionData:
    combined = pbs_seq + rt_seq
    return PegRNAExtensionData(
        GC_count_PBS=pbs_seq.count("G") + pbs_seq.count("C"),
        GC_count_RTT=rt_seq.count("G") + rt_seq.count("C"),
        GC_count_RT_PBS=combined.count("G") + combined.count("C"),
        GC_contents_PBS=100 * gc(pbs_seq),
        GC_contents_RTT=100 * gc(rt_seq),
        GC_contents_RT_PBS=100 * gc(combined),
    )


def determine_mfe(pbs_seq, rt_seq, guide_seq) -> MFEData:
    from RNA import fold_compound
    seq3 = (reverse_complement(pbs_seq + rt_seq) + "TTTTTT").replace("T", "U")
    _, mfe3 = fold_compound(seq3).mfe()
    seq4 = guide_seq.replace("T", "U")
    _, mfe4 = fold_compound(seq4).mfe()
    return MFEData(MFE_RT_PBS_polyT=round(mfe3, 1), MFE_Spacer=round(mfe4, 1))


def calculate_rha_len(rt_seq, edit_pos, alt_len, alt_type) -> int:
    if alt_type.lower().startswith("del"):
        return len(rt_seq) - edit_pos + 1
    return len(rt_seq) - edit_pos - alt_len + 1


def calculate_74nt_target(oligo_seq, guide_seq, pbs_rt_seq, pbs_len, rt_len, edit_pos) -> TargetSequenceData:
    match = re.search(f"{guide_seq}[ATCG]GG", oligo_seq, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Guide '{guide_seq}' + NGG not found in OligoSequence_fixed_length")

    nick = match.end() - 6
    wt74 = oligo_seq[nick - UPSTREAM_LENGTH_TO_NICK: nick + DOWNSTREAM_IDX_TO_NICK]
    assert len(wt74) == 74, f"74nt extraction failed: got {len(wt74)} nt"

    seq30 = oligo_seq[nick - UPSTREAM_LENGTH_TO_NICK: nick + 9]
    assert len(seq30) == 30, f"30nt window extraction failed: got {len(seq30)} nt"

    s5 = UPSTREAM_LENGTH_TO_NICK - pbs_len
    s3 = DOWNSTREAM_IDX_TO_NICK - rt_len
    ed74 = "x" * s5 + pbs_rt_seq + "x" * s3
    assert len(ed74) == 74, f"Edited 74nt construction failed: got {len(ed74)} nt"

    return TargetSequenceData(
        wild_type_sequence=wt74.upper(),
        deepspcas9_guide_30=seq30.upper(),
        prime_edited_sequence=ed74.upper(),
        edit_position=edit_pos,
    )


def determine_edit_type(alt_type: str) -> EditTypeClass:
    s = int(alt_type.lower().startswith("sub"))
    i = int(alt_type.lower().startswith("ins"))
    d = int(alt_type.lower().startswith("del"))
    assert s + i + d == 1, f"Invalid edit type: {alt_type}"
    return EditTypeClass(type_sub=s, type_ins=i, type_del=d)


def unpack_dataclass_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        expanded = pd.DataFrame.from_records(df[col].map(asdict))
        expanded.index = df.index
        overlap = [c for c in expanded.columns if c in df.columns]
        df = df.drop(columns=overlap)
        df = pd.concat([df.drop(columns=col), expanded], axis=1)
    return df


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Compute bio-features from raw pegRNA design parameters.

    Args:
        data: DataFrame with columns:
              REF_ID, WideTargetSequence, OligoSequence_fixed_length, Guide,
              PBS, RTT, Edit_type, Edit_len/Edit length, Edit_pos/Edit position,
              PBS_len/PBS_length, RTT_len/RT_length, leading G

    Returns:
        DataFrame with all bio-features in the format expected by PE6DeepPrimeDataset.
    """
    from genet.predict import SpCas9

    df = data.copy()

    # Normalize column names
    col_map = {
        "Edit_len": "Edit length",
        "Edit_pos": "Edit position",
        "PBS_len": "PBS_length",
        "RTT_len": "RT_length",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns and v not in df.columns})

    # Guide position in WideTargetSequence → Nicking site
    guide_pos = df[["WideTargetSequence", "Guide"]].apply(
        lambda x: find_guide_indices(x["WideTargetSequence"], x["Guide"]), axis=1
    )
    df["GuideStart"] = guide_pos.str[0]
    df["GuideEnd"] = guide_pos.str[1]
    df["Nicking"] = df["GuideEnd"] - 3

    # Tm sequences
    df["TmSequences"] = df.apply(
        lambda x: determine_seqs(
            x["Edit_type"], x["Edit length"], x["WideTargetSequence"],
            x["PBS"], x["RTT"], x["Nicking"],
        ), axis=1,
    )

    # Tm values, GC features, MFE
    df["TmData"] = df["TmSequences"].apply(determine_tm)
    df["PegRNAExtensionData"] = df[["PBS", "RTT"]].apply(
        lambda x: determine_gc(x["PBS"], x["RTT"]), axis=1
    )
    df["MFEData"] = df[["PBS", "RTT", "Guide"]].apply(
        lambda x: determine_mfe(x["PBS"], x["RTT"], x["Guide"]), axis=1
    )

    # Structural features
    df["RT_PBS"] = df["RTT"] + df["PBS"]
    df["TS_PBS_RT"] = df["RT_PBS"].map(reverse_complement)
    df["RT_PBS_len"] = df["RT_PBS"].apply(len)
    df["RHA_len"] = df[["RTT", "Edit position", "Edit length", "Edit_type"]].apply(
        lambda x: calculate_rha_len(*x), axis=1
    )

    # 74nt target sequences
    df["TargetSequenceData"] = df[
        ["OligoSequence_fixed_length", "Guide", "TS_PBS_RT", "PBS_length", "RT_length", "Edit position"]
    ].apply(lambda x: calculate_74nt_target(*x), axis=1)

    # Edit type one-hot
    df["EditTypeClass"] = df["Edit_type"].apply(determine_edit_type)

    # Spacer
    df["Spacer"] = df["leading G"] + df["Guide"]

    # Unpack dataclasses
    df = unpack_dataclass_columns(df, [
        "EditTypeClass", "MFEData", "PegRNAExtensionData",
        "TargetSequenceData", "TmData", "TmSequences",
    ])

    # DeepSpCas9 score
    valid_mask = df["deepspcas9_guide_30"].str.fullmatch(r"[ACGTacgt]+")
    if not valid_mask.all():
        print(f"Warning: dropping {(~valid_mask).sum()} rows with invalid SpCas9 window sequences")
        df = df[valid_mask].copy()
    df["DeepSpCas9_score"] = SpCas9().predict(df["deepspcas9_guide_30"])["SpCas9"]

    # Rename to model-expected names
    df = df.rename(columns=RENAME_MAP_FINAL)
    df = df.reset_index(drop=True)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.fillna(0)

    return df
