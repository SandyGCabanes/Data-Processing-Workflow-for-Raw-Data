# -*- coding: utf-8 -*-
import os
import re
import uuid
import unicodedata
import pandas as pd
import numpy as np

# -----------------------------
# COLUMN_MAP: short names (left) → exact raw column names (right)
# -----------------------------
COLUMN_MAP = {
    "resp_id": "resp_id",
    "timestamp": "Timestamp",
    "city": "City of Residence, current",
    "country": "Country of Residence, current",
    "gender": "Gender",
    "age": "Age",
    "educstat": "Latest education status",
    "digitools": "Choose the digital tools you are currently using for learning:",
    "bestproject": "Describe the best project that you did in the last 6 months:",
    "industry": "Industry that you are currently in:",
    "employertype": "Type of employer",
    "careerstg": "Current stage of DATA CAREER",
    "worksame": "Is your work city the same as your city of residence?",
    "workcity": "Work location - specify city",
    "successmethod": "Thinking of your most recent job, which platform or method gave you the most success?",
    "salary": "Monthly Salary Range (Or monthly income from main source)",
    "typework": "Type of work",
    "sitework": "Work set-up",
    "datarole": "What best describes MAJORITY of your day-to-day role?",
    "restofrole": "What other descriptions comprise the REST of your role? (Click all that apply)",
    "sizeteam": "What is the size of your Data Team?",
    "ingestion": "What are the data INGESTION tools you currently use? (Optional)",
    "transform": "What are the data TRANSFORMATION tools you currently use?  (Optional)",
    "warehs": "What are the data WAREHOUSES you currently use?   (Optional)",
    "orchest": "What are the data ORCHESTRATION tools you currently use?   (Optional)",
    "busint": "What are the BUSINESS INTELLIGENCE tools you currently use?  (Optional)",
    "reversetl": "What are the REVERSE ETL tools you currently use?   (Optional)",
    "dataqual": "What are the DATA QUALITY tools you currently use? (Optional)",
    "datacatalog": "What are the DATA CATALOGS you currently use?   (Optional)",
    "cloudplat": "What are the cloud platforms that you currently use?    (Optional)",
    "noncloudplat": "What are the non-cloud platforms that you currently use?  (Optional)",
    "generaltools": "Which of the following general tools do you use? Choose all that apply.",
    "whatused": "Which of the following do you use on a regular basis? Choose all that apply.",
    "useai": "Do you currently use AI in your workflow or study? Choose all that apply.",
    "hostedntbk": "Do you use any of the following hosted notebook products?",
    "hardware": "What hardware do you currently use for data?",
    "depwebsite": "Whether or not aware of the free resources in the DEP website",
    "depwebres": "If aware of the free resources, have you used at least one of the resources in the DEP website?",
    "otherfb": "Thinking of data-related communities, what other Facebook communities do you follow?",
    "spneeds": "Any specific needs you are trying to address by joining DEP Facebook group?",
    "volunteer": "Any specific tasks, skills, knowledge or resources you are willing to contribute to the group?",
    "comms": "Thinking of ways to improve communications in the group, do you have any suggestions?",
    # Derived
    "salary_short": "salary_shortversion",
    "age_group": "agegrp",
    "careerstg_cln": "careerstg_cln",
    "inphils_notinphils": "inphils_notinphils",
}

# -----------------------------
# Global normalization
# -----------------------------
def global_normalize(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None

# -----------------------------
# Mapping loaders
# -----------------------------
MAPPINGS_DIR = "mappings"

def load_exact_map(col):
    path = os.path.join(MAPPINGS_DIR, f"{col}.exact.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df.applymap(global_normalize)
    return {r["source"]: r["target"] for _, r in df.iterrows()}

def load_regex_map(col):
    path = os.path.join(MAPPINGS_DIR, f"{col}.regex.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df.applymap(global_normalize)
    return {r["pattern"]: r["target"] for _, r in df.iterrows()}

def load_set(col, kind="drop"):
    path = os.path.join(MAPPINGS_DIR, f"{col}.{kind}.csv")
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path, dtype=str).fillna("")
    return {global_normalize(v) for v in df["value"].tolist() if v}

# -----------------------------
# Multi-select explode
# -----------------------------
def split_multi(s):
    s = global_normalize(s)
    if s is None:
        return []
    return [global_normalize(p) for p in re.split(r"\s*,\s*", s) if p]

def explode_multiselect(df, col_in, id_col, col_out_name):
    if col_in not in df.columns:
        return pd.DataFrame(columns=[id_col, col_out_name])
    out = (
        df[[id_col, col_in]]
        .assign(_vals=lambda d: d[col_in].apply(split_multi))
        .explode("_vals")
        .dropna(subset=["_vals"])
        .rename(columns={"_vals": col_out_name})
        [[id_col, col_out_name]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return out

# -----------------------------
# Dimloc extraction
# -----------------------------
def extract_dimloc(df, out_csv="dimloc.csv"):
    id_col = COLUMN_MAP["resp_id"]
    cols_present = [COLUMN_MAP[c] for c in ["city", "workcity", "country"] if COLUMN_MAP[c] in df.columns]
    dimloc = df[[id_col] + cols_present].copy()
    for c in cols_present:
        dimloc[c] = dimloc[c].apply(global_normalize)
    dimloc = dimloc.drop_duplicates()
    dimloc.to_csv(out_csv, index=False)
    return dimloc

# -----------------------------
# Main process
# -----------------------------
def process(raw_df):
    df = raw_df.copy()
    df = df.applymap(global_normalize)

    cm = COLUMN_MAP

    # Ensure resp_id exists
    if cm["resp_id"] not in df.columns:
        df[cm["resp_id"]] = [str(uuid.uuid4()) for _ in range(len(df))]

    # Example: apply mappings to single-selects
    for key in ["gender", "educstat", "employertype", "careerstg", "typework", "sitework", "datarole", "sizeteam"]:
        col = cm[key]
        if col in df.columns:
            regex_map = load_regex_map(key)
            exact_map = load_exact_map(key)
            df[col] = df[col].apply(lambda s: global_normalize(s))
            if regex_map:
                df[col] = df[col].apply(lambda s: next((v for pat, v in regex_map.items() if re.match(pat, s or "", re.I)), s))
            if exact_map:
                df[col] = df[col].map(lambda s: exact_map.get(s, s))

    # Extract dimloc
    dimloc_df = extract_dimloc(df)

    # Explode multi-selects
    idc = cm["resp_id"]
    exploded_tables = {}
    for key in ["digitools", "successmethod", "ingestion", "transform", "warehs", "orchest", "busint", "reversetl",
                "dataqual", "datacatalog", "cloudplat", "noncloudplat", "generaltools", "whatused", "useai",
                "hostedntbk", "hardware", "depwebsite", "depwebres", "otherfb", "spneeds", "volunteer", "comms"]:
        col = cm[key]
        if col in df.columns:
            tbl = explode_multiselect(df, col, idc, key)
            # Apply optional drop/keep/exact/regex maps if they exist
            drops = load_set(key, kind="drop")
            if drops:
                tbl[key] = tbl[key].apply(lambda s: None if global_normalize(s) in drops else s)
                tbl = tbl.dropna(subset=[key])
            regex_map = load_regex_map(key)
            if regex_map:
                tbl[key] = tbl[key].apply(lambda s: next((v for pat, v in regex_map.items()
                                                          if re.match(pat, s or "", re.I)), s))
            exact_map = load_exact_map(key)
            if exact_map:
                tbl[key] = tbl[key].map(lambda s: exact_map.get(s, s))
            keeps = load_set(key, kind="keep")
            if keeps:
                tbl = tbl[tbl[key].apply(lambda s: global_normalize(s) in keeps)]
            exploded_tables[key] = tbl.reset_index(drop=True)

    # Return all outputs in one dictionary
    outputs = {"main": df.reset_index(drop=True), "dimloc": dimloc_df.reset_index(drop=True)}
    outputs.update(exploded_tables)
    return outputs
