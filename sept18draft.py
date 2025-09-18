# %%
# 0. Imports & Config
# ~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import unicodedata
from pathlib import Path
import numpy as np


# %%
lookup_dir = Path("lookup_dir")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
lookup_dir.mkdir(exist_ok=True)


# %%
# 1. Global Cleaning Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~
def normalize_text(val):
    if pd.isna(val):
        return ""
    val = unicodedata.normalize("NFKC", str(val))
    val = val.strip()
    val = " ".join(val.split())
    return val

def clean_dataframe(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(normalize_text)
        if col in ["city", "country"]:
            df[col] = df[col].str.title()
    return df



# %%
# 2. Load Raw Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~
df_raw = pd.read_csv("csv_raw.csv")




# %%
# 2a. Rename Columns (Raw → Transformed)
# ~~~~~~~~~~~~~~~~~~~~~~~~~
rename_map = {
    "Timestamp": "timestamp",
    "City of Residence, current": "city",
    "Country of Residence, current": "country",
    "Gender": "gender",
    "Age": "age",
    "Latest education status": "educstat",
    "Choose the digital tools you are currently using for learning:": "digitools",
    "Describe the best project that you did in the last 6 months:": "bestproject",
    "Industry that you are currently in:": "industry",
    "Current stage of DATA CAREER": "careerstg",
    "Is your work city the same as your city of residence?": "worksame",
    "Work location - specify city": "workcity",
    "Monthly Salary Range (Or monthly income from main source)": "salary",
    "Type of work": "typework",
    "Work set-up": "sitework",
    "What best describes MAJORITY of your day-to-day role?": "datarole",
    "What other descriptions comprise the REST of your role? (Click all that apply)": "restofrole",
    "What is the size of your Data Team?": "sizeteam",
    "What are the data INGESTION tools you currently use? (Optional)": "ingestion",
    "What are the data TRANSFORMATION tools you currently use?  (Optional)": "transform",
    "What are the data WAREHOUSES you currently use?   (Optional)": "warehs",
    "What are the data ORCHESTRATION tools you currently use?   (Optional)": "orchest",
    "What are the BUSINESS INTELLIGENCE tools you currently use?  (Optional)": "busint",
    "What are the REVERSE ETL tools you currently use?   (Optional)": "reversetl",
    "What are the DATA QUALITY tools you currently use? (Optional)": "dataqual",
    "What are the DATA CATALOGS you currently use?   (Optional)": "datacatalog",
    "What are the cloud platforms that you currently use?    (Optional)": "cloudplat",
    "What are the non-cloud platforms that you currently use?  (Optional)": "noncloudplat",
    "Which of the following general tools do you use? Choose all that apply.": "generaltools",
    "Which of the following do you use on a regular basis? Choose all that apply.": "whatused",
    "Do you currently use AI in your workflow or study? Choose all that apply.": "useai",
    "Do you use any of the following hosted notebook products?": "hostedntbk",
    "What hardware do you currently use for data?": "hardware",
    "Whether or not aware of the free resources in the DEP website": "depwebsite",
    "If aware of the free resources, have you used at least one of the resources in the DEP website?": "depwebres",
    "Thinking of data-related communities, what other Facebook communities do you follow?": "otherfb",
    "Any specific needs you are trying to address by joining DEP Facebook group?": "spneeds",
    "Any specific tasks, skills, knowledge or resources you are willing to contribute to the group?": "volunteer",
    "Thinking of ways to improve communications in the group, do you have any suggestions?": "comms",
    "Thinking of your most recent job, which platform or method gave you the most success?": "successmethod",
    "Type of employer": "employertype"
}




# %%
df_raw.rename(columns=rename_map, inplace=True)
df_raw.to_pickle(output_dir / "df_raw.pkl")



# %%
# 3. Global Cleaning (Inline)
# ~~~~~~~~~~~~~~~~~~~~~~~~~
for col in df_raw.select_dtypes(include="object").columns:
    # Normalize, strip, collapse spaces
    df_raw[col] = (
        df_raw[col]
        .fillna("")
        .astype(str)
        .apply(lambda v: " ".join(
            unicodedata.normalize("NFKC", v).strip().split()
            ))
    )
    # Title-case specific columns
    if col in ["city", "country", "workcity"]:
        df_raw[col] = df_raw[col].str.title()

# Replace strings that start with 'Other' with blanks
for col in df_raw.select_dtypes(include="object").columns:
    df_raw[col] = df_raw[col].apply(lambda v: "" if v.startswith("Other") else v)




# %%
# Add resp_id after cleaning
df_raw.insert(0, "resp_id", range(1, len(df_raw) + 1))
df_raw.to_csv("df_raw.csv", index = False)


# %%
# 4. Apply Lookups (Single-Response Columns)
# ~~~~~~~~~~~~~~~~~~~~~~~~~
df_raw = pd.read_csv("df_raw.csv", encoding = 'utf-8')
single_response_cols = [
    "gender", "age", "age_grp", "educstat", "industry", "careerstg", "worksame", "workcity", "salary", "typework", "sitework", "datarole", "sizeteam", "employertype"
]


def apply_lookup(df, column):
    lookup_file = lookup_dir / f"{column}_lookup.csv"
    if lookup_file.exists():
        map_df = pd.read_csv(lookup_file)
        mapping = dict(zip(map_df["raw"], map_df["clean"]))
        df[column] = df[column].map(mapping).fillna(df[column])
    return df



for col in single_response_cols:
    df_raw= apply_lookup(df_raw, col)



# %%
# 4. Create age bins after age column lookup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Discretizing the 'age' variable into 'age_grp'
bins = [-np.inf, 19, 25, 30, 35, 40, 45, 50, 55, np.inf]
labels = ["<19", "20 to 24", "25 to 29", "30 to 34", "35 to 39", "40 to 44", "45 to 49", "50 to 54", "55+"]
df_raw['age_grp'] = pd.cut(df_raw['age'], bins=bins, labels=labels, right=False, include_lowest=True)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Start using df instead of df_raw
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
# Make a copy of df_raw as df
df = df_raw.copy()



# %%
# 4. Create df_single response
df_single = df_raw[single_response_cols + ["resp_id", "timestamp", "city", "country"]]
df_single.to_pickle(output_dir / "df_single.pkl")




# %%
# 4. Explore unique values in df_single columns
# Unique values per column
unique_dir = Path("unique_dir")
unique_dir.mkdir(exist_ok=True)
for col in df_single.columns:
    print(f"{col}\n", df_single[col].unique())
    
# Write all unique values to a single file except resp-id and timestamp
single_columns = [col for col in df_single.columns if col not in ['resp_id', 'timestamp']]
with open(unique_dir / "unique_single.txt", "w", encoding="utf-8") as f:
    for col in single_columns:
        f.write(f"{col}\n")
        for val in df_single[col].dropna().unique():
            f.write(f"  {val}\n")
        f.write("\n")





# %%
# 5. Location Section
# ~~~~~~~~~~~~~~~~~~~~~~~~~
df_loc = df[["resp_id", "city", "country"]].drop_duplicates()
df_loc = apply_lookup(df_loc, "city")
df_loc.to_pickle(output_dir / "df_loc.pkl")





# %%
# 6. Multi-Response Section
# ~~~~~~~~~~~~~~~~~~~~~~~~~
multi_response_cols = [
    "digitools", "successmethod", "restofrole", "ingestion", "transform",
    "warehs", "orchest", "busint", "reversetl", "dataqual", "datacatalog",
    "cloudplat", "noncloudplat", "generaltools", "whatused", "useai",
    "hostedntbk", "hardware", "otherfb"
]

def explode_and_lookup(df, column):
    # Include resp_id and multi-response column
    exploded = (
        df[["resp_id", column]]
        .assign(**{column: df[column].str.split(",")})
        .explode(column)
    )
    exploded[column] = exploded[column].apply(normalize_text)
    exploded = apply_lookup(exploded, column)
    exploded.to_pickle(output_dir / f"{column}.pkl")
    return exploded

# Create a pickle file for each column with 
for col in multi_response_cols:
    explode_and_lookup(df, col)



# %%
# Store the multi-response dfs in a dictionary
exploded_dfs = {}
for col in multi_response_cols:
    exploded_dfs[col] = explode_and_lookup(df, col)






# %%
# 7. Free-Text Section
# ~~~~~~~~~~~~~~~~~~~~~~~~~
free_text_cols = ["spneeds", "volunteer", "comms", "bestproject"]
df_freetext = df[["resp_id"] + free_text_cols]
df_freetext.to_pickle(output_dir / "df_freetext.pkl" )



# ~~~~~~~~~~~~~~~~~~~~~~~
# Load into duckdb
# ~~~~~~~~~~~~~~~~~~~~~~~




# %%
# 11. Initialize duckdb
import duckdb


# Create persistent DuckDB file
con = duckdb.connect("data_model.duckdb")






# %%
# Register df_single, df_loc, df_freetext
con.register("df_single", df_single)
con.register("df_loc", df_loc)
con.register("df_freetext", df_freetext)


# Register exploded multi-response tables
for name, df_exp in exploded_dfs.items():
    con.register(f"df_{name}", df_exp)





# %%
# Make tables persist: single, location, freetext
con.execute("CREATE TABLE single AS SELECT * FROM df_single")
con.execute("CREATE TABLE location AS SELECT * FROM df_loc")
con.execute("CREATE TABLE freetext AS SELECT * FROM df_freetext")

# Persist exploded tables
for name in exploded_dfs:
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM df_{name}")




"""
single_response_cols = [
    "gender", "age", "educstat", "industry", "careerstg",
    "worksame", "workcity", "salary", "typework", "sitework",
    "datarole", "sizeteam", "employertype"
]
multi_response_cols = [
    "digitools", "successmethod", "restofrole", "ingestion", "transform",
    "warehs", "orchest", "busint", "reversetl", "dataqual", "datacatalog",
    "cloudplat", "noncloudplat", "generaltools", "whatused", "useai",
    "hostedntbk", "hardware", "otherfb"
]
"""



# %%
# 
query = """
    SELECT s.*, d.digitools_clean
    FROM single s
    LEFT JOIN digitools d ON s.resp_id = d.resp_id
"""
df_model = con.execute(query).fetchdf()





# %%
# Close duckcb
con.close()
