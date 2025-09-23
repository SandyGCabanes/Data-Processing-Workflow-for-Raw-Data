# %%
# 0. Imports & Config
# ~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import unicodedata
from pathlib import Path
import numpy as np


# %%
# 0. Create Directories
# ~~~~~~~~~~~~~~~~~~~~~~~~~
lookup_dir = Path("lookup_dir")
lookup_dir.mkdir(exist_ok=True)

outputs_dir = Path("outputs_dir")
outputs_dir.mkdir(exist_ok=True)

csv_outputs_dir = Path("csv_outputs_dir")
csv_outputs_dir.mkdir(exist_ok=True)

unique_dir = Path("unique_dir")
unique_dir.mkdir(exist_ok=True)





# %%
# 1. Load Raw Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~
df_raw = pd.read_csv("csv_raw.csv")




# %%
# 2. Rename Columns (Raw → Transformed)
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


df_raw.rename(columns=rename_map, inplace=True)




# %%
# 3a. First Global Cleaning (Inline)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Global Cleaning Functions
# Transform string values
def normalize_text(val):
    if pd.isna(val):
        return ""
    val = unicodedata.normalize("NFKC", str(val))
    val = val.strip()
    val = " ".join(val.split())
    return val

# Use normalize_text function above and title format for "city", "country", "workcity", "datarole", "industry"
def clean_dataframe(df):
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(normalize_text)
        if col in ["city", "country", "workcity", "datarole", "industry"]:
            df[col] = df[col].str.title()
    return df


# Apply Global Cleaning functions
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
    if col in ["city", "country", "workcity", "datarole", "industry"]:
        df_raw[col] = df_raw[col].str.title()

# Replace strings that start with 'Other' with blanks
for col in df_raw.select_dtypes(include="object").columns:
    df_raw[col] = df_raw[col].apply(lambda v: "" if v.startswith("Other") else v)




# %%
# 3b. Add resp_id to df_raw after first global cleaning
if "resp_id" not in df_raw.columns:
    df_raw.insert(0, "resp_id", range(1, len(df_raw) + 1))

df_raw.to_csv(csv_outputs_dir / "df_raw.csv", index=False)
df_raw.to_pickle(outputs_dir / "df_raw.pkl")

print("df_raw.csv created. Global cleaning done.")


# %%
# 4. Explore unique values before using lookup files
# No need to run after lookups are created.
# This section uses df_single_no_grps
single_cols_no_grps = [
    "gender", "age", "educstat", "industry", "careerstg", "worksame", "workcity", "salary", "typework", "sitework", "datarole", "sizeteam", "employertype",  "depwebsite", "depwebres", "resp_id"
]

df_single_no_grps = df_raw[single_cols_no_grps].copy()

# Check unique values first except resp_id  
# Write all unique values to a single file except resp_id 
single_columns_except_respid = [col for col in df_single_no_grps.columns if col !='resp_id']
with open(unique_dir / "unique_single.txt", "w", encoding="utf-8") as f:
    for col in single_columns_except_respid:
        f.write(f"{col}\n\n")
        for val in df_single_no_grps[col].dropna().unique():
            f.write(f"{val}\n")
        f.write("\n")
print("unique_single.txt created.  Inspect unique values.")



# %%
# 5. Initial Lookups (Single-Response Columns)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We apply lookups to single_columns_except_respid
 
def apply_lookup(df, column):
    lookup_file = lookup_dir / f"{column}_lookup.csv"
    if lookup_file.exists():
        map_df = pd.read_csv(lookup_file)
        mapping = dict(zip(map_df["raw"], map_df["clean"]))
        df[column] = df[column].map(mapping).fillna(df[column])
    return df

for col in single_columns_except_respid:
    df_single_no_grps= apply_lookup(df_single_no_grps, col)

df_single_no_grps.to_csv(csv_outputs_dir /"df_single_no_grps.csv", index = False)
print("Lookups done. df_single_no_grps.csv created.")

df_single_no_grps.to_pickle(outputs_dir/ "df_single_no_grps.pkl")
print("Lookups done. df_single_no_grps.pkl created.  ")


# Initialize the df_single dataframe for adding groups later
df_single_with_grps = df_single_no_grps.copy()
print("df_single_with_grps initialized.")
df_single_with_grps['datarole'].unique()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# At this point, df_single_no_grps single-response cols should have been cleaned. And df_single_with_grps is initialized.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# %% 
# 6. Create age_grp in df_single
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Binning the 'age' variable into 'age_grp'
bins = [-np.inf, 19, 25, 30, 35, 40, 45, 50, 55, np.inf]
labels = ["<19", "20 to 24", "25 to 29", "30 to 34", "35 to 39", "40 to 44", "45 to 49", "50 to 54", "55+"]
df_single_with_grps['age_grp'] = pd.cut(df_single_with_grps['age'], bins=bins, labels=labels, right=False, include_lowest=True)
print("age_grp column added to df_single_with_grps.")



# %% 
# 7. Create datarole_grp in df_single
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Remember that the lookup is in lookup_others, not the lookup_dir used for the initial single-response columns.
dgrp_mapping_df = pd.read_csv("lookup_others/datarolegrp_lookup.csv")

print(dgrp_mapping_df['datarole'].unique())


# %%
# 7a. Compare datarole in df with datarole in lookup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
list1 = list(df_single_with_grps['datarole'].unique())
list2 = list(dgrp_mapping_df['datarole'].unique())
unmatched = [x for x in list1 if x not in list2]
print(unmatched)




# %%
# 7b.Normalize values before applying dictionary mapping
dgrp_mapping_df["datarole"] = (
    dgrp_mapping_df["datarole"]
    .fillna("")
    .astype(str)
    .apply(normalize_text)
    .str.title()
)
# Create dictionary for mapping
dgrp_mapping = dict(zip(dgrp_mapping_df["datarole"], dgrp_mapping_df["datarole_group"]))


# %%
# 7c. Compare datarole in df with NORMALIZED datarole in lookup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
list1 = list(df_single_with_grps['datarole'].unique())
list3 = list(dgrp_mapping_df['datarole'].unique())
unmatched = [x for x in list1 if x not in list2]
print(unmatched)



# %%
# 7d. Create datarole_grp using map(dgrp_mapping)
df_single_with_grps["datarole_grpd"] = df_single_with_grps["datarole"].map(dgrp_mapping).fillna(df_single_with_grps["datarole"])

print("datarole_grpd column created. ")
print(df_single_with_grps.columns)

# %%
# 8. Check unique values again
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Write all unique values to a single file, this time age_grp and datarole_grpd included

single_columns_grpd_except_resp_id = [col for col in df_single_with_grps.columns if col != 'resp_id']
print(single_columns_grpd_except_resp_id)

with open(unique_dir / "unique_single_grpd.txt", "w", encoding="utf-8") as f:
    for col in single_columns_grpd_except_resp_id:
        f.write(f"{col}\n\n")
        for val in df_single_with_grps[col].dropna().unique():
            f.write(f"{val}\n")
        f.write("\n")



# %%
# 9. Export df_single to csv and pickle
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add back resp_id, age_grp and datarole_grpd columns
df_single = df_single_with_grps.copy()
df_single.to_csv(csv_outputs_dir/"df_single.csv", index = False)
df_single.to_pickle(outputs_dir / "df_single.pkl")

print("df_single.csv and df_single.pkl created.")







# %%
# 10. Location Section
# ~~~~~~~~~~~~~~~~~~~~~~~~~
df_loc = df_raw[["resp_id", "city", "country"]].drop_duplicates()


# Standardize encoding
df_loc["city"] = df_loc["city"].str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")

# Add the coordinates to df_loc
df_coordinates = pd.read_csv("lookup_others/locations_with_coordinates.csv")
df_loc = df_loc.merge(df_coordinates, left_on="city", right_on = "Cities_grouped", how="left")       
df_loc.to_csv(csv_outputs_dir/"df_loc.csv", index = False)
df_loc.to_pickle(outputs_dir / "df_loc.pkl")

# df_loc will be processed separately in a different python file.
# Html map will be created.







# %%
# 11. Multi-Response Section
# ~~~~~~~~~~~~~~~~~~~~~~~~~
multi_response_cols = ["digitools", "successmethod", "restofrole", "ingestion", "transform","warehs", "orchest", "busint", "reversetl", "dataqual", "datacatalog","cloudplat", "noncloudplat", "generaltools", "whatused", "useai","hostedntbk", "hardware", "otherfb"]

def explode_and_lookup(df, column):
    # Include resp_id and multi-response column
    exploded = (
        df[["resp_id", column]]
        .assign(**{column: df[column].str.split(",")})
        .explode(column)
    )
    exploded[column] = exploded[column].apply(normalize_text)
    exploded = apply_lookup(exploded, column)

    # Remove duplicates (resp_id, value) pairs
    exploded = exploded.drop_duplicates(subset=["resp_id", column], keep="first")
    exploded.to_pickle(outputs_dir / f"{column}.pkl")
    exploded.to_csv(csv_outputs_dir/f"{column}.csv", index = False)
    return exploded


df_multi = df_raw[multi_response_cols].copy()


# Create a pickle file for each column 
for col in multi_response_cols:
    explode_and_lookup(df_raw, col)


# Store the multi-response dfs in a dictionary exploded_dfs
exploded_dfs = {}
for col in multi_response_cols:
    exploded_dfs[col] = explode_and_lookup(df_raw, col)






# %%
# 12. Free-Text Section
# ~~~~~~~~~~~~~~~~~~~~~~~~~
free_text_cols = ["spneeds", "volunteer", "comms", "bestproject"]
df_freetext = df_raw[["resp_id"] + free_text_cols]
df_freetext.to_csv(csv_outputs_dir/"df_freetext.csv", index = False)
df_freetext.to_pickle(outputs_dir / "df_freetext.pkl" )

# df_freetext.csv will be processed in a separate python file



# %%
# 13. Load into duckdb
# ~~~~~~~~~~~~~~~~~~~~~~~
# At this point, there are four sub-dfs.  
# df_single, df_loc, df_freetext and exploded_dfs saved in the dictionary

# Import packages
import duckdb
import pandas as pd


# Initializing the connection to a persistent DuckDB file
con = duckdb.connect("data_tables.duckdb")

# Clearing all remnants from previous attempts
print("Clearing all previous tables and views...")

try:
    # Use SHOW ALL TABLES to get all objects (tables and views)
    all_objects = con.execute("SHOW ALL TABLES").fetchall()
    
    # Iterate and drop each object
    for obj in all_objects:
        name = obj[0]
        type_ = obj[2]
        
        if type_ == 'VIEW':
            con.execute(f"DROP VIEW IF EXISTS {name}")
            print(f"Dropped view: {name}")
        else:
            con.execute(f"DROP TABLE IF EXISTS {name}")
            print(f"Dropped table: {name}")

except Exception as e:
    print(f"Error during cleanup: {e}")

print("Cleanup complete.")

print("\nStarting the data loading process...")

# Directly write single response tables to duckdb
print("Writing single-response tables directly to DuckDB...")

# Drop and recreate single-response tables
con.execute("DROP TABLE IF EXISTS single")
con.execute("CREATE TABLE single AS SELECT * FROM df_single")
print("Successfully created and populated 'single' table.")

con.execute("DROP TABLE IF EXISTS location")
con.execute("CREATE TABLE location AS SELECT * FROM df_loc")
print("Successfully created and populated 'location' table.")

con.execute("DROP TABLE IF EXISTS freetext")
con.execute("CREATE TABLE freetext AS SELECT * FROM df_freetext")
print("Successfully created and populated 'freetext' table.")



# Directly write multi-response tables to duckdb
print("\nWriting multi-response tables directly to DuckDB...")

# Create and insert each multi-response table from dictionary
for name, df_multi in exploded_dfs.items():
    # Drop the table if it already exists
    con.execute(f"DROP TABLE IF EXISTS {name}")
    # Create the table from the current DataFrame
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM df_multi")
    print(f"Successfully created and populated table: {name}")



print("\nVerifying that all tables exist...")
# Verifying that all tables were created
print(con.execute("SHOW TABLES").fetchall())
print("Verification complete.")


# Closing the connection to save all changes

con.close()

print("\nProcess completed. Connection closed and data saved to 'data_model.duckdb'.")
