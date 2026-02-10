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
# 1. Load Raw Data and Rename Columns (Raw -> Transformed)
# ~~~~~~~~~~~~~~~~~~~~~~~~~
df_raw = pd.read_csv("csv_raw.csv")

# Rename Columns (Raw → Transformed)
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
# 2. First Global Cleaning (Inline)

# Apply global cleaning
for col in df_raw.select_dtypes(include="object").columns:
    # Normalize, strip, collapse spaces
    df_raw[col] = (
        df_raw[col]
        .fillna("")
        .astype(str)
        .apply(lambda v: " ".join(
            unicodedata.normalize("NFKC", v.replace("ñ", "n")).strip().split()
        ))
    )
    # Title-case specific columns
    if col in ["city", "country", "workcity", "datarole", "industry"]:
        df_raw[col] = df_raw[col].str.title()

# Replace strings that start with 'Other' with blanks
for col in df_raw.select_dtypes(include="object").columns:
    df_raw[col] = df_raw[col].apply(lambda v: "" if v.startswith("Other") else v)

# Replace age of 0 with min age, and cap at 95
valid_min_age = df_raw.loc[df_raw["age"] > 0, "age"].min()
df_raw["age"] = df_raw["age"].replace(0, valid_min_age).clip(upper=95)
max_age = df_raw["age"].max()
df_raw["age"] = df_raw["age"].apply(lambda x: max_age if x > 95 else x)

# Standardize delimiter in workcity column
df_raw["workcity"] = df_raw["workcity"].str.replace(",", "-", regex=False)






# %%
# 3. Add resp_id to df_raw after first global cleaning
if "resp_id" not in df_raw.columns:
    df_raw.insert(0, "resp_id", range(1, len(df_raw) + 1))

df_raw.to_csv(csv_outputs_dir / "df_raw.csv", index=False)
df_raw.to_pickle(outputs_dir / "df_raw.pkl")

print("df_raw.csv created. Global cleaning done.")



# %%
# 4a. Define single columns for lookups
# This section uses df_single_no_grps
single_cols_no_grps = [
    "gender",  "educstat", "industry", "careerstg", "worksame", "workcity", "salary", "typework", "sitework", "datarole", "sizeteam", "employertype",  "depwebsite", "depwebres", "resp_id", "age",
]

df_single_no_grps = df_raw[single_cols_no_grps].copy()

# %%
# 4a1. Replace "ñ" with "n"
# df_single_no_grps['workcity'].str.replace("ñ", "n", regex = False)
# df_workcity = df_single_no_grps.groupby('workcity')['resp_id'].count().reset_index()
# df_workcity.to_csv("df_workcity.csv", index = False, encoding = "utf-8")

# %%
# 4a2. Print out to terminal all unique values
# Check unique values first except resp_id  
# Write all unique values to a single file except resp_id, age

single_columns_non_numeric = [col for col in df_single_no_grps.columns if col not in ['resp_id', 'age']]

with open(unique_dir / "unique_single.txt", "w", encoding="utf-8") as f:
    for col in single_columns_non_numeric:
        f.write(f"{col}\n\n")
        for val in df_single_no_grps[col].dropna().unique():
            f.write(f"{val}\n")
        f.write("\n")
        
print("unique_single.txt created.  Inspect unique values.")
with open(unique_dir / "unique_single.txt", "r", encoding="utf-8") as f:
    print(f.read())


# %%
# 5. Lookups (Single-Response Columns)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We apply lookups  
def apply_lookup(df, column):
    lookup_file = lookup_dir / f"{column}_lookup.csv"
    if not lookup_file.exists():
        print(f"{column} lookup file not found.")
    else:
        map_df = pd.read_csv(lookup_file, encoding="windows-1252")
        # Replace NaN with empty string
        map_df["raw"] = map_df["raw"].fillna("")
        map_df["clean"] = map_df["clean"].fillna("")

        # Check for unmatched items
        list_raw = list(df[col].unique())
        list_lookup = list(map_df["raw"].unique())
        unmatched_lookups = [x for x in list_raw if x not in list_lookup]
        print(f"Unmatched items in {col}:\n")
        for item in unmatched_lookups:
            print(f"{repr(item)}")
        #print(f"\n\nUnique items in {col}_lookup 'raw':\n")
        #for item in list_lookup:
        #    print(f"- {repr(item)}")


        # Create mapping dictionary
        mapping = dict(zip(map_df["raw"], map_df["clean"]))

        # Apply mapping
        original = df[column]
        mapped = original.map(mapping)

        # Apply mapping with fallback to original
        df[column] = mapped.fillna(df[column])
    return df



for col in single_columns_non_numeric:
    df_single_no_grps= apply_lookup(df_single_no_grps, col)
    print(f"\n{col} lookup done.\n")



# %%
# 5a. Output converted to csv and pickle 
df_single_no_grps.to_csv(csv_outputs_dir /"df_single_no_grps.csv", index = False)
print("Lookups done. df_single_no_grps.csv created.")

df_single_no_grps.to_pickle(outputs_dir/ "df_single_no_grps.pkl")
print("Lookups done. df_single_no_grps.pkl created.  ")


# %%
# 5b. Initialize the df_single dataframe for adding groups later
df_single_with_grps = df_single_no_grps.copy()


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
dgrp_mapping_df = pd.read_csv("lookup_others/datarolegrp_lookup.csv").copy()  # not just view but persistent

dgrp_mapping = dict(zip(dgrp_mapping_df["datarole"], dgrp_mapping_df["datarole_group"]))

# Create datarole_grp using map(dgrp_mapping)
df_single_with_grps["datarole_grpd"] = df_single_with_grps["datarole"].map(dgrp_mapping).fillna(df_single_with_grps["datarole"])

print("datarole_grpd column created. Inspect unique values.")



# %%
# 7b. Compare the two dataroles - from the column and the lookup map
# ~~~~~~~~~~~~~~~~~~~~~~~~
list_dataroles1 = list(df_single_with_grps['datarole'].unique())
list_dataroles2 = list(dgrp_mapping_df['datarole'].unique())
datarole_not_similar = [x for x in list_dataroles1 if x not in list_dataroles2]


print("dataroles in column but not in lookup\n")
print(datarole_not_similar)  #We loop back if datarole_not_similar is not blank.

# %%
# 8. Check unique values again
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Write all unique values to a single file except resp_id and timestamp, this time age_grp and datarole_grpd included

single_columns_grpd_except_resp_id = [col for col in df_single_with_grps.columns if col not in['age', 'resp_id']]
with open(unique_dir / "unique_single_grpd.txt", "w", encoding="utf-8") as f:
    for col in single_columns_grpd_except_resp_id:
        f.write(f"{col}\n\n")
        for val in df_single_with_grps.columns[col].dropna().unique():
            f.write(f"{val}\n")
        f.write("\n")




# %%
# 9. Export df_single to csv and pickle
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


df_single = df_single_with_grps.copy()
df_single.to_csv(csv_outputs_dir/"df_single.csv", index = False)
df_single.to_pickle(outputs_dir / "df_single.pkl")







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

# Transform string values
def normalize_text(val):
    if pd.isna(val):
        return ""
    val = unicodedata.normalize("NFKC", str(val))
    val = val.strip()
    val = " ".join(val.split())
    return val

def explode_and_lookup(df, column):
    # Include resp_id and multi-response column
    exploded = (
        df[["resp_id", column]]
        .assign(**{column: df[column].str.split(",")})
        .explode(column)
    )
    exploded[column] = exploded[column].apply(normalize_text)
    # For future use: apply lookup
    # exploded = apply_lookup(exploded, column)

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
# ~~~~~~~~~~~~~~~~~~~~~~~
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

