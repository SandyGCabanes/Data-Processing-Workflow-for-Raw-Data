# SOTC survey cleaning – Python
import re
import uuid
import numpy as np
import pandas as pd

# -----------------------------
# Config: column names and constants
# -----------------------------
COLUMN_MAP = {
    # Core identifiers
    "resp_id": "resp_id",
    "timestamp": "timestamp",

    # Demographics
    "email": "email",
    "age": "age",
    "gender": "gender",
    "education": "latest_education",
    "career_stage": "career_stage",
    "country": "country",
    "province": "province",
    "city": "city",

    # Salary
    "salary": "salary",
    "salary_short": "salary_shortversion",

    # Location coordinates
    "lat": "latitude",
    "lon": "longitude",

    # Working section
    "employer_type": "employer_type",
    "type_work": "typework",
    "site_work": "sitework",
    "data_role_main": "data_role",
    "data_role_other": "restofrole",
    "team_size": "team_size",
    "industry": "industry",

    # Multi-selects
    "success_method": "successmethod",
    "cloud_platforms": "cloudplat",
    "noncloud_platforms": "noncloudplat",
    "general_tools": "generaltools",
    "regular_tools": "tools_regular",      # “What used on a regular basis”
    "ai_tools": "use_ai",
    "hosted_notebooks": "hosted_notebooks",
    "hardware": "hardware",
    "digital_learning": "digitools",
    "sp_needs": "spneeds",
    "volunteer": "volunteer",
    "comms": "comms",
    "fb_groups": "fb_groups",              # other facebook communities followed

    # Derived
    "age_group": "agegrp",
    "career_stage_clean": "careerstg_cln",
    "in_phils_flag": "inphils_notinphils"
}

AGE_MIN, AGE_MAX = 15, 100

# -----------------------------
# Mapping dictionaries and rules
# -----------------------------

# Gender
GENDER_BLANK_TO = "Prefer not to say"

# City text normalization
CITY_REPLACEMENTS = {
    "Ã±": "n",
    "CATBALOGAN": "Catbalogan",
}
CITY_NONE_TO = "Perth Australia"  # “None” -> Perth Australia (work location)
CITY_FILL_FROM = ["province", "country"]  # fill city from these if city missing

# Specific row deletions by city exact value (documented)
DELETE_CITY_EXACT = {
    "Borongan City, Eastern Samar",
    "Aleosan, North Cotabato",
}
# We keep “Borongan City” and “Aleosan, Philippines” as is.

# Salary short bins (verbatim)
SALARY_BINS_MAP = {
    "15,000 and below": "15K or less",
    "15,001 to 25,000": "15K+ to 25K",
    "25,001 to 35,000": "25K+ to 35K",
    "35,001 to 45,000": "35K+ to 45K",
    "45,001 to 55, 000": "45K+ to 55K",
    "55,001 to 65,000": "55K+ to 65K",
    "65,001 to 75,000": "65K+ to 75K",
    "75,001 to 85,000": "75K+ to 85K",
    "85,001 to 95,000": "85K+ to 95K",
    "95,001 to 100,000": "95K+ to 100K",
    "100,001 to 125,000": "100K+ to 125K",
    "125,001 to 250, 000": "125K+ to 250K",
    "250,001 and above": "250K+",
}

# Age groups (PIDS AAP + <19)
AGE_GROUPS = [
    ("<19", lambda a: a < 19),
    ("19-24", lambda a: 19 <= a <= 24),
    ("25-29", lambda a: 25 <= a <= 29),
    ("30-34", lambda a: 30 <= a <= 34),
    ("35-39", lambda a: 35 <= a <= 39),
    ("40-44", lambda a: 40 <= a <= 44),
    ("45-49", lambda a: 45 <= a <= 49),
    ("50-54", lambda a: 50 <= a <= 54),
    ("55-59", lambda a: 55 <= a <= 59),
    ("60+",  lambda a: a >= 60),
]

# Career stage cleaning
CAREER_STAGE_NORMALIZE = {
    # Shorten
    r"^\s*Student\s*/\s*New grad\s*/\s*Career break.*$": "Student/ New grad/ Career Break",
    # Opened new code Freelance (if appears as such keep Freelance)
}
# If not in {Student/ New grad/ Career Break, Career shifter, Professional, Freelance} -> Other

CAREER_STAGE_ALLOWED = {
    "Student/ New grad/ Career Break",
    "Career shifter",
    "Professional",
    "Freelance",
}
CAREER_STAGE_DEFAULT_OTHER = "Other"

# In-Philippines flag
IN_PHILS_TRUE = {"Philippines", "PH", "Pilipinas"}  # basic check on country

# Location lat/lon manual imputations
IMPUTE_COORDS_BY_PLACE = {
    # keys can be city, province, or country matches (lowercased)
    "calabarzon": {"city": "Cavite", "lat": 14.4791, "lon": 120.8969},
    "zambales": {"city": "Olongapo City", "lat": 14.8386, "lon": 120.2842},
    "united kingdom": {"city": "London", "lat": 51.5074, "lon": -0.1278},
}

# Employer type recode
EMPLOYER_RECODE = {
    r"^\s*Local\s*$": "Local",
    r"^\s*Multinational\s*$": "Multinational",
    r".*\b(International|US[- ]?based|Foreign)\b.*": "Foreign-Other",
    r".*\b(Self[- ]?employed|Freelance)\b.*": "Self-employed",
    r"^\s*Private\s*$": "Unspecified",
}

# Success method (multi-select) recode
SUCCESS_METHOD_TO_KEEP = {
    "LinkedIn", "Indeed", "Glassdoor", "Upwork", "Onlinejobs.ph", "networking",
    "Jobstreet", "Facebook", "Discord, Slack, Other",
    "Social Media - Unspecified", "My network – people I know",
    "Headhunter", "Online – Other",
}
SUCCESS_METHOD_MAP = {
    # Headhunter
    r"^Headhunt(er)?$": "Headhunter",

    # Online – Other
    r"^(Seek|Monster(\.com)?|Kalibrr|Local Posting|company website|Toptal|Udemy|Tableau Public|online job search abroad)$": "Online – Other",

    # Jobstreet typos
    r"^Jobstree(?:t|y|r)?$": "Jobstreet",

    # Facebook variants
    r"^(Facebook|Facebook groups|FB|Facebook freelancing groups|Postings in groups|Social Media \(FB ad\))$": "Facebook",

    # Discord/Slack
    r"^(Discord|Slack communities)$": "Discord, Slack, Other",

    # Social media unspecified
    r".*Social Media.*": "Social Media - Unspecified",

    # My network
    r"^(Colleague referral on my skillsets|Gradschool career center|Municipal LGU|Office other department)$": "My network – people I know",
}
SUCCESS_METHOD_DROP_TO_BLANK = {
    "N/A", "NA", "Na", "None", "Not working", "none as of the moment",
    "still looking for remote work", "Di pa hired as freelance", "None of the above",
    "I am still yet to try these job networking sites", "Haven’t tried", "No success yet", "Unemployed"
}

# Sitework recode
SITEWORK_RECODE = {
    r"^\s*(Not working)\s*$": None,
    r".*\bHybrid\b.*": "Hybrid",
    r"^\s*Field\s*$": "100% onsite",
    r".*\bonline\b.*": "Mostly work from home/ fully remote",
}

# Data role specific transforms (main)
DATA_ROLE_STANDARDIZE = {
    r"^admin(\b| .*)$": "Admin",
    r"^(Na|NA|N\.A|N/A|student|Studying|Unemployed|Housekeeping|Aspiring DA|career shifter|Not working)$": None,
    r"^Data Entry with Analysis$": "Data Analyst",
    r"^Data Strategist$": "Data Analyst",
    r"^Reports Analyst$": "Data Analyst",
    r"^Reports$": "Data Analyst",
    r"^Reports Developer$": "Data Analyst",
    r"^A mix of Data Engineering and Analysis$": "Data Engineering",
    r"^Water supply engineer, data management, power platform$": "Water supply engineer",
    r"^current role – autocad drafter$": "Autocad Drafter",
    r"^providing support to end users$": "Application Support",
    r"^Previously in management\. Currently in content production$": "Content Production",
}

# Role groups (lookup)
ROLE_GROUP_LOOKUP = {
    "Data Analysis & Insights": {"Data Analyst", "Business Analyst", "BI Analyst", "Insights Analyst"},
    "Data & Software Engineering": {"Data Engineer", "Software Engineer", "ML Engineer", "Data Engineering"},
    "Management & Leadership": {"Manager", "Team Lead", "Director", "Head of Data"},
    "Data Science & Research": {"Data Scientist", "Researcher", "ML Researcher"},
    "Technical Support & IT Operations": {"Application Support", "IT Support", "SysAdmin", "DevOps"},
    "Data Processing & Entry": {"Data Entry", "Encoder"},
    "Customer Service & Operations": {"Customer Support", "Operations"},
    "Other Specialized Roles": {"Autocad Drafter", "Water supply engineer", "Content Production", "Admin"},
}

# Non-cloud and cloud platform standardizations
STANDARDIZE_CANONICAL = {
    # Non-cloud examples
    r"^(SSMS|Ssms)$": "SSMS",
    r"^(SQL SERVER|MS SWL Server|Microsoft SQL Server)$": "MS SQL Server",
}

# Regular tools (languages) standardizations
REGULAR_TOOLS_STANDARDIZE = {
    r"^(None|none|NA|none yet|None right now|not daily basis|beginner)$": None,
    r"^(PivotChart|Microsoft Excel|excel|using Excel most of the time)$": "Excel",
    r"^(tableau|Tableau|Tableau calculation)$": "Tableau",
}

# AI tools standardizations
AI_TOOLS_STANDARDIZE = {
    r"^No, I do not use AI currently(,.*)?$": "None",
    r"^Chatgpt( .*)?$": "Chatgpt",
}

# Hosted notebooks standardizations
HOSTED_NOTEBOOKS_STANDARDIZE = {
    r"^(onenote|One Note|microsoft one note)$": "OneNote",
    r"^We use regular offline notebook$": None,
    r"^Amazon Sagemaker Studio Lab$": "Amazon Sagemaker Studio",
    r"^(I use none|I am not using any of the above|I currently have no knowledge of this|No\. Just local|not yet|No idea|n/?a|NA|NO|Nope|Microsoft Word)$": "None",
}

# Hardware standardizations
HARDWARE_STANDARDIZE = {
    r"^(Desktop and Laptop|Both laptop and desktop)$": "laptop and desktop",
    r"^N/?A$": None,
    r"^Phone$": "mobile phone",
}

# Digital learning standardizations
DIGITAL_LEARNING_STANDARDIZE = {
    r".*(Youtube|youtube|YouTube).*": "Youtube",
    r"^(great learning|Great Learning|Great learning free certification)$": "Great Learning",
    r"^Not currently using any digital tools for learning$": "None",
}
DIGITAL_LEARNING_CLEAN = {
    "/": ","
}

# DE tools – ingestion
INGESTION_STANDARDIZE = {
    r"^(Python|python|Python jobs|In house tools developed using python|Python scripts|Random throw away python scripts)$": "Python",
    r"^(Custom scripts|Custom|Tools developed internally)$": "Custom scripts",
    r"^(Google sheet|Google sheets|Google Sheets)$": "Google sheets",
    r"^(MS Excel|Only excel|Microsoft Excel|Excel)$": "Excel",
    r"^(Adf|ADF)$": "Azure Data Factory",
    r"^(AWS Glue|Glue)$": "AWS Glue",
    r"^(Don’t know/ None|Don’t know|Not needed|N/?A|None)$": None,
}
INGESTION_REMOVE_IF_OTHERS_PRESENT = {"Don’t know", "None"}

# DE tools – transformation
TRANSFORMATION_STANDARDIZE = {
    r"^Xstl$": "Xslt",
    r"^Power BI$": "Power Query",
}

# Multi-select parsing
MULTI_SELECT_DELIMS = r"\s*,\s*"

# Values treated as blanks consistently
CANONICAL_BLANKS = {
    "", " ", "N/A", "NA", "Na", "n/a", "None", "NONE", "none"
}

# -----------------------------
# Helper functions
# -----------------------------
def is_blank(x):
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    s = str(x).strip()
    return s in CANONICAL_BLANKS

def normalize_text(s):
    if is_blank(s):
        return None
    return re.sub(r"\s+", " ", str(s)).strip()

def apply_regex_map(value, regex_map):
    val = normalize_text(value)
    if val is None:
        return None
    for pattern, repl in regex_map.items():
        if re.match(pattern, val, flags=re.IGNORECASE):
            return repl
    return val

def map_exact(value, mapping):
    val = normalize_text(value)
    if val is None:
        return None
    return mapping.get(val, val)

def split_multi(s):
    s = normalize_text(s)
    if s is None:
        return []
    parts = re.split(MULTI_SELECT_DELIMS, s)
    return [p.strip() for p in parts if normalize_text(p) is not None]

def generate_resp_id(n):
    return [str(uuid.uuid4()) for _ in range(n)]

def fill_from(df, target, sources):
    df[target] = df[target].where(~df[target].isna(), None)
    for src in sources:
        df[target] = df[target].mask(df[target].isna(), df[src])
    return df

def enforce_age_rules(df):
    age_col = COLUMN_MAP["age"]
    edu_col = COLUMN_MAP["education"]
    # bounds
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    df.loc[(df[age_col] < AGE_MIN) | (df[age_col] > AGE_MAX), age_col] = np.nan
    # replace one 0 with 18 if Secondary High School; keep general rule: any 0 + secondary HS -> 18
    df.loc[(df[age_col] == 0) & df[edu_col].str.contains("Secondary High School", case=False, na=True), age_col] = 18
    return df

def assign_age_group(a):
    try:
        a = float(a)
    except Exception:
        return np.nan
    for label, rule in AGE_GROUPS:
        if rule(a):
            return label
    return np.nan

def normalize_city(df):
    city = COLUMN_MAP["city"]
    # replace specific sequences
    for bad, good in CITY_REPLACEMENTS.items():
        df[city] = df[city].astype(str).str.replace(bad, good, regex=False)
    # None -> Perth Australia
    df[city] = df[city].apply(lambda x: CITY_NONE_TO if normalize_text(x) == "None" else x)
    # title case for all-caps like CATBALOGAN handled in replacements; general capitalization pass if all caps
    df[city] = df[city].apply(lambda s: s.title() if isinstance(s, str) and s.isupper() else s)
    # fill from province, then country if city missing
    df = fill_from(df, city, [COLUMN_MAP[p] for p in CITY_FILL_FROM])
    return df

def impute_coords(df):
    city, prov, ctry, lat, lon = (COLUMN_MAP["city"], COLUMN_MAP["province"], COLUMN_MAP["country"], COLUMN_MAP["lat"], COLUMN_MAP["lon"])
    for idx, row in df.iterrows():
        if pd.isna(row.get(lat)) or pd.isna(row.get(lon)):
            keys = [str(row.get(city) or ""), str(row.get(prov) or ""), str(row.get(ctry) or "")]
            keys = [k.lower() for k in keys]
            hit = None
            for k in keys:
                for im_key, payload in IMPUTE_COORDS_BY_PLACE.items():
                    if im_key in k:
                        hit = payload
                        break
                if hit:
                    break
            if hit:
                if is_blank(row.get(city)):
                    df.at[idx, city] = hit["city"]
                df.at[idx, lat] = hit["lat"]
                df.at[idx, lon] = hit["lon"]
    return df

def deduplicate(df):
    ts = COLUMN_MAP["timestamp"]
    city = COLUMN_MAP["city"]
    # Delete exact city rows per doc
    df = df[~df[city].isin(DELETE_CITY_EXACT)].copy()
    # Keep later timestamp for duplicate respondents based on a similarity key
    # Use a subset of columns with “too many similarities” as proxy key
    key_cols = [
        COLUMN_MAP["city"],
        COLUMN_MAP["age"],
        COLUMN_MAP["gender"],
        COLUMN_MAP["education"],
        COLUMN_MAP["industry"],
        COLUMN_MAP["career_stage"],
    ]
    # If timestamp missing, create a surrogate order
    if ts not in df.columns:
        df["_order"] = np.arange(len(df))
        ts_sort = ["_order"]
    else:
        ts_sort = [ts]
    df = df.sort_values(ts_sort).drop_duplicates(subset=key_cols, keep="last")
    df = df.drop(columns=[c for c in ["_order"] if c in df.columns])
    return df

def recode_career_stage(s):
    s1 = apply_regex_map(s, CAREER_STAGE_NORMALIZE)
    if s1 in CAREER_STAGE_ALLOWED:
        return s1
    # keep Freelance if exact
    if normalize_text(s1) == "Freelance":
        return "Freelance"
    if s1 is None:
        return None
    # otherwise Other
    return CAREER_STAGE_DEFAULT_OTHER

def set_working_section_blanks_for_students(df):
    cs = COLUMN_MAP["career_stage_clean"]
    # “If Student/ New grad/ Career break, delete the responses (working section)”
    working_cols = [
        COLUMN_MAP["employer_type"],
        COLUMN_MAP["type_work"],
        COLUMN_MAP["site_work"],
        COLUMN_MAP["salary"],
        COLUMN_MAP["salary_short"],
        COLUMN_MAP["data_role_main"],
        COLUMN_MAP["data_role_other"],
        COLUMN_MAP["team_size"],
        COLUMN_MAP["cloud_platforms"],
        COLUMN_MAP["noncloud_platforms"],
    ]
    for col in working_cols:
        if col in df.columns:
            df.loc[df[cs] == "Student/ New grad/ Career Break", col] = np.nan
    return df

def recode_employer_type(s):
    return apply_regex_map(s, EMPLOYER_RECODE)

def recode_sitework(s):
    return apply_regex_map(s, SITEWORK_RECODE)

def standardize_with_map(s, mapping):
    return apply_regex_map(s, mapping)

def explode_multiselect(df, col_in, id_col, col_out_name):
    out = (
        df[[id_col, col_in]]
        .assign(_vals=lambda d: d[col_in].apply(split_multi))
        .explode("_vals")
        .dropna(subset=["_vals"])
        .rename(columns={"_vals": col_out_name})
        [[id_col, col_out_name]]
        .reset_index(drop=True)
    )
    return out

def success_method_clean(df):
    idcol = COLUMN_MAP["resp_id"]
    col = COLUMN_MAP["success_method"]
    # split to rows
    sm = explode_multiselect(df, col, idcol, "success_method_raw")

    # drop BLANK-like values
    sm["success_method_raw_norm"] = sm["success_method_raw"].apply(normalize_text)
    sm = sm[~sm["success_method_raw_norm"].isin({normalize_text(x) for x in SUCCESS_METHOD_DROP_TO_BLANK})].copy()

    # recode
    sm["success_method"] = sm["success_method_raw"].apply(lambda s: apply_regex_map(s, SUCCESS_METHOD_MAP))

    # Keep only allowed + original explicit platforms
    sm["success_method"] = sm["success_method"].apply(lambda s: s if s in SUCCESS_METHOD_TO_KEEP or s in {"LinkedIn","Indeed","Glassdoor","Upwork","Onlinejobs.ph","networking"} else s)
    sm = sm.dropna(subset=["success_method"]).drop(columns=["success_method_raw","success_method_raw_norm"]).drop_duplicates()
    return sm

def fill_typework_unspecified_if_salary(df):
    tcol = COLUMN_MAP["type_work"]
    sal = COLUMN_MAP["salary"]
    if tcol in df.columns and sal in df.columns:
        df[tcol] = df.apply(lambda r: "Unspecified" if is_blank(r[tcol]) and not is_blank(r[sal]) else r[tcol], axis=1)
    return df

def clean_data_role(df):
    main = COLUMN_MAP["data_role_main"]
    other = COLUMN_MAP["data_role_other"]
    # main standardization
    if main in df.columns:
        df[main] = df[main].apply(lambda s: apply_regex_map(s, DATA_ROLE_STANDARDIZE))
    # rest of role: split, drop vague/long or duplicates vs main
    idcol = COLUMN_MAP["resp_id"]
    if other in df.columns:
        rr = explode_multiselect(df, other, idcol, "role_other_raw")
        # Drop vague/long-winded: keep only up to 4 words as a heuristic for specificity, else blank
        rr["role_other_raw"] = rr["role_other_raw"].apply(lambda x: x if len(x.split()) <= 6 else None)
        rr = rr.dropna(subset=["role_other_raw"])
        # Remove duplicates already in main
        main_map = df[[idcol, main]].rename(columns={main: "role_main"})
        rr = rr.merge(main_map, on=idcol, how="left")
        rr = rr[rr["role_other_raw"].str.lower() != rr["role_main"].astype(str).str.lower()]
        rr = rr[[idcol, "role_other_raw"]].drop_duplicates()
    else:
        rr = pd.DataFrame(columns=[COLUMN_MAP["resp_id"], "role_other_raw"])
    return df, rr

def assign_role_group(role_value):
    if role_value is None or (isinstance(role_value, float) and np.isnan(role_value)):
        return None
    for grp, items in ROLE_GROUP_LOOKUP.items():
        if role_value in items:
            return grp
    return "Other Specialized Roles" if role_value else None

def blank_team_size_if_no_role(df):
    role = COLUMN_MAP["data_role_main"]
    team = COLUMN_MAP["team_size"]
    if role in df.columns and team in df.columns:
        df.loc[df[role].isna(), team] = np.nan
    return df

def standardize_series_by_map_regex(ser, mapping):
    return ser.apply(lambda s: apply_regex_map(s, mapping))

def standardize_series_by_exact(ser, mapping):
    return ser.apply(lambda s: map_exact(s, mapping))

def standardize_text_replacements(ser, replacements):
    out = ser.astype(str)
    for old, new in replacements.items():
        out = out.str.replace(old, new, regex=False)
    return out

def remove_none_like_in_comms(df):
    col = COLUMN_MAP["comms"]
    if col in df.columns:
        df[col] = df[col].apply(lambda s: None if normalize_text(s) in {"None", "None at the moment", "Nothing", "n/a", "N/A", "NA"} else s)
    return df

def make_in_phils_flag(df):
    country = COLUMN_MAP["country"]
    flag = COLUMN_MAP["in_phils_flag"]
    df[flag] = df[country].apply(lambda s: True if normalize_text(s) in IN_PHILS_TRUE else False)
    return df

# -----------------------------
# Main pipeline
# -----------------------------
def process(df):
    cm = COLUMN_MAP

    # 0) Add resp_id if missing
    if cm["resp_id"] not in df.columns:
        df[cm["resp_id"]] = generate_resp_id(len(df))

    # 1) Remove or blank out email (anonymize)
    if cm["email"] in df.columns:
        df = df.drop(columns=[cm["email"]])

    # 2) Blanks: accept as blanks (handled through normalize_text/is_blank usage)

    # 3) Age rules
    df = enforce_age_rules(df)

    # 4) Gender: blanks -> Prefer not to say
    if cm["gender"] in df.columns:
        df[cm["gender"]] = df[cm["gender"]].apply(lambda s: GENDER_BLANK_TO if is_blank(s) else s)

    # 5) City normalization and fills
    df = normalize_city(df)

    # 6) Coordinates imputation rules
    df = impute_coords(df)

    # 7) Duplicates: keep later timestamp; delete specific city rows
    df = deduplicate(df)

    # 8) Salary shortversion
    if cm["salary"] in df.columns:
        df[cm["salary_short"]] = df[cm["salary"]].apply(lambda s: map_exact(s, SALARY_BINS_MAP))

    # 9) Age group
    df[cm["age_group"]] = df[cm["age"]].apply(assign_age_group)

    # 10) Career stage cleaned
    df[cm["career_stage_clean"]] = df[cm["career_stage"]].apply(recode_career_stage)

    # 11) In-Philippines flag
    df = make_in_phils_flag(df)

    # 12) Working section blanks for students/new grads/career break
    df = set_working_section_blanks_for_students(df)

    # 13) Employer type recode
    if cm["employer_type"] in df.columns:
        df[cm["employer_type"]] = df[cm["employer_type"]].apply(recode_employer_type)

    # 14) Success method split + clean (separate table)
    success_tbl = success_method_clean(df)

    # 15) Typework unspecified if salary present but typework blank
    df = fill_typework_unspecified_if_salary(df)

    # 16) Sitework normalization
    if cm["site_work"] in df.columns:
        df[cm["site_work"]] = df[cm["site_work"]].apply(recode_sitework)

    # 17) Data role main + rest of role split
    df, rest_role_tbl = clean_data_role(df)

    # 18) Role group (lookup on main role)
    if cm["data_role_main"] in df.columns:
        df["role_group"] = df[cm["data_role_main"]].apply(assign_role_group)

    # 19) Blank team size if data role is blank
    df = blank_team_size_if_no_role(df)

    # 20) Non-cloud and cloud platform standardizations
    for col in [cm["noncloud_platforms"], cm["cloud_platforms"]]:
        if col in df.columns:
            df[col] = standardize_series_by_map_regex(df[col], STANDARDIZE_CANONICAL)

    # 21) General tools multi-select split table
    general_tools_tbl = explode_multiselect(df, cm["general_tools"], cm["resp_id"], "general_tool") if cm["general_tools"] in df.columns else pd.DataFrame()

    # 22) Regular tools standardizations (split table)
    if cm["regular_tools"] in df.columns:
        reg_tbl = explode_multiselect(df, cm["regular_tools"], cm["resp_id"], "regular_tool")
        reg_tbl["regular_tool"] = reg_tbl["regular_tool"].apply(lambda s: apply_regex_map(s, REGULAR_TOOLS_STANDARDIZE))
        reg_tbl = reg_tbl.dropna(subset=["regular_tool"]).drop_duplicates()
    else:
        reg_tbl = pd.DataFrame()

    # 23) AI tools split + standardize; handle “No, I do not use AI currently”
    if cm["ai_tools"] in df.columns:
        ai_tbl = explode_multiselect(df, cm["ai_tools"], cm["resp_id"], "ai_tool_raw")
        ai_tbl["ai_tool"] = ai_tbl["ai_tool_raw"].apply(lambda s: apply_regex_map(s, AI_TOOLS_STANDARDIZE))
        # When “No, I do not use AI currently” appears with real tools during parsing, keep the real tools; “None” collapses to a single None row we drop.
        ai_tbl = ai_tbl[ai_tbl["ai_tool"].notna()].drop(columns=["ai_tool_raw"]).drop_duplicates()
    else:
        ai_tbl = pd.DataFrame()

    # 24) Hosted notebooks split + standardize
    if cm["hosted_notebooks"] in df.columns:
        hn_tbl = explode_multiselect(df, cm["hosted_notebooks"], cm["resp_id"], "hosted_notebook_raw")
        hn_tbl["hosted_notebook"] = hn_tbl["hosted_notebook_raw"].apply(lambda s: apply_regex_map(s, HOSTED_NOTEBOOKS_STANDARDIZE))
        hn_tbl = hn_tbl[hn_tbl["hosted_notebook"].notna()].drop(columns=["hosted_notebook_raw"]).drop_duplicates()
    else:
        hn_tbl = pd.DataFrame()

    # 25) Hardware standardize (single-select)
    if cm["hardware"] in df.columns:
        df[cm["hardware"]] = df[cm["hardware"]].apply(lambda s: apply_regex_map(s, HARDWARE_STANDARDIZE))

    # 26) Digital learning: replace "/" with "," then split and standardize
    if cm["digital_learning"] in df.columns:
        df[cm["digital_learning"]] = standardize_text_replacements(df[cm["digital_learning"]], DIGITAL_LEARNING_CLEAN)
        dl_tbl = explode_multiselect(df, cm["digital_learning"], cm["resp_id"], "digital_learning_raw")
        dl_tbl["digital_learning"] = dl_tbl["digital_learning_raw"].apply(lambda s: apply_regex_map(s, DIGITAL_LEARNING_STANDARDIZE))
        dl_tbl = dl_tbl[dl_tbl["digital_learning"].notna()].drop(columns=["digital_learning_raw"]).drop_duplicates()
    else:
        dl_tbl = pd.DataFrame()

    # 27) DEP: website/resources – no transforms

    # 28) Special needs: split text to rows, trim, clean
    if cm["sp_needs"] in df.columns:
        sp_tbl = explode_multiselect(df, cm["sp_needs"], cm["resp_id"], "special_need")
    else:
        sp_tbl = pd.DataFrame()

    # 29) Volunteer: split and clean; remove None-like
    if cm["volunteer"] in df.columns:
        vol_tbl = explode_multiselect(df, cm["volunteer"], cm["resp_id"], "volunteer_pref")
        vol_tbl["volunteer_pref"] = vol_tbl["volunteer_pref"].apply(lambda s: None if normalize_text(s) in {"None", "NA", "N/A"} else s)
        vol_tbl = vol_tbl.dropna(subset=["volunteer_pref"]).drop_duplicates()
    else:
        vol_tbl = pd.DataFrame()

    # 30) Comms: delete None-like suggestions
    df = remove_none_like_in_comms(df)

    # 31) DE Tools – ingestion: split and standardize; drop “Don’t know”/“None” when others present
    if "ingestion" in df.columns:
        ing_tbl = explode_multiselect(df, "ingestion", cm["resp_id"], "ingestion_raw")
        ing_tbl["ingestion"] = ing_tbl["ingestion_raw"].apply(lambda s: apply_regex_map(s, INGESTION_STANDARDIZE))
        ing_tbl = ing_tbl.dropna(subset=["ingestion"])
        # If a respondent has other entries, drop Don’t know / None for that respondent
        to_drop = ing_tbl[ing_tbl["ingestion"].isin(INGESTION_REMOVE_IF_OTHERS_PRESENT)][cm["resp_id"]]
        keep_ids = ing_tbl.groupby(cm["resp_id"])["ingestion"].nunique()
        has_others = keep_ids[keep_ids > 1].index
        ing_tbl = ing_tbl[~(ing_tbl[cm["resp_id"]].isin(has_others) & ing_tbl["ingestion"].isin(INGESTION_REMOVE_IF_OTHERS_PRESENT))]
        ing_tbl = ing_tbl.drop(columns=["ingestion_raw"]).drop_duplicates()
    else:
        ing_tbl = pd.DataFrame()

    # 32) DE Tools – transformation: split and standardize
    if "transformation" in df.columns:
        tr_tbl = explode_multiselect(df, "transformation", cm["resp_id"], "transformation_raw")
        tr_tbl["transformation"] = tr_tbl["transformation_raw"].apply(lambda s: apply_regex_map(s, TRANSFORMATION_STANDARDIZE))
        tr_tbl = tr_tbl.drop(columns=["transformation_raw"]).drop_duplicates()
    else:
        tr_tbl = pd.DataFrame()

    # 33) FB groups: split to rows
    if cm["fb_groups"] in df.columns:
        fb_tbl = explode_multiselect(df, cm["fb_groups"], cm["resp_id"], "fb_group")
    else:
        fb_tbl = pd.DataFrame()

    # 34) Size: retain only later timestamp already handled; final respondent count is len(df)

    return {
        "main": df.reset_index(drop=True),
        "success_method": success_tbl.reset_index(drop=True),
        "rest_of_role": rest_role_tbl.reset_index(drop=True),
        "general_tools": general_tools_tbl.reset_index(drop=True),
        "regular_tools": reg_tbl.reset_index(drop=True),
        "ai_tools": ai_tbl.reset_index(drop=True),
        "hosted_notebooks": hn_tbl.reset_index(drop=True),
        "digital_learning": dl_tbl.reset_index(drop=True),
        "special_needs": sp_tbl.reset_index(drop=True),
        "volunteer": vol_tbl.reset_index(drop=True),
        "ingestion": ing_tbl.reset_index(drop=True),
        "transformation": tr_tbl.reset_index(drop=True),
        "fb_groups": fb_tbl.reset_index(drop=True),
    }


