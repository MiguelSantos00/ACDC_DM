"""Cleaning helpers for the ACDC EDA notebook (Group 55)."""

import pandas as pd


# ---------------------------------------------------------------- Category spelling

def normalize_country(value):
    if pd.isna(value):
        return value

    value = value.strip().lower()
    value = value.replace("-", " ").replace("_", " ")
    value = " ".join(value.split())

    countries = {
        # Portugal
        "pt": "portugal",
        "portucal": "portugal",

        # Germany
        "de": "germany",
        "deutschland": "germany",

        # United Kingdom
        "uk": "united kingdom",
        "u.k.": "united kingdom",
        "gb": "united kingdom",

        # Ireland
        "ie": "ireland",

        # Spain
        "es": "spain",
        "españa": "spain",
        "espana": "spain",

        # Netherlands
        "nl": "netherlands",
        "the netherlands": "netherlands",
        "holland": "netherlands",

        # France
        "fr": "france",

        # Other
        "other (europe)": "other (europe)"
    }

    return countries.get(value, value)


def normalize_gender(value):
    if pd.isna(value):
        return value

    value = value.strip().lower()

    mapping = {
        "f": "female",
        "m": "male",
        "o": "other"
    }

    return mapping.get(value, value)


def normalize_industry(value):
    if pd.isna(value):
        return value

    value = value.strip().lower()
    value = " ".join(value.replace("-", " ").split())

    mapping = {
        "it": "information technology",
        "tech": "information technology",

        "marketing and pr": "marketing & pr",
        "marketing/pr": "marketing & pr",

        "design and creative": "design & creative",
        "design/creative": "design & creative",

        "realestate": "real estate",
    }

    return mapping.get(value, value)


def normalize_plan_tier(value):
    if pd.isna(value):
        return value

    value = value.strip().lower()
    value = value.replace("_", "").replace(" ", "").replace("-", "")

    mapping = {
        "hotdesk": "hot desk",
        "dedicateddesk": "dedicated desk",
        "fulltime": "full time",
        "privateoffice": "private office",
    }

    return mapping.get(value, value)


# ---------------------------------------------------------------- Outliers

def iqr_summary(df, features, zero_inflated=()):
    """IQR outlier limits and counts per feature. For the zero_inflated features the quartiles
    are computed on the non-zero values only (users of the service)."""
    rows = {}
    for feature in features:
        values = df[feature].dropna()
        reference = values[values > 0] if feature in zero_inflated else values   # users only, for the service variables
        q1, q3 = reference.quantile([0.25, 0.75])
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        flagged = (values < lower) | (values > upper)
        rows[feature] = {"lower limit": lower, "upper limit": upper, "outliers": flagged.sum(),
                         "outliers (%)": flagged.mean() * 100, "99th percentile": values.quantile(0.99), "max": values.max()}
    return pd.DataFrame(rows).T.round(2)
