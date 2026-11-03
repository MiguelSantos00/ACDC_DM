"""Chart helpers and project constants for the ACDC EDA notebook (Group 55)."""

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Colours kept consistent with the rest of the project
MAIN_COLOR = "#0062FF"
TIER_ORDER = ["hot desk", "flex", "dedicated desk", "full time", "private office"]
TIER_COLORS = dict(zip(TIER_ORDER, ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]))   # light = cheaper plan, dark = premium

ADDON_COLUMNS = ["uses_meeting_rooms", "uses_event_space", "uses_guest_passes", "uses_print", "uses_mail_handling",
                 "uses_phone_booths", "uses_day_passes", "uses_storage", "uses_bike_locker"]


# ---------------------------------------------------------------- Distributions

def plot_histograms(df, features, title, discrete_features=(), log_counts=True, n_cols=3):
    """Plot one histogram per feature in a grid with n_cols columns."""
    n_rows = ceil(len(features) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), layout="constrained", squeeze=False)
    axes = axes.flatten()

    for i, (ax, feature) in enumerate(zip(axes, features)):
        # Whole-number variables get one bar per value; the others are split into 20 bins
        sns.histplot(data=df, x=feature, bins=20, discrete=feature in discrete_features, ax=ax,
                     color=MAIN_COLOR, edgecolor="black", alpha=1)

        y_label = "Members"
        if log_counts:
            ax.set_yscale("log")      # keeps values held by only a few members visible
            y_label += " (log scale)"

        ax.set_title(feature, fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel(y_label if i % n_cols == 0 else "")   # y label only on the first column

    # Hide the empty panels left in the grid
    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=16)
    plt.show()


# ---------------------------------------------------------------- Relationships between variables

def chart_view(df):
    """Copy of the data with the few helper columns the charts need. Works before and after the cleaning."""
    view = df.copy()
    view["plan_tier"] = pd.Categorical(view["plan_tier"], categories=TIER_ORDER, ordered=True)
    view["active"] = view["membership_end"].isna() | (view["membership_end"] == pd.Timestamp("1970-01-01"))
    # Share of the visits the plan allowed that were actually used (1 = came every day the plan allows)
    view["utilisation"] = (view["actual_visits_last_period"] / view["entitled_visits_last_period"]).where(view["entitled_visits_last_period"] > 0)
    view["recency"] = view["days_since_last_visit"].where(view["days_since_last_visit"].between(0, 998))   # leaves out 999 and negatives
    view["ltv_positive"] = view["lifetime_value"].where(view["lifetime_value"] > 0)                        # log scale needs values above 0
    return view


def bar_labels(ax, values, horizontal=True, fmt="{:.1f}%"):
    for pos, value in enumerate(values):
        if horizontal:
            ax.text(value + 0.5, pos, fmt.format(value), va="center", fontsize=9)
        else:
            ax.text(pos, value + 1, fmt.format(value), ha="center", fontsize=10)


def plot_category_shares(df, columns, title, n_cols=3):
    n_rows = ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.2 * n_rows), layout="constrained", squeeze=False)
    axes = axes.flatten()
    for ax, column in zip(axes, columns):
        shares = df[column].value_counts(normalize=True, dropna=False).sort_values() * 100
        labels = ["missing" if pd.isna(label) else str(label) for label in shares.index]
        ax.barh(labels, shares.values, color=MAIN_COLOR, edgecolor="black", height=0.6)
        bar_labels(ax, shares.values)
        ax.set_title(column, fontsize=13)
        ax.set_xlim(0, shares.max() * 1.18)
    for ax in axes[len(columns):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=16)
    plt.show()


def plot_addon_adoption(df):
    share = df[ADDON_COLUMNS].mean().sort_values() * 100
    share.index = share.index.str.replace("uses_", "").str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.barh(share.index, share.values, color=MAIN_COLOR, edgecolor="black", height=0.6)
    bar_labels(ax, share.values)
    ax.set_xlabel("Members who use the service (%)")
    ax.set_title("Adoption of each add-on service", fontsize=13)
    plt.show()
    n_used = df[ADDON_COLUMNS].sum(axis=1)
    return (n_used.value_counts(normalize=True).sort_index() * 100).round(1).to_frame("% of members").rename_axis("Add-ons used")


def plot_plan_overview(view):
    palette = [TIER_COLORS[t] for t in TIER_ORDER]
    box_style = dict(showfliers=False, width=0.55, linecolor="black", medianprops={"color": "black", "linewidth": 2})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained")

    sns.boxplot(data=view, x="plan_tier", y="ltv_positive", order=TIER_ORDER, hue="plan_tier", palette=palette, legend=False, ax=axes[0], **box_style)
    axes[0].set_yscale("log")
    axes[0].set_title("Lifetime value per plan (€, log scale)", fontsize=13)

    active_share = view.groupby("plan_tier", observed=True)["active"].mean() * 100
    axes[1].bar(active_share.index.astype(str), active_share.values, color=palette, edgecolor="black", width=0.6)
    bar_labels(axes[1], active_share.values, horizontal=False, fmt="{:.0f}%")
    axes[1].set_title("Members still active (%)", fontsize=13)

    sns.boxplot(data=view, x="plan_tier", y="utilisation", order=TIER_ORDER, hue="plan_tier", palette=palette, legend=False, ax=axes[2], **box_style)
    axes[2].set_title("Share of entitled visits used (last period)", fontsize=13)

    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=20)
    plt.show()

    return view.groupby("plan_tier", observed=True).agg(
        members=("plan_tier", "size"),
        median_lifetime_value=("ltv_positive", "median"),
        active_pct=("active", lambda s: s.mean() * 100),
        median_utilisation=("utilisation", "median"),
        median_renewals=("n_renewals", lambda s: s[s >= 0].median()),
    ).round(2)


def plot_addons_by_plan(view):
    table = view.groupby("plan_tier", observed=True)[ADDON_COLUMNS].mean().T * 100
    table.index = table.index.str.replace("uses_", "").str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    sns.heatmap(table, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100, linewidths=1, linecolor="white",
                cbar_kws={"label": "Members using the service (%)"}, ax=ax)
    ax.set_title("Add-on adoption per plan (%)", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.show()


def plot_usage_patterns(view):
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), layout="constrained")

    sns.histplot(view["utilisation"].clip(upper=1.2), bins=24, color=MAIN_COLOR, edgecolor="black", alpha=1, ax=axes[0])
    axes[0].axvline(1, color="#ff6666", linestyle="--", linewidth=1.5)
    axes[0].set_title("Share of entitled visits used in the last period", fontsize=13)
    axes[0].set_xlabel("Actual / entitled visits (1 = came every day the plan allows)")
    axes[0].set_ylabel("Members")

    active = view[view["active"] & view["avg_weekly_visits"].between(0, 7)]
    hb = axes[1].hexbin(active["avg_weekly_visits"], active["recency"].clip(upper=400), gridsize=35, cmap="Blues", mincnt=1, bins="log")
    fig.colorbar(hb, ax=axes[1], label="Members (log scale)")
    axes[1].set_title("Active members: visit frequency vs days since last visit", fontsize=13)
    axes[1].set_xlabel("Average weekly visits")
    axes[1].set_ylabel("Days since last visit (capped at 400)")
    plt.show()

    groups = pd.cut(view["utilisation"], [-0.01, 0.2, 0.8, np.inf], labels=["below 20%", "20-80%", "above 80%"])
    return (pd.crosstab(view["plan_tier"], groups, normalize="index") * 100).round(1)


def dormant_members(view, days=90):
    dormant = view["active"] & (view["recency"] > days)
    print(f"Active members: {view['active'].sum()}  |  no visit for more than {days} days: {dormant.sum()} ({dormant.sum() / view['active'].sum():.0%})")
    return (view.loc[dormant, "plan_tier"].value_counts(normalize=True) * 100).round(1).to_frame("% of dormant active members")


def plot_active_vs_ended(view):
    """Compare the members who are still active with the members whose membership ended."""
    status = view["active"].map({True: "Active", False: "Ended"})
    colors = {"Ended": "#b0b0b0", "Active": MAIN_COLOR}
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), layout="constrained", gridspec_kw={"width_ratios": [0.7, 1.1, 1.2]})

    sns.boxplot(x=status, y=view["utilisation"], order=["Ended", "Active"], hue=status, palette=colors, legend=False,
                showfliers=False, width=0.5, linecolor="black", medianprops={"color": "black", "linewidth": 2}, ax=axes[0])
    axes[0].set_title("Share of entitled visits used (last period)", fontsize=13)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    # Share still active within each plan, so that the plan mix does not drive the comparison
    groups = pd.cut(view["utilisation"], [-0.01, 0.2, 0.8, np.inf], labels=["below 20%", "20-80%", "above 80%"])
    active_share = pd.crosstab(view["plan_tier"], groups, values=view["active"], aggfunc="mean") * 100
    sns.heatmap(active_share, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100, linewidths=1, linecolor="white",
                cbar_kws={"label": "Members still active (%)"}, ax=axes[1])
    axes[1].set_title("Members still active (%), by plan and utilisation", fontsize=13)
    axes[1].set_xlabel("Share of entitled visits used (last period)")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", labelrotation=0)

    addons = view.groupby(status)[ADDON_COLUMNS].mean().T * 100
    addons.index = addons.index.str.replace("uses_", "").str.replace("_", " ")
    addons = addons.sort_values("Active")[["Ended", "Active"]]
    addons.plot(kind="barh", color=[colors["Ended"], colors["Active"]], edgecolor="black", linewidth=0.5, width=0.75, ax=axes[2])
    axes[2].set_title("Members using each add-on (%)", fontsize=13)
    axes[2].set_xlabel("% of members")
    axes[2].set_ylabel("")
    axes[2].legend(frameon=False, loc="lower right")
    plt.show()

    return view.assign(status=status, addons_used=view[ADDON_COLUMNS].sum(axis=1)).groupby("status").agg(
        members=("status", "size"),
        median_utilisation=("utilisation", "median"),
        median_addons_used=("addons_used", "median"),
        median_weekly_visits=("avg_weekly_visits", "median"),
        median_days_since_last_visit=("recency", "median"),
    ).reindex(["Ended", "Active"]).round(2)


def plot_profile_vs_plan(view):
    palette = [TIER_COLORS[t] for t in TIER_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), layout="constrained", gridspec_kw={"width_ratios": [1.6, 1]})

    by_industry = pd.crosstab(view["industry"], view["plan_tier"], normalize="index") * 100
    by_industry = by_industry.loc[by_industry[["dedicated desk", "full time", "private office"]].sum(axis=1).sort_values().index]
    by_industry.plot(kind="barh", stacked=True, color=palette, edgecolor="black", linewidth=0.5, width=0.7, ax=axes[0])
    axes[0].set_title("Plan mix per industry (% of members)", fontsize=13)
    axes[0].set_xlabel("% of members")
    axes[0].set_ylabel("")
    axes[0].legend(title="Plan tier", bbox_to_anchor=(1, 1), frameon=False)

    age_groups = pd.cut(view["age"], [17, 25, 35, 45, 100], labels=["18-25", "26-35", "36-45", "46-100"])   # impossible ages left out
    by_age = pd.crosstab(age_groups, view["plan_tier"], normalize="index") * 100
    by_age.plot(kind="bar", stacked=True, color=palette, edgecolor="black", linewidth=0.5, width=0.6, legend=False, ax=axes[1])
    axes[1].set_title("Plan mix per age group (% of members)", fontsize=13)
    axes[1].set_xlabel("Age group")
    axes[1].tick_params(axis="x", labelrotation=0)
    plt.show()

    return pd.crosstab(view["plan_tier"], view["is_team_account"], normalize="index").mul(100).round(1)
