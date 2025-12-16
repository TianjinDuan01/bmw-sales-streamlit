import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Product & Sales Mix Analysis", layout="wide")
st.title("Product & Sales Mix Analysis")


# Load data

@st.cache_data
def load_df():
    return pd.read_csv("BMW sales data (2010-2024).csv")

df = load_df()

# Derived columns

df = df.copy()
df["Revenue"] = df["Price_USD"] * df["Sales_Volume"]

df["Engine_Bin"] = pd.cut(
    df["Engine_Size_L"],
    bins=[0, 1.6, 2.0, 3.0, 5.0],
    labels=["≤1.6L", "1.7–2.0L", "2.1–3.0L", ">3.0L"]
)

df["Usage_Category"] = pd.cut(
    df["Mileage_KM"],
    bins=[0, 20000, 80000, np.inf],
    labels=[
        "Nearly new (≤20k km)",
        "Moderate use (20k–80k km)",
        "High use (>80k km)"
    ]
)

# Sidebar filters

st.sidebar.header("Filters")

year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
year_range = st.sidebar.slider("Year", year_min, year_max, (year_min, year_max))

regions = st.sidebar.multiselect(
    "Region", sorted(df["Region"].unique()),
    default=sorted(df["Region"].unique())
)

models = st.sidebar.multiselect(
    "Model", sorted(df["Model"].unique()),
    default=sorted(df["Model"].unique())
)

fuels = st.sidebar.multiselect(
    "Fuel Type", sorted(df["Fuel_Type"].unique()),
    default=sorted(df["Fuel_Type"].unique())
)

f = df[
    df["Year"].between(*year_range)
    & df["Region"].isin(regions)
    & df["Model"].isin(models)
    & df["Fuel_Type"].isin(fuels)
].copy()


# KPIs
k1, k2, k3 = st.columns(3)
k1.metric("Sales Volume", f"{f['Sales_Volume'].sum():,}")
k2.metric("Revenue", f"${f['Revenue'].sum():,.0f}")
k3.metric("Avg Price", f"${f['Price_USD'].mean():,.0f}")

st.divider()

# (A) Top Models by Revenue
st.subheader("Top Models by Revenue")

top_rev = (
    f.groupby("Model", as_index=False)["Revenue"]
     .sum()
     .sort_values("Revenue", ascending=False)
     .head(8)
     .reset_index(drop=True)
)

top_rev["Rank"] = np.arange(1, len(top_rev) + 1)
top_rev["Share"] = top_rev["Revenue"] / top_rev["Revenue"].sum()

top_rev["Band"] = np.where(top_rev["Rank"] == 1, "Top 1",
                    np.where(top_rev["Rank"] == 2, "Top 2",
                    np.where(top_rev["Rank"] == 3, "Top 3", "Others")))

blue_palette = {
    "Top 1": "#003A70",   # deep blue
    "Top 2": "#005FA3",   # medium-deep blue
    "Top 3": "#0096D6",   # light blue
    "Others": "#BFD7EA"   # very light blue
}

figA = px.bar(
    top_rev,
    x="Model",
    y="Revenue",
    color="Band",
    color_discrete_map=blue_palette,
    text=top_rev["Share"].map(lambda x: f"{x*100:.1f}%"),
    hover_data={"Revenue":":,.0f", "Share":":.2%", "Rank":True, "Band":True},
    category_orders={"Model": top_rev["Model"].tolist()}
)
figA.update_traces(textposition="outside", cliponaxis=False)

figA.update_traces(marker_line_width=1.2, marker_line_color="rgba(0,0,0,0.35)")

figA.update_xaxes(title="", tickangle=25)
figA.update_yaxes(title="Revenue (USD)")

figA.update_layout(
    legend_title_text="",
    hovermode="x unified",
    margin=dict(l=10, r=10, t=30, b=10)
)

st.plotly_chart(figA, use_container_width=True)


# Row 2

c1, c2 = st.columns(2, gap="large")

# (B) Sales Classification Mix by Region
with c1:
    st.subheader("Sales Classification Mix by Region")

    region_class = (
        f.groupby(["Region", "Sales_Classification"], as_index=False)["Sales_Volume"]
         .sum()
    )
    region_class["Share"] = (
        region_class["Sales_Volume"] /
        region_class.groupby("Region")["Sales_Volume"].transform("sum")
    )

    top_regions = (
        f.groupby("Region")["Sales_Volume"].sum()
         .sort_values(ascending=False)
         .head(6)
         .index
    )

    region_class = region_class[region_class["Region"].isin(top_regions)]

    figB = px.bar(
        region_class,
        x="Region",
        y="Share",
        color="Sales_Classification",
        barmode="stack",
        text=region_class["Share"].map(lambda x: f"{x*100:.1f}%")
    )
    figB.update_yaxes(tickformat=".0%")
    st.plotly_chart(figB, use_container_width=True)

# (C) Heatmap: Model × Fuel Type
with c2:
    st.subheader("Sales Heatmap: Model × Fuel Type")

    pivot = f.pivot_table(
        index="Model",
        columns="Fuel_Type",
        values="Sales_Volume",
        aggfunc="sum",
        fill_value=0
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    K = st.slider("How many key cells to label", 3, 25, 10)

    z = pivot.values

    flat_order = np.argsort(z.ravel())[::-1]
    coords = [np.unravel_index(i, z.shape) for i in flat_order[:K]]

    text = np.full(z.shape, "", dtype=object)
    for r, c in coords:
        v = z[r, c]
        if v > 0:
            text[r, c] = f"{v/1e6:.2f}M"

    figC = px.imshow(
        z,
        x=pivot.columns,
        y=pivot.index,
        aspect="auto",
        color_continuous_scale="Blues"
    )

    figC.update_traces(text=text, texttemplate="%{text}", textfont_size=12)

    figC.update_layout(
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(figC, use_container_width=True)


# Row 3

c3, c4 = st.columns(2, gap="large")

# (D) Top Configuration Combinations
with c3:
    st.subheader("Top Configuration Combinations by Sales Volume")

    f["Config"] = (
        f["Model"] + " | " +
        f["Fuel_Type"] + " | " +
        f["Engine_Bin"].astype(str)
    )

    top_cfg = (
        f.groupby("Config", as_index=False)["Sales_Volume"]
         .sum()
         .sort_values("Sales_Volume", ascending=False)
         .head(8)
         .sort_values("Sales_Volume")
    )

    figD = px.bar(
        top_cfg,
        x="Sales_Volume",
        y="Config",
        orientation="h",
        text_auto=".3s"
    )
    st.plotly_chart(figD, use_container_width=True)

# (E) Price Distribution by Usage Category
with c4:
    st.subheader("Price Distribution by Usage Category")

    order = [
        "Nearly new (≤20k km)",
        "Moderate use (20k–80k km)",
        "High use (>80k km)"
    ]

    usage_df = f.dropna(subset=["Usage_Category"]).copy()
    usage_df["Usage_Category"] = pd.Categorical(
        usage_df["Usage_Category"], categories=order, ordered=True
    )

    max_points = st.slider(
        "Points to display (sampled)",
        200, 3000, 1400, step=200
    )

    sampled = (
        usage_df.groupby("Usage_Category", group_keys=False)
        .apply(lambda g: g.sample(
            min(len(g), max(1, max_points // len(order))),
            random_state=7
        ))
    )

    figE = px.box(
        usage_df,
        x="Usage_Category",
        y="Price_USD",
        category_orders={"Usage_Category": order},
        points=False
    )

    figE.update_traces(
        boxmean=True,
        hoveron="boxes"
    )

    figPts = px.strip(
        sampled,
        x="Usage_Category",
        y="Price_USD"
    )

    figPts.update_traces(
        jitter=0.35,
        marker=dict(size=4, opacity=0.25)
    )

    for tr in figPts.data:
        figE.add_trace(tr)

    figE.update_layout(
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10)
    )

    figE.update_yaxes(title="Price (USD)")
    figE.update_xaxes(title="Usage Category")

    st.plotly_chart(figE, use_container_width=True)
