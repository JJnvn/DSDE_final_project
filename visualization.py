import streamlit as st
import pandas as pd
from sklearn.cluster import DBSCAN
import pydeck as pdk
import matplotlib.pyplot as plt
import plotly.express as px
import textwrap
import redis, json
import streamlit as st, pandas as pd
from sklearn.cluster import DBSCAN
import pydeck as pdk, matplotlib.pyplot as plt, plotly.express as px, textwrap

# Load Data
@st.cache_data
def load_data():
    rd = redis.Redis(encoding="utf‑8", decode_responses=True)

    # grab all row keys that our loader created
    keys = rd.smembers("cleanv3:keys")          # same SET_NAME as above
    if not keys:
        st.stop()   # nothing in Redis; avoids crashing

    # pipeline gets everything in one round‑trip
    pipe = rd.pipeline()
    for k in keys:
        pipe.get(k)
    rows_json = pipe.execute()

    rows = [json.loads(r) for r in rows_json]   # back to dicts
    df = pd.DataFrame(rows)
    return df

def preprocess_data(df):
    df[['longitude', 'latitude']] = df['coords'].str.split(',', expand=True).astype(float)
    df['type_cleaned'] = df['type'].str.strip('{}').str.replace(' ', '')
    return df

def extract_unique_types(df):
    type_lists = df['type'].dropna().apply(lambda x: x.strip('{}').replace(' ', '').split(','))
    unique_types = sorted(set(t for sublist in type_lists for t in sublist if t))
    return unique_types

# Title and header section
st.markdown("<h1 style='text-align: center;'>🚦 Traffy Fondue: Bangkok Issue Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>"
    "An interactive dashboard to visualize, filter, and cluster public issue reports in Bangkok from the Traffy Fondue platform."
    "</p>", 
    unsafe_allow_html=True
)
st.markdown("---")  # Horizontal line separator

raw_df = load_data()
df = preprocess_data(raw_df)

all_types = extract_unique_types(df)

st.sidebar.header('Select Type')
selected_type = st.sidebar.selectbox(
    'Select Issue Type', 
    options = all_types,
    index = 0
)

df_selected = df[df['type_cleaned'].str.contains(selected_type, na=False)].copy()
all_districts = ["All"] + sorted(df_selected['district'].dropna().unique())

st.sidebar.header("Type's Map Settings")
map_style = st.sidebar.selectbox(
    'Select Base Map Style',
    options = ['Dark', 'Light', 'Road', 'Satellite'],
    index = 0
)
MAP_STYLES = {
    'Dark': 'mapbox://styles/mapbox/dark-v10',
    'Light': 'mapbox://styles/mapbox/light-v10',
    'Road': 'mapbox://styles/mapbox/streets-v11',
    'Satellite': 'mapbox://styles/mapbox/satellite-v9'
}

selected_district = st.sidebar.selectbox(
    'Select District',
    options = all_districts,
    index = 0
)
if selected_district != "All":
    df_selected = df_selected[df_selected['district'] == selected_district]

min_samples_value = st.sidebar.slider(
    'DBSCAN min_samples', 
    min_value = 1, 
    max_value = 20, 
    value = 5
)

eps_value = st.sidebar.slider(
    'DBSCAN eps', 
    min_value = 0.001, 
    max_value = 0.05, 
    value = 0.01, 
    step = 0.005
)

# ================== Charts ==================

filtered_df = df.copy()
if selected_district != "All":
    filtered_df = filtered_df[filtered_df["district"] == selected_district]
if selected_type != "All":
    filtered_df = filtered_df[filtered_df["type"].str.contains(selected_type, na=False)]

# Bar: Case count by district
st.subheader("📊 Case Count by District")
district_df = filtered_df[filtered_df["district"].notna()]
district_count = district_df["district"].value_counts().reset_index()
district_count.columns = ["district", "case_count"]
fig_district = px.bar(district_count, x="district", y="case_count", text="case_count")
st.plotly_chart(fig_district, use_container_width=True)

# Bar: Top organizations
st.subheader("🏢 Top Organizations by Case Count")
org_count = filtered_df["organization"].value_counts().nlargest(10).reset_index()
org_count.columns = ["organization", "case_count"]
fig_org = px.bar(org_count, x="organization", y="case_count", text="case_count")
st.plotly_chart(fig_org, use_container_width=True)

st.subheader("📌 Distribution of Issue Types (Top 10 + Others)")

# Count and group types
type_count = filtered_df["type"].value_counts()
top_types = type_count.head(10)
others_sum = type_count[10:].sum()

# Combine into one DataFrame
type_summary = top_types.copy()
if others_sum > 0:
    type_summary["Others"] = others_sum
type_summary = type_summary.reset_index()
type_summary.columns = ["type", "count"]

# Pie chart
fig_type = px.pie(type_summary, names="type", values="count", title="Issue Type Distribution (Top 10 + Others)")
st.plotly_chart(fig_type, use_container_width=True)

# Reopen count
st.subheader("🔁 Reopen Count Histogram")
fig_reopen = px.histogram(
    filtered_df,
    x="count_reopen",
    nbins=20,
    title="Histogram of Case Reopen Counts",
    labels={"count_reopen": "Reopen Count"},
    color_discrete_sequence=["#636EFA"]
)
st.plotly_chart(fig_reopen, use_container_width=True)

if len(df_selected) > 0:
    coords = df_selected[['latitude', 'longitude']]
    db = DBSCAN(eps=eps_value, min_samples=min_samples_value).fit(coords)
    df_selected['cluster'] = db.labels_

    df_selected = df_selected[df_selected['cluster'] != -1]

    clusters_count = df_selected['cluster'].value_counts()
    clusters_count = clusters_count[clusters_count.index != -1]

    unique_clusters = df_selected['cluster'].unique()
    num_clusters = len(unique_clusters)

    colormap = plt.get_cmap('hsv')
    cluster_colors = {cluster: [int(x*255) for x in colormap(i/num_clusters)[:3]]
                        for i, cluster in enumerate(unique_clusters)}

    df_selected['color'] = df_selected['cluster'].map(cluster_colors)

    st.subheader(f"Raw Data's DBSCAN of '{selected_type}' ")

    # Limit comment length for display in tooltips
    # Format tooltip info (short and multi-line)
    df_selected['hover_info'] = df_selected.apply(
    lambda row: f"Cluster {row['cluster']}:\n" + textwrap.shorten(str(row['comment']), width=50, placeholder="..."),
    axis=1
)

    # Tooltip config
    tooltip = {
    "html": "<div style='font-size:10px; max-width:180px; white-space:pre-wrap;'>{hover_info}</div>",
    "style": {
        "backgroundColor": "rgba(255,255,255,0.9)",
        "color": "black",
        "padding": "5px",
        "borderRadius": "4px",
        "boxShadow": "0 0 6px rgba(0,0,0,0.2)",
        }   
    }

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_selected,
        get_position="[longitude, latitude]",
        get_color='color',
        get_radius=200,
        opacity=0.5,
        pickable=True,
        tooltip=tooltip
    )

    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=df_selected,
        get_position="[longitude, latitude]",
        opacity=0.4
    )

    view_state = pdk.ViewState(
        latitude=df_selected['latitude'].mean(),
        longitude=df_selected['longitude'].mean(),
        zoom=10
    )

    st.pydeck_chart(pdk.Deck(
        layers=[scatter_layer, heatmap_layer],
        initial_view_state=view_state,
        map_style=MAP_STYLES[map_style],
        tooltip={"text": "{hover_info}"}
    ))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Summary")
        st.metric(label="จำนวน Ticket", value=len(df_selected))
        st.metric(label="จำนวน Cluster", value=num_clusters)

    with col2:
        top_3_clusters = clusters_count.head(3).reset_index()
        top_3_clusters.columns = ['Cluster ID', 'Ticket Count']

        st.markdown("### 🔝 Top 3 Clusters")
        st.table(top_3_clusters)

    st.markdown("### 💬 Sample Comments from Each Cluster")
    for cluster_id in sorted(df_selected['cluster'].unique()):
        st.markdown(f"#### 🗂️ Cluster {cluster_id}")
        comments = df_selected[df_selected['cluster'] == cluster_id]['comment'].dropna().head(3)
        for comment in comments:
            wrapped = textwrap.fill(comment, width=80)
            st.code(wrapped)
else:
    st.warning("⚠️ no data available for the selected type and district.")

