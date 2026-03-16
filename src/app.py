import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import plotly.express as px

st.set_page_config(page_title="AI Job Market Intelligence", layout="wide")

st.title("🚀 AI Job Market Intelligence Platform")
st.markdown("### Real-time analytics for AI jobs, salaries and skills")

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("jobs.csv")

# -----------------------
# Skill Processing (needed for insights)
# -----------------------
skills_series = df["skills"].dropna().astype(str)

skills_series = skills_series.str.replace("[","",regex=False)\
                               .str.replace("]","",regex=False)\
                               .str.replace("'","",regex=False)

skills_list = ",".join(skills_series).split(",")

skill_counts = Counter(skills_list)

# -----------------------
# AI Market Insights
# -----------------------
st.markdown("## AI Market Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Jobs", len(df))

with col2:
    st.metric("Top Hiring Location", df["location"].value_counts().idxmax())

with col3:
    st.metric("Most In Demand Skill", max(skill_counts, key=skill_counts.get))

# -----------------------
# Salary Processing
# -----------------------
df["min_salary"] = df["salary"].str.extract(r'€([\d,]+)')
df["max_salary"] = df["salary"].str.extract(r'-\s*€([\d,]+)')

df["min_salary"] = df["min_salary"].str.replace(",", "", regex=True).astype(float)
df["max_salary"] = df["max_salary"].str.replace(",", "", regex=True).astype(float)

df["avg_salary"] = (df["min_salary"] + df["max_salary"]) / 2

# -----------------------
# Encode categorical data
# -----------------------
company_encoder = LabelEncoder()
location_encoder = LabelEncoder()

df["company_encoded"] = company_encoder.fit_transform(df["company"].astype(str))
df["location_encoded"] = location_encoder.fit_transform(df["location"].astype(str))

# -----------------------
# Train Model
# -----------------------
model_df = df.dropna(subset=["avg_salary"])

X = model_df[["company_encoded","location_encoded"]]
y = model_df["avg_salary"]

model = LinearRegression()
model.fit(X,y)

# -----------------------
# Sidebar Filters
# -----------------------
st.sidebar.title("Dashboard Filters")

location_filter = st.sidebar.selectbox(
    "Select Location",
    ["All"] + list(df["location"].dropna().unique())
)

industry_filter = st.sidebar.selectbox(
    "Select Industry",
    ["All"] + list(df["industry"].dropna().unique())
)

filtered_df = df.copy()

if location_filter != "All":
    filtered_df = filtered_df[filtered_df["location"] == location_filter]

if industry_filter != "All":
    filtered_df = filtered_df[filtered_df["industry"] == industry_filter]

# -----------------------
# Salary Prediction
# -----------------------
st.header("AI Salary Prediction")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Select Company", df["company"].unique())

with col2:
    location = st.selectbox("Select Location", df["location"].unique())

if st.button("Predict Salary"):

    company_val = company_encoder.transform([company])[0]
    location_val = location_encoder.transform([location])[0]

    prediction = model.predict([[company_val,location_val]])

    st.success(f"Estimated Salary: €{int(prediction[0])}")

st.markdown("---")

# -----------------------
# Top Skills Demand
# -----------------------
st.header("Top Skills in Demand")

top_skills = dict(skill_counts.most_common(10))

st.bar_chart(top_skills)

# -----------------------
# Salary Distribution
# -----------------------
st.header("Salary Distribution")

if not filtered_df["avg_salary"].dropna().empty:

    fig, ax = plt.subplots()

    sns.histplot(filtered_df["avg_salary"].dropna(), bins=30)

    st.pyplot(fig)

else:
    st.warning("No salary data available for selected filters")

# -----------------------
# Salary Trend Analysis
# -----------------------
st.header("Salary Trend Analysis")

salary_trend = df.groupby("location")["avg_salary"].mean().dropna().sort_values(ascending=False).head(10)

fig_trend = px.line(
    salary_trend,
    title="Average Salary Trend by Location",
)

st.plotly_chart(fig_trend)

# -----------------------
# Top Hiring Companies
# -----------------------
st.header("Top Hiring Companies")

top_companies = filtered_df["company"].value_counts().head(10)

st.bar_chart(top_companies)

# -----------------------
# Job Demand by Industry
# -----------------------
st.header("Industry Job Demand")

industry_counts = filtered_df["industry"].value_counts().head(10)

if len(industry_counts) > 0:

    fig2, ax2 = plt.subplots()

    industry_counts.plot(kind="bar", ax=ax2, color="skyblue")

    st.pyplot(fig2)

else:
    st.warning("No data available for the selected filters")

# -----------------------
# Interactive Job Map
# -----------------------
st.header("Global Job Demand Map")

location_counts = df["location"].value_counts().head(20)

map_df = location_counts.reset_index()

map_df.columns = ["location","jobs"]

fig_map = px.bar(
    map_df,
    x="location",
    y="jobs",
    title="Job Demand by Location"
)

st.plotly_chart(fig_map)

# -----------------------
# Dataset Preview
# -----------------------
st.header("Dataset Preview")

st.dataframe(filtered_df.head(100))

st.markdown("---")

# -----------------------
# AI Skill Recommendation
# -----------------------
st.header("🧠 AI Skill Recommendation")

industry_choice = st.selectbox(
    "Select Industry for Skill Recommendation",
    df["industry"].dropna().unique()
)

industry_data = df[df["industry"] == industry_choice]

skills_series2 = industry_data["skills"].dropna().astype(str)

skills_series2 = skills_series2.str.replace("[","",regex=False)\
                               .str.replace("]","",regex=False)\
                               .str.replace("'","",regex=False)

skills_list2 = ",".join(skills_series2).split(",")

skill_counts2 = Counter(skills_list2)

top_skills2 = dict(skill_counts2.most_common(10))

fig3 = px.bar(
    x=list(top_skills2.keys()),
    y=list(top_skills2.values()),
    title="Recommended Skills for This Industry",
    color=list(top_skills2.values()),
    color_continuous_scale="viridis"
)

st.plotly_chart(fig3)

# -----------------------
# Salary Trend Visualization
# -----------------------
st.header("📈 Salary Trend Visualization")

salary_trend2 = df.groupby("industry")["avg_salary"].mean().dropna()

fig4 = px.bar(
    x=salary_trend2.index,
    y=salary_trend2.values,
    title="Average Salary by Industry",
    color=salary_trend2.values,
    color_continuous_scale="plasma"
)

st.plotly_chart(fig4)

# -----------------------
# AI Job Demand Insights
# -----------------------
st.header("🔥 AI Job Demand Insights")

job_demand = df["industry"].value_counts().head(10)

fig5 = px.pie(
    values=job_demand.values,
    names=job_demand.index,
    title="Job Demand by Industry"
)

st.plotly_chart(fig5)

# -----------------------
# Final AI Insight
# -----------------------
st.header("AI Job Market Insight")

top_skill = max(skill_counts, key=skill_counts.get)
top_location = df["location"].value_counts().idxmax()

st.info(
    f"AI Insight: The job market currently shows strong demand for **{top_skill}** skills. "
    f"The highest hiring activity is observed in **{top_location}**."
)
