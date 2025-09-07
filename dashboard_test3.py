import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Diabetes Readmission Dashboard", layout="wide")

# =========================
# Load and preprocess data
# =========================
@st.cache_data
def load_data(url):
    df = pd.read_csv(url, nrows=1000)
    # Fill missing values
    df['weight'] = df['weight'].fillna('Missing')
    df['payer_code'] = df['payer_code'].fillna('Unknown')
    df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')
    df['max_glu_serum'] = df['max_glu_serum'].fillna('Missing')
    df['A1Cresult'] = df['A1Cresult'].fillna('Missing')
    return df

url = "https://archive.ics.uci.edu/static/public/296/data.csv"
df = load_data(url)

st.title("Diabetes Hospital Readmission Dashboard")

# =========================
# Demographics Module
# =========================
st.header("Demographics Analysis")
age_options = df['age'].unique().tolist()
selected_age = st.multiselect("Select Age Groups:", age_options, default=age_options[:3])

gender_options = df['gender'].unique().tolist()
selected_gender = st.multiselect("Select Gender:", gender_options, default=gender_options)

demo_filtered = df[(df['age'].isin(selected_age)) & (df['gender'].isin(selected_gender))]

if len(demo_filtered) > 0:
    col1, col2 = st.columns(2)
    
    # Age distribution
    with col1:
        st.subheader("Age Distribution")
        age_dist = demo_filtered['age'].value_counts().sort_index().reset_index()
        age_dist.columns = ['age', 'count']
        fig_age = px.bar(age_dist, x='age', y='count', title="Age Distribution", color='age')
        st.plotly_chart(fig_age, use_container_width=True)
    
    # Gender distribution
    with col2:
        st.subheader("Gender Distribution")
        gender_dist = demo_filtered['gender'].value_counts().reset_index()
        gender_dist.columns = ['gender', 'count']
        fig_gender = px.pie(gender_dist, names='gender', values='count', title="Gender Distribution", hole=0.3)
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Readmission rate
    st.subheader("Readmission Rate for Selected Demographics")
    readmit_demo = demo_filtered['readmitted'].value_counts(normalize=True).reset_index()
    readmit_demo.columns = ['readmitted', 'rate']
    fig_demo = px.bar(readmit_demo, x='readmitted', y='rate', color='readmitted', text='rate',
                      title="Readmission Rate", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_demo, use_container_width=True)
else:
    st.write("No data available for this selection.")

# =========================
# Hospital Stay Module
# =========================
st.header("Hospital Stay Analysis")
time_min = int(df['time_in_hospital'].min())
time_max = int(df['time_in_hospital'].max())
selected_time = st.slider("Select Length of Stay (days):", time_min, time_max, (time_min, time_max))

time_filtered = df[(df['time_in_hospital'] >= selected_time[0]) & (df['time_in_hospital'] <= selected_time[1])]

if len(time_filtered) > 0:
    col1, col2 = st.columns(2)
    # Distribution
    with col1:
        st.subheader("Length of Stay Distribution")
        fig_stay = px.histogram(time_filtered, x='time_in_hospital', nbins=14,
                                title="Hospital Stay Distribution", color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_stay, use_container_width=True)
    # Readmission
    with col2:
        st.subheader("Readmission Rate by Stay Duration")
        readmit_time = time_filtered['readmitted'].value_counts(normalize=True).reset_index()
        readmit_time.columns = ['readmitted', 'rate']
        fig_time = px.bar(readmit_time, x='readmitted', y='rate', color='readmitted', text='rate',
                          title="Readmission Rate", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_time, use_container_width=True)
else:
    st.write("No data available for this length of stay.")

# =========================
# Visit History Module
# =========================
st.header("Visit History Analysis")
visit_type = st.selectbox("Select Visit Type:", ["number_outpatient", "number_emergency", "number_inpatient"])
visit_filtered = df[[visit_type, 'readmitted']]
visit_grouped = visit_filtered.groupby(visit_type)['readmitted'].value_counts(normalize=True).unstack().fillna(0)
fig_visit = px.line(visit_grouped, x=visit_grouped.index, y=visit_grouped.columns,
                    title=f"Readmission Trend by {visit_type} Count", markers=True)
st.plotly_chart(fig_visit, use_container_width=True)

# =========================
# Medication & Treatment Module
# =========================
st.header("Medication & Treatment Analysis")
# List all drug columns dynamically
drug_columns = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide',
                'glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol',
                'troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin',
                'glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']

selected_drug = st.selectbox("Select a Drug/Medication:", drug_columns, index=drug_columns.index('insulin'))

med_filtered = df[df[selected_drug].isin(['Up','Down','Steady','No'])]

if not med_filtered.empty:
    st.subheader(f"Readmission by {selected_drug}")
    med_grouped = med_filtered.groupby(selected_drug)['readmitted'].value_counts(normalize=True).unstack().fillna(0)
    fig_med = px.bar(med_grouped, x=med_grouped.index, y=med_grouped.columns,
                     title=f"Readmission Rate by {selected_drug} Usage", 
                     barmode='stack', color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_med, use_container_width=True)
else:
    st.write(f"No data available for {selected_drug} usage.")

# =========================
# Overall Readmission
# =========================
st.header("Overall Readmission Overview")
overall_readmit = df['readmitted'].value_counts().reset_index()
overall_readmit.columns = ['readmitted', 'count']

col1, col2 = st.columns(2)
with col1:
    fig_pie = px.pie(overall_readmit, values='count', names='readmitted', hole=0.3,
                     title="Overall Readmission Distribution", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_pie, use_container_width=True)
with col2:
    fig_bar = px.bar(overall_readmit, x='readmitted', y='count', color='readmitted',
                     title="Overall Readmission Bar Chart", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_bar, use_container_width=True)
