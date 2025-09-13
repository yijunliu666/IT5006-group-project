import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr, chi2_contingency

st.set_page_config(page_title="Diabetes Readmission Dashboard", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
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


# Demographics Module
st.header("Demographics Analysis")

age_options = df['age'].unique().tolist()
selected_age = st.multiselect("Select Age Groups:", age_options, default=age_options)

gender_options = df['gender'].unique().tolist()
selected_gender = st.multiselect("Select Gender:", gender_options, default=gender_options)

# Drop NA values in race before options
race_options = df['race'].dropna().unique().tolist()
selected_race = st.multiselect("Select Race:", race_options, default=race_options)

demo_filtered = df[
    (df['age'].isin(selected_age)) &
    (df['gender'].isin(selected_gender)) &
    (df['race'].isin(selected_race))
]

if len(demo_filtered) > 0:
    col1, col2, col3 = st.columns(3)
    
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
        fig_gender = px.pie(gender_dist, names='gender', values='count',
                            title="Gender Distribution", hole=0.3)
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Race distribution (Bar Chart, NA dropped)
    with col3:
        st.subheader("Race Distribution")
        race_dist = demo_filtered['race'].value_counts().reset_index()
        race_dist.columns = ['race', 'count']
        fig_race = px.bar(race_dist, x='race', y='count', color='race',
                          title="Race Distribution")
        st.plotly_chart(fig_race, use_container_width=True)
    
    # Readmission rate
    st.subheader("Readmission Rate for Selected Demographics")
    readmit_demo = demo_filtered['readmitted'].value_counts(normalize=True).reset_index()
    readmit_demo.columns = ['readmitted', 'rate']
    fig_demo = px.bar(readmit_demo, x='readmitted', y='rate', color='readmitted', text='rate',
                      title="Readmission Rate", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_demo, use_container_width=True)
else:
    st.write("No data available for this selection.")


# Hospital Stay Module
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


# Visit History Module
st.header("Visit History Analysis")
visit_type = st.selectbox("Select Visit Type:", ["number_outpatient", "number_emergency", "number_inpatient"])
visit_filtered = df[[visit_type, 'readmitted']]
visit_grouped = visit_filtered.groupby(visit_type)['readmitted'].value_counts(normalize=True).unstack().fillna(0)
fig_visit = px.line(visit_grouped, x=visit_grouped.index, y=visit_grouped.columns,
                    title=f"Readmission Trend by {visit_type} Count", markers=True)
st.plotly_chart(fig_visit, use_container_width=True)


# Medication & Treatment Module
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


# Numeric Feature Analysis Module
st.header("Numeric Feature Analysis")

# Select numeric columns (exclude IDs and mostly categorical)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove IDs or columns not meaningful for analysis
for col in ['encounter_id', 'patient_nbr',
            'number_outpatient', 'number_emergency', 'number_inpatient']:
    if col in numeric_cols:
        numeric_cols.remove(col)

selected_numeric = st.selectbox("Select a Numeric Feature:", numeric_cols)

# Only keep Box Plot now
fig_num = px.box(
    df, x='readmitted', y=selected_numeric,
    color='readmitted',
    points="all",  # show all points
    notched=True,  # notched box
    title=f"Box Plot of {selected_numeric} by Readmission (with all points and mean)",
    labels={'readmitted': 'Readmission Status', selected_numeric: selected_numeric}
)

# Add mean points manually
means = df.groupby('readmitted')[selected_numeric].mean().reset_index()
for i, row in means.iterrows():
    fig_num.add_scatter(
        x=[row['readmitted']], y=[row[selected_numeric]],
        mode='markers+text', marker=dict(color='black', size=12, symbol='diamond'),
        text=[f"Mean: {row[selected_numeric]:.2f}"], textposition="top center", showlegend=False
    )

st.plotly_chart(fig_num, use_container_width=True)


# Correlation Analysis Module
st.header("Correlation Analysis")

# Cramér’s V for categorical variables
cat_features_all = ['race', 'gender', 'age', 'time_in_hospital', 'diag_1', 'diag_2', 'diag_3',
                    'metformin', 'repaglinide','glimepiride','glipizide', 'glyburide',
                    'pioglitazone','rosiglitazone','insulin','change','diabetesMed']

selected_cat_features = st.multiselect("Select Categorical Features for Correlation:",
                                       cat_features_all,
                                       default=cat_features_all)

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1),(rcorr-1)))

# Categorical correlation heatmap
if selected_cat_features:
    st.subheader("Categorical Features Correlation (Cramér’s V)")
    cramers_results = pd.DataFrame(index=selected_cat_features, columns=selected_cat_features)
    for col1 in selected_cat_features:
        for col2 in selected_cat_features:
            cramers_results.loc[col1, col2] = cramers_v(df[col1], df[col2])
    cramers_results = cramers_results.astype(float)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(cramers_results, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Numerical Correlation (Spearman) 
num_features_all = ['number_inpatient', 'number_emergency', 'number_outpatient',
                    'number_diagnoses', 'num_medications', 'time_in_hospital',
                    'num_procedures', 'num_lab_procedures']

selected_num_features = st.multiselect("Select Numerical Features for Correlation:",
                                       num_features_all,
                                       default=num_features_all)

if selected_num_features:
    st.subheader("Numerical Features Correlation (Spearman)")
    spearman_corr = df[selected_num_features].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)  # same cmap as categorical
    st.pyplot(fig)

# Correlation with Readmission
st.subheader("Correlation with Readmission")
df['readmitted_binary'] = df['readmitted'].apply(lambda x: 0 if x=="NO" else 1)

# Categorical vs readmission
cat_readmit_corr = {col: cramers_v(df[col], df['readmitted_binary']) for col in selected_cat_features}
cat_readmit_corr = pd.Series(cat_readmit_corr).sort_values(ascending=True)
fig_cat_corr = px.bar(cat_readmit_corr, x=cat_readmit_corr.values, y=cat_readmit_corr.index,
                      orientation='h', title="Categorical Features vs Readmission (Cramér’s V)",
                      labels={'x':'Cramér’s V', 'y':'Feature'})
st.plotly_chart(fig_cat_corr, use_container_width=True)

# Numerical vs readmission
num_readmit_corr = {col: spearmanr(df[col], df['readmitted_binary'])[0] for col in selected_num_features}
num_readmit_corr = pd.Series(num_readmit_corr).sort_values(ascending=True)
fig_num_corr = px.bar(num_readmit_corr, x=num_readmit_corr.values, y=num_readmit_corr.index,
                      orientation='h', title="Numerical Features vs Readmission (Spearman ρ)",
                      labels={'x':'Spearman ρ', 'y':'Feature'})
st.plotly_chart(fig_num_corr, use_container_width=True)


# Overall Readmission
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
