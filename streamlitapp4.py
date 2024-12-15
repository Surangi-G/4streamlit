
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Streamlit App Title
st.title("Ecosoil Insight AKL: Data Cleaning App")

# File Upload Section
st.header("Upload Dataset")
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    try:
        # Read the uploaded Excel file
        df = pd.read_excel(uploaded_file)
        st.write("### Original Dataset")
        st.dataframe(df)

        # Display basic dataset information
        st.header("Dataset Information")
        st.write("**Shape of the dataset:**", df.shape)
        st.write("**Missing Values in Each Column:**")
        st.write(df.isnull().sum())

        # Validation: Check for critical columns
        critical_columns = ['pH', 'TC %', 'TN %', 'Olsen P', 'AMN', 'BD']
        missing_critical = [col for col in critical_columns if col not in df.columns]
        if missing_critical:
            st.error(f"The following critical columns are missing: {missing_critical}")
            st.stop()

        # Drop rows with missing critical values
        rows_before = len(df)
        df = df.dropna(subset=critical_columns, how='any')
        rows_after = len(df)
        st.write(f"Rows removed due to missing critical values: {rows_before - rows_after}")

        # Check for duplicates
        duplicates = df.duplicated().sum()
        st.write(f"**Number of duplicate rows:** {duplicates}")
        if duplicates > 0:
            st.write(f"**Percentage of duplicate rows:** {duplicates / len(df) * 100:.2f}%")
        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.write("Duplicates removed!")

        # Extract sample count from 'Site No.1'
        if 'Site No.1' in df.columns:
            df['Sample Count'] = df['Site No.1'].str.extract(r'-(\d{2})$').astype(int)
        else:
            st.warning("Column 'Site No.1' is missing. Sample count extraction skipped.")

        # Add period labels
        if 'Year' in df.columns:
            conditions = [
                (df['Year'] >= 1995) & (df['Year'] <= 2000),
                (df['Year'] >= 2008) & (df['Year'] <= 2012),
                (df['Year'] >= 2013) & (df['Year'] <= 2017),
                (df['Year'] >= 2018) & (df['Year'] <= 2023)
            ]
            period_labels = ['1995-2000', '2008-2012', '2013-2017', '2018-2023']
            df['Period'] = np.select(conditions, period_labels, default='Unknown')
        else:
            st.warning("Column 'Year' is missing. Period assignment skipped.")

        # Keep latest sample count for each site-period
        if 'Site Num' in df.columns and 'Period' in df.columns:
            df = df.loc[df.groupby(['Site Num', 'Period'])['Sample Count'].idxmax()].reset_index(drop=True)
        else:
            st.warning("Columns 'Site Num' or 'Period' are missing. Filtering latest samples skipped.")

        # Replace '<' values
        columns_with_less_than = [col for col in df.columns if df[col].astype(str).str.contains('<').any()]
        for column in columns_with_less_than:
            df[column] = df[column].apply(lambda x: float(x[1:]) / 2 if isinstance(x, str) and x.startswith('<') else x)

        # Imputation using IterativeImputer (only for numerical columns)
        non_predictive_columns = ['Site No.1', 'Site Num', 'Year', 'Sample Count', 'Period']
        df_for_imputation = df.drop(columns=non_predictive_columns, errors='ignore')
        numerical_columns = df_for_imputation.select_dtypes(include=['number']).columns.tolist()

        imputer = IterativeImputer(max_iter=10, random_state=0)
        imputed_data = imputer.fit_transform(df_for_imputation[numerical_columns])
        df_imputed = pd.DataFrame(imputed_data, columns=numerical_columns)

        # Reattach non-imputed columns to the imputed dataset
        df_final = pd.concat([df[non_predictive_columns].reset_index(drop=True), df_imputed], axis=1)

        # Visualize before and after imputation
        columns_imputed = ['MP-10', 'As', 'Cd', 'Cr', 'Cu', 'Ni', 'Pb', 'Zn']
        columns_imputed = [col for col in columns_imputed if col in df.columns and col in df_final.columns]

        st.header("Column Distribution Before and After Imputation")
        for column in columns_imputed:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[column], color='red', label='Before Imputation', kde=True, bins=30, alpha=0.6, ax=ax)
            sns.histplot(df_final[column], color='green', label='After Imputation', kde=True, bins=30, alpha=0.6, ax=ax)
            plt.title(f"Distribution Comparison: {column}")
            plt.legend()
            st.pyplot(fig)

        # Kolmogorov-Smirnov Test
        st.header("Kolmogorov-Smirnov Test Results")
        ks_results = {}
        for column in columns_imputed:
            before = df[column].dropna()
            after = df_final[column].dropna()
            ks_stat, p_value = ks_2samp(before, after)
            ks_results[column] = {'KS Statistic': ks_stat, 'p-value': p_value}
        ks_results_df = pd.DataFrame(ks_results).T
        st.write(ks_results_df)

        # Contamination Index
        native_means = {
            "As": 6.2, "Cd": 0.375, "Cr": 28.5, "Cu": 23.0, "Ni": 17.95, "Pb": 33.0, "Zn": 94.5
        }

        for element, mean_value in native_means.items():
            df_final[f"CI_{element}"] = (df_final[element] / mean_value).round(2)

        ci_columns = [f"CI_{element}" for element in native_means.keys()]
        df_final["ICI"] = df_final[ci_columns].mean(axis=1).round(2)

        def classify_ici(ici):
            if ici <= 1:
                return "Low Contamination"
            elif 1 < ici <= 3:
                return "Moderate Contamination"
            else:
                return "High Contamination"

        df_final["ICI_Class"] = df_final["ICI"].apply(classify_ici)
        st.write("### Final Dataset with Contamination Index")
        st.dataframe(df_final)

        # File Download
        st.header("Download Cleaned Dataset")
        from io import BytesIO
        buffer = BytesIO()
        df_final.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)

        st.download_button(
            label="Download as Excel",
            data=buffer,
            file_name="cleaned_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload a dataset to start the cleaning process.")
