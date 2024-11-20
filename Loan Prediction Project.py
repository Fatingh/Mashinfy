import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Custom Header
st.title("Loan Approval Prediction Dashboard")
st.markdown("### Presented by **Faten Abdulhaimd Al Refaai**")
st.markdown("**For Mashinfy Academy: Data Science Diploma**")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Dataset Overview", "Visualizations", "Analytics", "Model Training"])

# File Upload Section
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type="csv")

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Remove the `id` column if it exists
    if "id" in df.columns:
        df = df.drop(columns=["id"])
        st.sidebar.success("The 'id' column has been removed from the dataset!")

    # Define numeric and categorical columns globally
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # Dynamically calculate insights for the Dashboard
    total_loans = len(df)
    approved_loans = 0
    rejected_loans = 0
    avg_income = "N/A"
    avg_loan_amount = "N/A"
    avg_age = "N/A"
    approval_rate = "N/A"
    rejection_rate = "N/A"

    # Handle loan_status with 0 as Rejected and 1 as Approved
    if "loan_status" in df.columns:
        approved_loans = df[df["loan_status"] == 1].shape[0]
        rejected_loans = df[df["loan_status"] == 0].shape[0]

        if total_loans > 0:
            approval_rate = (approved_loans / total_loans) * 100
            rejection_rate = (rejected_loans / total_loans) * 100

    # Calculate average age
    if "person_age" in df.columns:
        if pd.api.types.is_numeric_dtype(df["person_age"]):
            avg_age = df["person_age"].mean()

    # Calculate average loan amount
    if "loan_amnt" in df.columns:
        if pd.api.types.is_numeric_dtype(df["loan_amnt"]):
            avg_loan_amount = df["loan_amnt"].mean()

    # Calculate average income
    if "person_income" in df.columns:
        if pd.api.types.is_numeric_dtype(df["person_income"]):
            avg_income = df["person_income"].mean()

    if page == "Dashboard":
        st.header("Dashboard Overview")
        st.markdown("### Key Metrics and Insights")
        st.markdown("This section provides an overview of key metrics such as total loans, approval rates, rejection rates, and averages for applicant age, income, and loan amount.")

        # Display Metrics
        st.markdown("#### Quick Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Loans", value=f"{total_loans}")
        with col2:
            st.metric(label="Approved (%)", value=f"{approval_rate:.2f}%" if approval_rate != "N/A" else "N/A")
        with col3:
            st.metric(label="Rejected (%)", value=f"{rejection_rate:.2f}%" if rejection_rate != "N/A" else "N/A")

        st.markdown("#### Quick Insights")
        st.write(f"- **Total Applicants:** {total_loans}")
        st.write(f"- **Average Age:** {avg_age:.2f}" if avg_age != "N/A" else "- **Average Age:** N/A")
        st.write(f"- **Average Income:** {avg_income:.2f}" if avg_income != "N/A" else "- **Average Income:** N/A")
        st.write(f"- **Average Loan Amount:** {avg_loan_amount:.2f}" if avg_loan_amount != "N/A" else "- **Average Loan Amount:** N/A")

        st.markdown("### Navigation")
        st.write("Use the sidebar to navigate to different sections of the app.")

    if page == "Dataset Overview":
        st.header("Dataset Overview and Insights")
        st.markdown("This section provides an overview of the dataset, including its shape, missing values, and summary statistics for continuous and categorical variables.")

        st.write("### Dataset Preview")
        st.write(df.head())
        st.write("### Dataset Info")
        st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        st.write("### Missing Values")
        st.write(df.isnull().sum())

        # Continuous Variables Summary
        st.subheader("Summary Statistics for Continuous Variables")
        if len(numeric_columns) > 0:
            st.write(df[numeric_columns].describe())
        else:
            st.write("No continuous variables found.")

        # Categorical Variables Summary
        st.subheader("Summary Statistics for Categorical Variables")
        if len(categorical_columns) > 0:
            cat_summary = pd.DataFrame({
                "Unique Values": df[categorical_columns].nunique(),
                "Most Frequent Value": df[categorical_columns].mode().iloc[0],
                "Frequency of Most Frequent": df[categorical_columns].apply(lambda x: x.value_counts().iloc[0])
            })
            st.write(cat_summary)
        else:
            st.write("No categorical variables found.")

    if page == "Visualizations":
        st.header("Data Visualizations")
        st.markdown("This section allows you to visualize different aspects of the dataset, such as histograms, boxplots, pie charts, and correlation heatmaps. Use these visualizations to explore relationships and trends in the data.")

        # Histogram
        st.markdown("### Histogram")
        selected_column = st.selectbox("Select numeric column for histogram", numeric_columns)
        if selected_column:
            st.write(f"Histogram for `{selected_column}`")
            fig = px.histogram(df, x=selected_column, title=f"Histogram of {selected_column}")
            st.plotly_chart(fig)

        # Boxplot
        st.markdown("### Boxplot")
        selected_box_col = st.selectbox("Select numeric column for boxplot", numeric_columns)
        if selected_box_col:
            st.write(f"Boxplot for `{selected_box_col}`")
            fig = px.box(df, y=selected_box_col, title=f"Boxplot of {selected_box_col}")
            st.plotly_chart(fig)

        # Pie Chart
        st.markdown("### Pie Chart")
        st.markdown("Visualize the distribution of a categorical variable using a pie chart.")
        selected_pie_col = st.selectbox("Select categorical column for pie chart", categorical_columns)
        if selected_pie_col:
            st.write(f"Pie Chart for `{selected_pie_col}`")
            pie_data = df[selected_pie_col].value_counts().reset_index()
            pie_data.columns = [selected_pie_col, "Count"]
            fig = px.pie(pie_data, names=selected_pie_col, values="Count", title=f"Pie Chart of {selected_pie_col}")
            st.plotly_chart(fig)

        # Correlation Heatmap
        st.markdown("### Correlation Heatmap")
        st.markdown("This heatmap shows the correlations between numeric variables in the dataset. Correlations close to +1 or -1 indicate strong relationships.")
        if numeric_columns:
            corr = df[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    if page == "Analytics":
        st.header("Analytics")
        st.markdown("This section provides advanced visualizations to explore relationships and trends in the dataset. Use these insights to identify patterns and support decision-making.")

        # Loan Status Distribution
        if "loan_status" in df.columns:
            st.subheader("Loan Status Distribution")
            st.markdown("This bar chart shows the distribution of different loan statuses (e.g., Approved, Rejected).")
            status_counts = df["loan_status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig = px.bar(status_counts, x="Status", y="Count", title="Loan Status Distribution")
            st.plotly_chart(fig)

        # Loan Status vs Income Boxplot
        if "loan_status" in df.columns and "person_income" in df.columns:
            st.subheader("Loan Status vs Income")
            st.markdown("This boxplot shows the distribution of applicant incomes across loan statuses (Approved vs. Rejected).")
            fig = px.box(df, x="loan_status", y="person_income", title="Income Distribution by Loan Status", labels={"loan_status": "Loan Status", "person_income": "Income"})
            st.plotly_chart(fig)

        # Loan Status vs Loan Amount Boxplot
        if "loan_status" in df.columns and "loan_amnt" in df.columns:
            st.subheader("Loan Status vs Loan Amount")
            st.markdown("This boxplot shows the distribution of loan amounts across loan statuses (Approved vs. Rejected).")
            fig = px.box(df, x="loan_status", y="loan_amnt", title="Loan Amount Distribution by Loan Status", labels={"loan_status": "Loan Status", "loan_amnt": "Loan Amount"})
            st.plotly_chart(fig)

        # Average Income by Loan Grade (Bar Chart)
        if "loan_grade" in df.columns and "person_income" in df.columns:
            st.subheader("Average Income by Loan Grade")
            st.markdown("This bar chart shows the average income of applicants grouped by their loan grade.")
            income_by_grade = df.groupby("loan_grade")["person_income"].mean().reset_index()
            fig = px.bar(income_by_grade, x="loan_grade", y="person_income", title="Average Income by Loan Grade", labels={"loan_grade": "Loan Grade", "person_income": "Average Income"})
            st.plotly_chart(fig)

        # Interactive Scatter Plot
        if len(numeric_columns) > 1:
            st.subheader("Interactive Scatter Plot")
            st.markdown("Select two numeric columns to explore their relationship using a scatter plot.")
            scatter_x = st.selectbox("Select X-axis", numeric_columns, index=0)
            scatter_y = st.selectbox("Select Y-axis", numeric_columns, index=1)
            if scatter_x and scatter_y:
                fig = px.scatter(df, x=scatter_x, y=scatter_y, title=f"Scatter Plot: {scatter_x} vs {scatter_y}", labels={scatter_x: scatter_x, scatter_y: scatter_y})
                st.plotly_chart(fig)

    if page == "Model Training":
        st.header("Train a Machine Learning Model")
        st.markdown("In this section, you can train a machine learning model to predict loan approval. Select the target column and click 'Train Model' to begin.")

        target_column = st.selectbox("Select Target Column", df.columns)
        if st.button("Train Model"):
            if target_column:
                X = df.drop(columns=[target_column])
                y = df[target_column]
                X = pd.get_dummies(X, drop_first=True)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Model Accuracy: {accuracy:.2f}")

                cm = confusion_matrix(y_test, y_pred)
                st.write("Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
                st.pyplot(fig)

else:
    if page == "Dashboard":
        st.header("Dashboard Overview")
        st.warning("Please upload a dataset to view insights.")

