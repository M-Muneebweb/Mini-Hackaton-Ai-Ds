import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# Sidebar navigation
st.sidebar.title('ML Model Explorer')
mode = st.sidebar.selectbox("Select Learning Type", ["Supervised", "Unsupervised"])

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Data")
    st.dataframe(data.head())
    
    # Preprocessing
    X = data.select_dtypes(include=[np.number])  # Only numeric columns for models
    X = X.fillna(X.mean())  # Handle missing values

    # Encode categorical columns
    for col in data.select_dtypes(include=[object]).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if mode == "Supervised":
        st.sidebar.write("Select Target Column:")
        target_col = st.sidebar.selectbox("", data.columns)
        model_choice = st.sidebar.selectbox("Model", ["Decision Tree", "Random Forest", "SVM"])
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
        if st.sidebar.button("Train Model"):
            y = data[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
            if model_choice == "Decision Tree":
                max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
                model = DecisionTreeClassifier(max_depth=max_depth)
            elif model_choice == "Random Forest":
                n_estimators = st.sidebar.slider("Estimators", 10, 200, 50)
                model = RandomForestClassifier(n_estimators=n_estimators)
            else:
                c_value = st.sidebar.slider("C (SVM)", 0.01, 10.0, 1.0)
                model = SVC(C=c_value)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {acc:.2f}")
            st.write("Classification Report")
            st.text(classification_report(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            st.pyplot(plt)

    else:  # Unsupervised learning
        model_choice = st.sidebar.selectbox("Model", ["KMeans", "Agglomerative", "DBSCAN"])
        if model_choice == "KMeans":
            n_clusters = st.sidebar.slider("Clusters", 2, 10, 3)
            model = KMeans(n_clusters=n_clusters)
        elif model_choice == "Agglomerative":
            n_clusters = st.sidebar.slider("Clusters", 2, 10, 3)
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            eps = st.sidebar.slider("DBSCAN: eps", 0.1, 5.0, 0.5)
            model = DBSCAN(eps=eps)
        if st.sidebar.button("Run Clustering"):
            labels = model.fit_predict(X_scaled)
            st.write("Cluster Labels:", np.unique(labels))
            if hasattr(model, 'cluster_centers_'):
                st.write("Centers:", model.cluster_centers_)
            st.write(f"Silhouette Score: {silhouette_score(X_scaled, labels):.2f}")
            plt.figure(figsize=(6,4))
            sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette="deep")
            st.pyplot(plt)

else:
    st.info("Upload a CSV file to begin.")


