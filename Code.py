import streamlit as st
import pandas as pd

st.title("ML Model Comparator")
st.subheader("Upload your dataset")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    st.subheader("Column Information")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Select Target Column")
    metadata = ['id', 'name', 'email', 'phone', 'address', 'index']
    good_columns = []
    for col in df.columns:
        if col.lower() in metadata:
            continue
        if df[col].nunique() == len(df):
            continue
        good_columns.append(col)

    suggested = good_columns[-1]
    st.info(f"Suggested target column: {suggested}")
    target_col = st.selectbox("Confirm or change target column", good_columns, index=good_columns.index(suggested))
    st.success(f"Target column selected: {target_col}")

    st.subheader("Select Task Type")
    task = st.radio("What kind of problem is this?", ["Classification", "Regression"])
    st.success(f"Task selected: {task}")

    st.subheader("Text Column Detection")
    text_columns = [col for col in df.columns if df[col].dtype == 'object' and col != target_col]

    is_text_data = False
    text_col = None
    ngram_range = (1, 2)

    if text_columns:
        st.info(f"Text columns found: {text_columns}")
        text_col = st.selectbox("Select text column", text_columns)
        is_text_data = st.checkbox("Use NLP pipeline (TF-IDF) for this column?")
        if is_text_data:
            ngram_choice = st.radio("Select N-gram Range", ["Unigram (1,1)", "Bigram (1,2)", "Trigram (1,3)"], index=1)
            ngram_map = {"Unigram (1,1)": (1,1), "Bigram (1,2)": (1,2), "Trigram (1,3)": (1,3)}
            ngram_range = ngram_map[ngram_choice]
            st.info(f"N-gram range: {ngram_range}")
    else:
        st.info("No text columns found - using tabular pipeline")

    st.subheader("Preprocessing Data")
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    le = LabelEncoder()

    if is_text_data and text_col:
        from sklearn.feature_extraction.text import TfidfVectorizer
        st.write("Running TF-IDF...")
        tfidf = TfidfVectorizer(ngram_range=ngram_range, max_features=5000)
        X = tfidf.fit_transform(df[text_col])
        y = df[target_col]
        st.success(f"TF-IDF done! Shape: {X.shape}")
    else:
        df = df.dropna()
        st.write("Rows after dropping missing values:", len(df))
        X = df.drop(columns=[target_col])
        y = df[target_col]
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col])
        if task == "Classification":
            if y.dtype == 'object':
                y = le.fit_transform(y)
        st.success("Preprocessing done!")
        st.dataframe(X.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write("Training size:", X_train.shape)
    st.write("Testing size:", X_test.shape)

    if st.button("Run All Models"):

        if task == "Classification":
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import accuracy_score, f1_score

            models = {
                "Logistic Regression" : LogisticRegression(max_iter=1000),
                "Decision Tree"       : DecisionTreeClassifier(),
                "Random Forest"       : RandomForestClassifier(),
                "SVM"                 : SVC(),
                "KNN"                 : KNeighborsClassifier()
            }

            results = []
            for name, model in models.items():
                with st.spinner(f"Training {name}..."):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    f1  = f1_score(y_test, y_pred, average='weighted')
                    results.append({
                        "Model"    : name,
                        "Accuracy" : round(acc * 100, 2),
                        "F1 Score" : round(f1, 4)
                    })

        elif task == "Regression":
            from sklearn.linear_model import LinearRegression
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.svm import SVR
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.metrics import mean_absolute_error, r2_score

            models = {
                "Linear Regression" : LinearRegression(),
                "Decision Tree"     : DecisionTreeRegressor(),
                "Random Forest"     : RandomForestRegressor(),
                "SVR"               : SVR(),
                "KNN"               : KNeighborsRegressor()
            }

            results = []
            for name, model in models.items():
                with st.spinner(f"Training {name}..."):
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2  = r2_score(y_test, y_pred)
                    results.append({
                        "Model"    : name,
                        "MAE"      : round(mae, 4),
                        "R2 Score" : round(r2, 4)
                    })

        st.subheader("Model Comparison Results")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        if task == "Classification":
            best = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
            st.success(f"Best Model: {best}")
            st.subheader("Visual Comparison")
            st.bar_chart(results_df.set_index("Model")["Accuracy"])
        else:
            best = results_df.loc[results_df["R2 Score"].idxmax(), "Model"]
            st.success(f"Best Model: {best}")
            st.subheader("Visual Comparison")
            st.bar_chart(results_df.set_index("Model")["R2 Score"])
