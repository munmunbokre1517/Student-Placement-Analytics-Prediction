## Streamlit Dashboard


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Student Placement Analytics",
    page_icon="🎓",
    layout="wide"
)


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🎓 Student Placement Analytics & Prediction System")

st.write(
    "This dashboard analyzes student academic, skill-based and "
    "professional factors associated with placement outcomes "
    "and provides placement predictions using Machine Learning."
)


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():

    data = pd.read_csv("Dataset/train (1).csv")

    return data


train = load_data()


# ---------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------

ml_data = train.copy()

ml_data["ExtracurricularActivities"] = ml_data[
    "ExtracurricularActivities"
].map({
    "Yes": 1,
    "No": 0
})

ml_data["PlacementTraining"] = ml_data[
    "PlacementTraining"
].map({
    "Yes": 1,
    "No": 0
})

ml_data["PlacementStatus"] = ml_data[
    "PlacementStatus"
].map({
    "Placed": 1,
    "NotPlaced": 0
})


# ---------------------------------------------------
# MACHINE LEARNING MODEL
# ---------------------------------------------------

X = ml_data.drop("PlacementStatus", axis=1)

y = ml_data["PlacementStatus"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


@st.cache_resource
def train_model():

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

model = train_model()

# Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.sidebar.metric("Model Accuracy", f"{accuracy * 100:.2f}%")



# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📌 Dashboard Menu")

section = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Placement Analysis",
        "Factor Analysis",
        "Placement Prediction"
    ]
)


# ===================================================
# OVERVIEW
# ===================================================

if section == "Overview":

    st.header("📊 Placement Overview")

    total_students = len(train)

    placed_students = (
        train["PlacementStatus"] == "Placed"
    ).sum()

    not_placed_students = (
        train["PlacementStatus"] == "NotPlaced"
    ).sum()

    placement_rate = (
        placed_students / total_students
    ) * 100


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Total Students",
        total_students
    )

    col2.metric(
        "Placed Students",
        placed_students
    )

    col3.metric(
        "Not Placed",
        not_placed_students
    )

    col4.metric(
        "Placement Rate",
        f"{placement_rate:.2f}%"
    )


    st.subheader("Placement Status Distribution")


    placement_counts = train[
        "PlacementStatus"
    ].value_counts()


    fig, ax = plt.subplots()

    ax.bar(
        placement_counts.index,
        placement_counts.values
    )

    ax.set_xlabel("Placement Status")

    ax.set_ylabel("Number of Students")

    ax.set_title("Overall Placement Distribution")

    st.pyplot(fig)


# ===================================================
# PLACEMENT ANALYSIS
# ===================================================

elif section == "Placement Analysis":

    st.header("📈 Placement Analysis")


    # ------------------------------------------------
    # CGPA
    # ------------------------------------------------

    st.subheader("CGPA Distribution by Placement Status")


    fig, ax = plt.subplots()

    sns.boxplot(
        data=train,
        x="PlacementStatus",
        y="CGPA",
        ax=ax
    )

    ax.set_title(
        "CGPA vs Placement Status"
    )

    st.pyplot(fig)


    # ------------------------------------------------
    # INTERNSHIPS
    # ------------------------------------------------

    st.subheader("💼 Internships vs Placement")


    internship_rate = (
        train.groupby("Internships")[
            "PlacementStatus"
        ]
        .value_counts(normalize=True)
        .mul(100)
        .rename("Percentage")
        .reset_index()
    )


    fig, ax = plt.subplots()

    sns.barplot(
        data=internship_rate,
        x="Internships",
        y="Percentage",
        hue="PlacementStatus",
        ax=ax
    )

    ax.set_title(
        "Internships and Placement Outcomes"
    )

    ax.set_ylabel("Percentage")

    st.pyplot(fig)


    # ------------------------------------------------
    # PROJECTS
    # ------------------------------------------------

    st.subheader("📁 Projects vs Placement")


    project_rate = (
        train.groupby("Projects")[
            "PlacementStatus"
        ]
        .value_counts(normalize=True)
        .mul(100)
        .rename("Percentage")
        .reset_index()
    )


    fig, ax = plt.subplots()

    sns.barplot(
        data=project_rate,
        x="Projects",
        y="Percentage",
        hue="PlacementStatus",
        ax=ax
    )

    ax.set_title(
        "Projects and Placement Outcomes"
    )

    ax.set_ylabel("Percentage")

    st.pyplot(fig)


# ===================================================
# FACTOR ANALYSIS
# ===================================================

elif section == "Factor Analysis":

    st.header("🔎 Factors Associated with Placement")


    # ------------------------------------------------
    # APTITUDE
    # ------------------------------------------------

    st.subheader("🧠 Aptitude Test Score")


    fig, ax = plt.subplots()

    sns.boxplot(
        data=train,
        x="PlacementStatus",
        y="AptitudeTestScore",
        ax=ax
    )

    ax.set_title(
        "Aptitude Score by Placement Status"
    )

    st.pyplot(fig)


    # ------------------------------------------------
    # SOFT SKILLS
    # ------------------------------------------------

    st.subheader("🗣️ Soft Skills Rating")


    fig, ax = plt.subplots()

    sns.boxplot(
        data=train,
        x="PlacementStatus",
        y="SoftSkillsRating",
        ax=ax
    )

    ax.set_title(
        "Soft Skills Rating by Placement Status"
    )

    st.pyplot(fig)


    # ------------------------------------------------
    # CERTIFICATIONS
    # ------------------------------------------------

    st.subheader("📜 Workshops / Certifications")


    certification_rate = (
        train.groupby(
            "Workshops/Certifications"
        )["PlacementStatus"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("Percentage")
        .reset_index()
    )


    fig, ax = plt.subplots()

    sns.barplot(
        data=certification_rate,
        x="Workshops/Certifications",
        y="Percentage",
        hue="PlacementStatus",
        ax=ax
    )

    ax.set_title(
        "Workshops / Certifications vs Placement"
    )

    ax.set_ylabel("Percentage")

    st.pyplot(fig)


    # ------------------------------------------------
    # CORRELATION
    # ------------------------------------------------

    st.subheader("📊 Correlation with Placement")


    analysis_df = pd.get_dummies(
        train,
        columns=[
            "ExtracurricularActivities",
            "PlacementTraining",
            "PlacementStatus"
        ],
        drop_first=True
    )


    corr = analysis_df.corr(
        numeric_only=True
    )


    placement_corr = (
        corr["PlacementStatus_Placed"]
        .sort_values(
            ascending=False
        )
    )


    st.dataframe(
        placement_corr
    )


# ===================================================
# PLACEMENT PREDICTION
# ===================================================

elif section == "Placement Prediction":

    st.header("🤖 Student Placement Prediction")


    st.write(
        "Enter the student's details below to predict "
        "the expected placement status."
    )


    col1, col2 = st.columns(2)


    with col1:

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )


        internships = st.number_input(
            "Number of Internships",
            min_value=0,
            max_value=10,
            value=1
        )


        projects = st.number_input(
            "Number of Projects",
            min_value=0,
            max_value=10,
            value=2
        )


        certifications = st.number_input(
            "Workshops / Certifications",
            min_value=0,
            max_value=10,
            value=1
        )


        aptitude = st.number_input(
            "Aptitude Test Score",
            min_value=0,
            max_value=100,
            value=75
        )


    with col2:

        soft_skills = st.number_input(
            "Soft Skills Rating",
            min_value=0.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )


        extracurricular = st.selectbox(
            "Extracurricular Activities",
            ["No", "Yes"]
        )


        placement_training = st.selectbox(
            "Placement Training",
            ["No", "Yes"]
        )


        ssc_marks = st.number_input(
            "SSC Marks",
            min_value=0,
            max_value=100,
            value=70
        )


        hsc_marks = st.number_input(
            "HSC Marks",
            min_value=0,
            max_value=100,
            value=75
        )


    # ------------------------------------------------
    # PREDICTION BUTTON
    # ------------------------------------------------

    if st.button(
        "🔮 Predict Placement",
        use_container_width=True
    ):

        input_data = pd.DataFrame({

            "CGPA": [cgpa],

            "Internships": [internships],

            "Projects": [projects],

            "Workshops/Certifications": [
                certifications
            ],

            "AptitudeTestScore": [
                aptitude
            ],

            "SoftSkillsRating": [
                soft_skills
            ],

            "ExtracurricularActivities": [
                1 if extracurricular == "Yes"
                else 0
            ],

            "PlacementTraining": [
                1 if placement_training == "Yes"
                else 0
            ],

            "SSC_Marks": [ssc_marks],

            "HSC_Marks": [hsc_marks]
        })


        prediction = model.predict(
            input_data
        )[0]


        probability = model.predict_proba(
            input_data
        )[0]


        if prediction == 1:

            st.success(
                "🎉 Predicted Status: PLACED"
            )

        else:

            st.error(
                "Predicted Status: NOT PLACED"
            )


        placement_probability = probability[1] * 100

        if placement_probability < 40:
            risk_level = "LOW"
        elif placement_probability < 70:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

            st.write(
                f"Placement Probability: {placement_probability:.2f}%"
)


        if risk_level == "HIGH":
            st.success("🟢 Placement Probability Level: HIGH")
        elif risk_level == "MEDIUM":
            st.warning("🟡 Placement Probability Level: MEDIUM")
        else:
            st.error("🔴 Placement Probability Level: LOW")


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.info(
    "Student Placement Analytics & Prediction System"
)