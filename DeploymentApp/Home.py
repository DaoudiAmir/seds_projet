
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import subprocess
import pickle
import os

data = pd.read_csv('https://drive.google.com/uc?export=download&id=1I6lZwW1kaAD8mXWd55RIxK-4t-VZRLOD')
data1 = pd.read_csv('https://drive.google.com/uc?export=download&id=1gWfSxVA0pvHxgPP_Rhz10SGPnVOjKTVV')

def redirect_to_prediction_page():
    # Exécuter le fichier Python pour la prédiction
    subprocess.run(["streamlit", "run", "customer_segmentation_app.py", "--server.port", "8080"])
def highlight_missing(val):
    if pd.isnull(val):
        return 'background-color: red'
    else:
        return ''
 #function explore_database
def explore_database():
    st.markdown('<p style="background-color:#29285D;font-family:verdana;color:white;font-size:100%;text-align:center;'
            'border-radius:10px 10px;letter-spacing:0.5px;padding: 10px">Database Exploration </p>', 
            unsafe_allow_html=True)
    
    
    # Allow users to upload the product sales dataset
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Display dataset description
        st.subheader("1. How big is the data?")
        st.text(f"Number of Rows: {data.shape[0]}, Number of Columns: {data.shape[1]}")

        # Display a sample of the data
        st.subheader("2. What does the data look like?")
        st.dataframe(data)

        # Display data types of columns
        st.subheader("3. What is the data type of each column?")
        st.dataframe(data.dtypes.reset_index().rename(columns={0: 'Data Type', 'index': 'Column Name'}))

        # Display non-null counts
        st.subheader("Non-null counts:")
        st.dataframe(data.count())
        
        # Display the number of missing values per column
        
        st.subheader("4. Are there any missing values?")
        missing_values = data.isnull().sum()
        total_missing = missing_values.sum()

        # Affiche le total des valeurs manquantes 
        st.markdown(f"Total missing values: {total_missing}")

        # Affiche les valeurs manquantes par colonne
        st.write("Missing values per column:")
        st.write(missing_values)

        


        # Display the mathematical summary of the data
        st.subheader("5. How does the data look mathematically?")
        st.dataframe(data.describe())

        # Display the number of duplicate values
        st.subheader("6. Are there duplicate values?")
        st.text(f"Number of duplicate rows: {data.duplicated().sum()}")
        html_code = """
     <div style="color:white;
           display:fill;
           background-color:#29285D;
           padding: 10px;
           font-family:Verdana;
           letter-spacing:0.5px">
     <h3 style="color:white;padding-left:20px"><b>Observations 👀</b></h3>
       <p style="color:white;font-size:110%;padding-left:50px">
           1. Data contains 2240 rows and 29 columns <br>
           2. It has 1 float, 3 object and 25 int columns <br>
           3. Datetime columns have int datatype, need to fix that <br>
           4. Many categorical columns are already in int format, like AcceptedCmp1, Response <br>
           5. There are 24 missing values in the Income column <br>
           6. There are no duplicate values within data
        </p>
     </div>
     """   

     # Affiche le code HTML dans Streamlit
        st.markdown(html_code, unsafe_allow_html=True)
def data_cleaning():
 st.markdown('<p style="background-color:#29285D;font-family:verdana;color:white;font-size:100%;text-align:center;'
            'border-radius:10px 10px;letter-spacing:0.5px;padding: 10px">DATA CLEANING & FEATURE CONSTRUCTION</p>', 
            unsafe_allow_html=True)
    # Display the first set of steps in a styled box
 st.markdown(
    """
    <div style="background-color:#f4f4f4; padding: 20px; border-radius: 10px;">
        <h2 style="color:#29285D;">Steps To Follow:</h2>
        <ol style="color:#333; font-size:110%; padding-left: 20px;">
            <li>Fix the column names.</li>
            <li>Convert the datetime column into the correct format.</li>
            <li>Handle missing values.</li>
            <li>Examine unique values within categorical columns.</li>
            <li>Check the timeline of the data since we have datetime columns given.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True
 )

 # Display the code snippet for renaming columns and converting datetime columns
 st.code(
    """
    # Rename columns
    data.rename(columns={'MntGoldProds': 'MntGoldProducts'}, inplace=True)

    # Convert columns to DateTime format
    data['Year_Birth'] = pd.to_datetime(data['Year_Birth'], format='%Y')
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')
    """,
    language="python"
)

 # Display the alert about skewness
 st.markdown(
    """
    <div class="alert alert-info" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
        📌 &nbsp; If the skewness is between -0.5 and 0.5, the data are fairly symmetrical.
        If the skewness is between -1 and – 0.5 or between 0.5 and 1, the data are moderately skewed.
        If the skewness is less than -1 or greater than 1, the data are highly skewed.
    </div>
    """,
    unsafe_allow_html=True
)

 # Display the code snippet for replacing null values in the 'Income' column
 st.code(
    """
    # Replace null values with median (because the data is skewed)
    data['Income'].fillna(data['Income'].median(), inplace=True)
    """,
    language="python"
)

 # Display the code snippet for checking unique values in 'Education' and 'Marital_Status' columns
 st.code(
    """
    # Check unique values in 'Education' column
    data['Education'].value_counts()

    # Check unique values in 'Marital_Status' column
    data['Marital_Status'].value_counts()
    """,
    language="python"
 )
  # Display the second set of steps in a styled box
 st.markdown(
    """
    <div style="background-color:#f4f4f4; padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h2 style="color:#29285D;">Steps To Follow:</h2>
        <ol style="color:#333; font-size:110%; padding-left: 20px;">
            <li>Create 'Age', 'Years_Customer', and 'Days_Customer' columns by subtracting last date of 'Dt_Customer' from 'Year_Birth' and 'Dt_Customer'.</li>
            <li>Create 'TotalMntSpent', 'TotalNumPurchases', and 'TotalAccCmp' by adding the relative columns.</li>
            <li>Create 'Year_Joined', 'Month_Joined', and 'Day_Joined' columns through 'Dt_Customer'.</li>
            <li>Create 'Age_Group' column by dividing 'Age' column into different groups.</li>
            <li>Create 'Children' column by adding 'Kidhome' and 'Teenhome' columns.</li>
            <li>Create 'Partner' and 'Education_Level' for simplifying 'Marital_Status' and 'Education' columns.</li>
            <li>Drop the redundant columns.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True
)

 # Display the code snippet for creating additional columns
 st.code(
    """
    # Creating Age and Years_Customer (Amount of years a person has been a customer) columns
    data['Age'] = (data['Dt_Customer'].dt.year.max()) - (data['Year_Birth'].dt.year)
    data['Years_Customer'] = (data['Dt_Customer'].dt.year.max()) - (data['Dt_Customer'].dt.year)
    data['Days_Customer'] = (data['Dt_Customer'].max()) - (data['Dt_Customer'])

    # Total amount spent on products
    data['TotalMntSpent'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProducts']

    # Total number of purchases made
    data['TotalNumPurchases'] = data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases'] + data['NumDealsPurchases']

    # Total number of accepted campaigns
    data['Total_Acc_Cmp'] = data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] + data['AcceptedCmp4'] + data['AcceptedCmp5'] + data['Response']

    # Adding columns about the day, month, and year customer joined
    data['Year_Joined'] = data['Dt_Customer'].dt.year
    data['Month_Joined'] = data['Dt_Customer'].dt.strftime("%B")
    data['Day_Joined'] = data['Dt_Customer'].dt.day_name()

    # Dividing age into groups
    data['Age_Group'] = pd.cut(x=data['Age'], bins=[17, 24, 44, 64, 150], labels=['Young adult', 'Adult', 'Middle Aged', 'Senior Citizen'])

    # Total children living in the household
    data["Children"] = data["Kidhome"] + data["Teenhome"]

    # Deriving living situation by marital status
    data["Partner"] = data["Marital_Status"].replace({"Married": "Yes", "Together": "Yes", "Absurd": "No", "Widow": "No", "YOLO": "No", "Divorced": "No", "Single": "No", "Alone": "No"})

    # Segmenting education levels into three groups
    data["Education_Level"] = data["Education"].replace({"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"})

    # Dropping useless columns
    data.drop(['ID', 'Z_CostContact', 'Z_Revenue', 'Year_Birth', 'Dt_Customer'], axis=1, inplace=True)

    # Converting Days_Joined to int format
    data['Days_Customer'] = data['Days_Customer'].dt.days.astype('int16')
    print(data.shape)
    data.sample(5)
    """,
    language="python"
)

 # Display the alert about using a separate notebook for EDA
 st.markdown(
    """
    <div class="alert alert-info" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
        📌 &nbsp; The above dataset is suitable for performing EDA. However, for clustering, I'll use a subset of columns available for new customers for prediction.
    </div>
    """,
    unsafe_allow_html=True
)

 # Display the code snippet for capping outliers
 st.code(
    """
    # Capping outliers using the capping technique
    num_col = data1.select_dtypes(include=np.number).columns
    for col in num_col:
        q1 = data1[col].quantile(0.25)
        q3 = data1[col].quantile(0.75)
        iqr = q3 - q1
        ll = q1 - (1.5 * iqr)
        ul = q3 + (1.5 * iqr)
        for ind in data1
                if data1.loc[ind, col] > ul:
            data1.loc[ind, col] = ul
        elif data1.loc[ind, col] < ll:
            data1.loc[ind, col] = ll
        else:
            pass
    print("Outliers have been taken care of")
    """,
    language="python"
)

 # Display the final steps with a styled box
 
  
 # Check the Datetime Column:
 st.subheader("Check the Datetime Column:")
 st.write(data1['Dt_Customer'] if 'Dt_Customer' in data1.columns else "Column 'Dt_Customer'  found.")

 # Check the Skewness of 'Income' Column:
 st.subheader("Check the Skewness of 'Income' Column:")
 st.write(data1['Income'].skew() if 'Income' in data1.columns else "Column 'Income'  found.")

 # Check Unique Values in 'Education' Column:
 st.subheader("Check Unique Values in 'Education' Column:")
 st.write(data1['Education'].value_counts() if 'Education' in data1.columns else "Column 'Education'  found.")

 # Check Unique Values in 'Marital_Status' Column:
 st.subheader("Check Unique Values in 'Marital_Status' Column:")
 st.write(data1['Marital_Status'].value_counts() if 'Marital_Status' in data1.columns else "Column 'Marital_Status'  found.")

 # Find First and Last Date in Dataset:
 st.subheader("Find First and Last Date in Dataset:")
 first_year = data1["Dt_Customer"].dt.year.min() if 'Dt_Customer' in data1.columns else "Column 'Dt_Customer'  found."
 last_year = data1["Dt_Customer"].dt.year.max() if 'Dt_Customer' in data1.columns else "Column 'Dt_Customer'  found."
 st.write(f"First Year: {first_year}, Last Year: {last_year}")

 # Check DataFrame Shape:
 st.subheader("Check DataFrame Shape:")
 st.write(data1.shape)

 #  Sample 5 Rows from DataFrame:
 st.subheader("Sample 5 Rows from DataFrame:")
 st.write(data1.sample(5) if not data1.empty else "DataFrame is empty.")
def feature_tranformation():
    # Feature Transformation Title
    st.markdown('<p style="background-color:#29285D;font-family:verdana;color:white;font-size:100%;text-align:center;'
                'border-radius:10px 10px;letter-spacing:0.5px;padding: 10px">Feature Transformation</p>',
                unsafe_allow_html=True)

    # Data Selection
    global subset
    subset = data1[['Income', 'Kidhome', 'Teenhome', 'Age', 'Partner', 'Education_Level']]
    st.subheader("Data to Use for Clustering:")
    st.write(subset.head())

    # Make the Pipelines
    st.subheader("Make the Pipelines:")
    # Numeric Columns
    num_cols = ['Income', 'Age']
    numeric_pipeline = make_pipeline(StandardScaler())
    st.code("Numeric Columns Pipeline: \n" + str(numeric_pipeline))

    # Ordinal Columns
    ord_cols = ['Education_Level']
    ordinal_pipeline = make_pipeline(OrdinalEncoder(categories=[['Undergraduate', 'Graduate', 'Postgraduate']]))
    st.code("Ordinal Columns Pipeline: \n" + str(ordinal_pipeline))

    # Nominal Columns
    nom_cols = ['Partner']
    nominal_pipeline = make_pipeline(OneHotEncoder())
    st.code("Nominal Columns Pipeline: \n" + str(nominal_pipeline))

    # Column Transformer
    transformer = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, num_cols),
        ('ordinal', ordinal_pipeline, ord_cols),
        ('nominal', nominal_pipeline, nom_cols)
    ])
    st.subheader("Column Transformer:")
    st.code(str(transformer))

    # Fit and Transform the Data
    transformed_data = transformer.fit_transform(subset)
    st.success("Data has been Transformed.")
    st.subheader("Transformed Data:")
    st.write(transformed_data)
    return transformed_data, subset

    # Feature Transformation
def clustering(transformed_data, subset):

 

    palette = ["#FF5733", "#33FF57", "#3357FF"]
    plt.rcParams['tk.window_focus'] = False
    matplotlib.use('Agg')

    # Affichage du titre
    st.markdown('<p style="background-color:#29285D;font-family:verdana;color:white;font-size:100%;text-align:center;'
                'border-radius:10px 10px;letter-spacing:0.5px;padding: 10px">K-MEANS CLUSTERING & CLUSTER ANALYSIS </p>', 
                unsafe_allow_html=True)

    # Utilisation de KElbowVisualizer pour trouver le nombre optimal de clusters
    fig, ax = plt.subplots()
    elbow_graph = KElbowVisualizer(KMeans(random_state=43), k=10, ax=ax)
    elbow_graph.fit(transformed_data)
    st.pyplot(fig)

    # Détermination du nombre optimal de clusters (k=4)
    # Utilisation de K-Means pour former les clusters
    kmeans = KMeans(n_clusters=4, random_state=42)
    subset['Clusters'] = kmeans.fit_predict(transformed_data)

    # Affichage du nombre de clients dans chaque cluster 
    fig, ax = plt.subplots()
    sns.countplot(x='Clusters', data=subset, palette=['red', 'blue', 'green'], ax=ax)
    plt.title('Customer Distribution Within Clusters')
    st.pyplot(fig)

    # Liste des catégories à analyser
    count_cols = ['Kidhome', 'Teenhome', 'Partner', 'Education_Level']

    # Création d'une sous-trame avec les colonnes sélectionnées
    _, ax1 = plt.subplots(2, 2, figsize=(25, 22))

    # Affichage des countplots pour chaque variable dans chaque cluster
    for i, col in enumerate(count_cols):
        sns.countplot(x='Clusters', data=subset, ax=ax1[i//2, i%2], hue=col, palette=palette)

    st.pyplot(fig)

    # Graphiques des facteurs contributifs au revenu
    catcols = ['Kidhome', 'Teenhome', 'Partner', 'Education_Level']

    _, ax1 = plt.subplots(2, 2, figsize=(25, 22))

    for i, col in enumerate(catcols):
        sns.barplot(x='Clusters', y='Income', data=subset, ax=ax1[i//2, i%2], hue=col, palette=palette)

    st.pyplot(fig)
    plt.show()
def redirect_to_another_page():
    
    page = "INITIAL ANALYSIS"
    st.markdown(f'<meta http-equiv="refresh" content="0;URL=/{page}">', unsafe_allow_html=True)



def clusteringgg(transformed_data, subset):

    palette = ["#FF5733", "#33FF57", "#3357FF"]
    plt.rcParams['tk.window_focus'] = False
    matplotlib.use('Agg')
    # Affichage du titre
    st.markdown('<p style="background-color:#29285D;font-family:verdana;color:white;font-size:100%;text-align:center;'
                'border-radius:10px 10px;letter-spacing:0.5px;padding: 10px">K-MEANS CLUSTERING & CLUSTER ANALYSIS </p>', 
                unsafe_allow_html=True)

    # Utilisation de KElbowVisualizer pour trouver le nombre optimal de clusters
    fig, ax = plt.subplots()
    elbow_graph = KElbowVisualizer(KMeans(random_state=43), k=10, ax=ax)
    elbow_graph.fit(transformed_data)
    st.pyplot(fig)

    # Affichage des observations sur le nombre optimal de clusters
    st.markdown('<div style="color:white;display:fill;background-color:#29285D;padding: 10px;font-family:Verdana;'
                'letter-spacing:0.5px"><h3 style="color:white;padding-left:20px"><b>Observations 👀</b></h3>'
                '<p style="color:white;font-size:110%;padding-left:50px">'
                'Le nombre optimal de clusters est sélectionné en fonction de la méthode du coude (Elbow Method).'
                'Il est déterminé en recherchant le point du coude sur le graphique où l\'inertie commence à diminuer de manière linéaire. '
                'Dans ce cas, le nombre optimal de clusters semble être 4.'
                '</p></div>', unsafe_allow_html=True)

    # Détermination du nombre optimal de clusters (k=4)
    # Utilisation de K-Means pour former les clusters
    kmeans = KMeans(n_clusters=4, random_state=42)
    subset['Clusters'] = kmeans.fit_predict(transformed_data)

    # Affichage du nombre de clients dans chaque cluster 
    fig, ax = plt.subplots()
    sns.countplot(x='Clusters', data=subset, palette=['red', 'blue', 'green'], ax=ax)
    plt.title('Customer Distribution Within Clusters')
    st.pyplot(fig)

    # Affichage des observations sur les clusters
    st.markdown('<div style="color:white;display:fill;background-color:#29285D;padding: 10px;font-family:Verdana;'
                'letter-spacing:0.5px"><h3 style="color:white;padding-left:20px"><b>Observations 👀</b></h3>'
                '<p style="color:white;font-size:110%;padding-left:50px">'
                '1. Cluster 2 has highest number of customers <br>'
                '2. Cluster 3 has least number of customers <br>'
                '</p></div>', unsafe_allow_html=True)

    # Liste des catégories à analyser
    count_cols = ['Kidhome', 'Teenhome', 'Partner', 'Education_Level']

    # Création d'une sous-trame avec les colonnes sélectionnées
    _, ax1 = plt.subplots(2, 2, figsize=(25, 22))

    # Affichage des countplots pour chaque variable dans chaque cluster
    for i, col in enumerate(count_cols):
        sns.countplot(x='Clusters', data=subset, ax=ax1[i//2, i%2], hue=col, palette=palette)
    
    st.pyplot(fig)

    # Affichage des observations sur les countplots
    st.markdown('<div style="color:white;display:fill;background-color:#29285D;padding: 10px;font-family:Verdana;'
                'letter-spacing:0.5px"><h3 style="color:white;padding-left:20px"><b>Observations 👀</b></h3>'
                '<p style="color:white;font-size:110%;padding-left:50px">'
                '<strong>Kidhome:</strong>'
                '<ul><li>Cluster 0 mostly has customers with 1 kid in household</li>'
                '<li>Cluster 1 has customers with no kids in household</li>'
                '<li>Cluster 2 also has a large number of customers with no kids in household</li>'
                '<li>Cluster 3 has customers with 0 and 1 kids in household</li></ul>'
                '<strong>Teenhome:</strong>'
                '<ul><li>Cluster 0 consists of customers with no teen in household & few of them have 1 Teen in household</li>'
                '<li>Same goes for cluster 1 & 3</li>'
                '<li>Cluster 2 has customers with 1 Teen in household</li></ul>'
                '<strong>Partner:</strong>'
                '<ul><li>All the customers in cluster 0 have a partner</li>'
                '<li>All the customers in cluster 3 have no partner</li>'
                '<li>Cluster 1 & 2 have customers with and without a partner, but most of them have a partner</li></ul>'
                '<strong>Education_Level:</strong>'
                '<ul><li>All clusters have customers with graduate, postgraduate, and undergraduate  background</li>'
                '<li>All clusters have fewer customers with an undergraduate background</li>'
                '<li>Cluster 2 has the highest number of postgraduates and graduates</li></ul>'
                '</p></div>', unsafe_allow_html=True)

    # Distribution des clients ayant des enfants dans différents clusters
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=subset["Clusters"], y=subset["Kidhome"], palette=palette, ax=ax)
    plt.title("Enfants dans le ménage vs Clusters", size=15)
    st.pyplot(fig)

    # Distribution des clients ayant des adolescents dans différents clusters
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=subset["Clusters"], y=subset["Teenhome"], palette=palette, ax=ax)
    plt.title("Adolescents dans le ménage vs Clusters", size=15)
    st.pyplot(fig)

    # Affichage des observations sur les distributions
    st.markdown('<div style="color:white;display:fill;background-color:#29285D;padding: 10px;font-family:Verdana;'
                'letter-spacing:0.5px"><h3 style="color:white;padding-left:20px"><b>Observations 👀</b></h3>'
                '<p style="color:white;font-size:110%;padding-left:50px">'
                '<strong>Enfants dans le ménage:</strong>'
                '<ul><li>Cluster 0 et 3 ont le nombre maximal de clients avec des enfants dans le ménage</li>'
                '<li>Cluster 1 et 2 ont le moins de clients avec des enfants dans le ménage</li></ul>'
                '<strong>Adolescents dans le ménage:</strong>'
                '<ul><li>Cluster 2 a le nombre maximal de clients ayant des adolescents dans le ménage</li>'
                '<li>Les autres clusters ont également des clients avec des adolescents dans le ménage, mais ils sont moins nombreux par rapport au cluster 2</li></ul>'
                '</p></div>', unsafe_allow_html=True)

    # Revenu des clients dans différents clusters
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=subset["Clusters"], y=subset["Income"], palette=palette, ax=ax)
    plt.title("Revenu vs Clusters", size=15)
    st.pyplot(fig)

    # Affichage des observations sur les revenus
    st.markdown('<div style="color:white;display:fill;background-color:#29285D;padding: 10px;font-family:Verdana;'
                'letter-spacing:0.5px"><h3 style="color:white;padding-left:20px"><b>Observations 👀</b></h3>'
                '<p style="color:white;font-size:110%;padding-left:50px">'
                '<strong>Revenu:</strong>'
                '<ul><li>Cluster 1 a un revenu élevé suivi de près par le cluster 2. Cela est un peu étrange car le cluster 2 a le plus grand nombre de clients et le plus grand nombre de diplômés et de diplômés en comparaison avec le cluster 1</li>'
                '<li>Les clusters 0 et 3 ont le moins de revenus</li></ul>'
                '</p></div>', unsafe_allow_html=True)

    # Graphiques des facteurs contributifs au revenu
    catcols = ['Kidhome', 'Teenhome', 'Partner', 'Education_Level']

    fig, ax1 = plt.subplots(2, 2, figsize=(25, 22))

    for i, col in enumerate(catcols):
        sns.barplot(x='Clusters', y='Income', data=subset, ax=ax1[i//2, i%2], hue=col, palette=palette)
    
    st.pyplot(fig)

st.set_page_config(
    page_title="Customer Classification Web App",
    page_icon="🧊",

   
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# Sidebar structure
st.sidebar.title("Customer Classification Web App")
st.sidebar.image("1.png", width=150)  # Ajoutez le chemin de votre logo
st.sidebar.markdown("---")  # Ligne de séparation

# Sélection de la page de navigation
selected_page = st.sidebar.radio('Navigation', ['Home', 'INITIAL ANALYSIS', 'DATA CLEANING & FEATURE CONSTRUCTION',
                                               'FEATURE TRANSFORMATION', 'K-MEANS CLUSTERING & ANALYSIS', 'Introduction To MLOps'])

# Définition du style pour le contenu principal
if selected_page != 'Home':
    st.markdown(
        f"<h1 style='color:#29285D;text-align:center;'>{selected_page}</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<hr style='border: 2px solid #29285D;'>",
        unsafe_allow_html=True
    )

# Home Page
if selected_page == 'Home':
    #load the model
    # Get the absolute path to the 'classifier.pkl' file
    file_path = os.path.join(os.path.dirname(__file__), 'classifier.pkl')

        # Load the classifier
    classifier = pickle.load(open(file_path, 'rb'))


    #page configuration

    st.title('Customer Classification Web App')


    # customer segmentation function
    def segment_customers(input_data):
    
        prediction=classifier.predict(pd.DataFrame(input_data, columns=['Income','Kidhome','Teenhome','Age','Partner','Education_Level']))
        print(prediction)
        pred_1 = 0
        if prediction == 0:
            pred_1 = 'cluster 0'

        elif prediction == 1:
            pred_1 = 'cluster 1'

        elif prediction == 2:
            pred_1 = 'cluster 2'

        elif prediction == 3:
            pred_1 = 'cluster 3'

        return pred_1

    def main():
    
        Income = st.text_input("Type In The Household Income")
        Kidhome = st.radio ( "Select Number Of Kids In Household", ('0', '1','2') )
        Teenhome = st.radio ( "Select Number Of Teens In Household", ('0', '1','2') )
        Age = st.slider ( "Select Age", 18, 85 )
        Partner = st.radio ( "Livig With Partner?", ('Yes', 'No') )
        Education_Level = st.radio ( "Select Education", ("Undergraduate", "Graduate", "Postgraduate") )
    
        result = ""

        # when 'Predict' is clicked, make the prediction and store it
        if st.button("Classify Customer"):
            result=segment_customers([[Income,Kidhome,Teenhome,Age,Partner,Education_Level]])
    
        st.success(result)
    


    
    html_str1 = """ <p style="background-color:#682F2F;font-family:newtimeroman;color:#FFF9ED;font-size:150%;text-align:center;border-radius:10px 10px;">Customer Segmentation</p>

<img src="https://github.com/KarnikaKapoor/Files/blob/main/Colorful%20Handwritten%20About%20Me%20Blank%20Education%20Presentation.gif?raw=true"  alt="Centered Image" style=" max-width: 100%;
      max-height: 100%;">
"""    
    html_str3 = """<p style="font-weight: bold;">Study prepared and conducted by Daoudi Amir Salah Eddine and Tbahriti Mohammed.</p>"""

        

    html_str4 = """<p>In this project, we performed an unsupervised clustering of data on the customer's records from a groceries firm's database. Customer segmentation is the practice of separating customers into groups that reflect similarities among customers in each cluster.</p>
<p>We will divide customers into segments to optimize the significance of each customer to the business. This allows us to modify products according to the distinct needs and behaviors of the customers, ultimately helping the business cater to the concerns of different types of customers.</p>"""


    html_str2 =  """ <h3>The Study Outcome</h3>

    <section>
        <h2>Cluster 0: </h2>
        <ul>
            <li>Average Income of $34,865 yearly.</li>
            <li>Average Spending is $500.</li>
            <li>The majority of them have not accepted any promotions yet (822).</li>
            <li>Most of them have completed purchases using a discount half of the time (1/2).</li>
            <li>Have either 1 or 2 children.</li>
            <li>Their age ranges between 25 and 70.</li>
            <li>Are at the graduate, postgraduate, or undergraduate level.</li>
            <li>Most are married, only a few are unmarried.</li>
            <li>Most of them are parents, very few are not parents.</li>
        </ul>
    </section>

    <section>
        <h2>Cluster 1: </h2>
        <ul>
            <li>Average Income of $65,463 yearly.</li>
            <li>Average Spending is between $550 and $2000.</li>
            <li>The majority of them have not accepted any promotions yet (428).</li>
            <li>Most of them have completed purchases using a discount a quarter of the time (1/4).</li>
            <li>Most of them have one child.</li>
            <li>Their age ranges between 35 and 70.</li>
            <li>Are at the graduate or postgraduate level.</li>
            <li>Most of them are married.</li>
            <li>Are not parents.</li>
        </ul>
    </section>

    <section>
        <h2>Cluster 2: </h2>
        <ul>
            <li>Average Income of $78,413 yearly.</li>
            <li>Average Spending is between $750 and $2250.</li>
            <li>The majority of them have not accepted any promotions yet (210).</li>
            <li>Most of them have completed purchases using a discount half of the time (1/2).</li>
            <li>Don't have any children.</li>
            <li>Their age ranges between 40 and 70.</li>
            <li>Are at the graduate, and very few are at the postgraduate level.</li>
            <li>Most of them are married.</li>
            <li>Are parents.</li>
        </ul>
    </section>


    <section>
        <h2>Cluster 3:</h2>
        <ul>
            <li>Average Income of $45,902 yearly.</li>
            <li>Average Spending is between $0 and $1000.</li>
            <li>The majority of them have not accepted any promotions yet (297).</li>
            <li>Most of them have completed purchases using a discount 3 to 5 times.</li>
            <li>Have one, two, or three children.</li>
            <li>Their age ranges between 40 and 70.</li>
            <li>All of them are at the graduate and postgraduate level.</li>
            <li>Most of them are married.</li>
            <li>Are parents.</li>
        </ul>
    </section>
    <img src="https://i.imgur.com/ga3acNx.png"  alt="Centered Image" style=" max-width: 100%;
      max-height: 100%;">
    """
    html_str5 = """<h1> Making new Predictions using the Model : </h1>"""

    st.markdown(html_str1, unsafe_allow_html= True)    
    st.markdown(html_str3, unsafe_allow_html= True)   
    st.markdown(html_str4, unsafe_allow_html= True)   
    st.markdown(html_str2, unsafe_allow_html= True)  
    st.markdown(html_str5, unsafe_allow_html= True)    

    if __name__ == '__main__':
        main ()
# Database Exploration
elif selected_page == 'INITIAL ANALYSIS':
    explore_database()
      
elif selected_page == 'DATA CLEANING & FEATURE CONSTRUCTION':
    data_cleaning()
    
    
elif selected_page == 'FEATURE TRANSFORMATION':
    feature_tranformation()
    
elif selected_page == 'K-MEANS CLUSTERING & ANALYSIS':
   transformed_data, subset = feature_tranformation()
   clusteringgg(transformed_data, subset)
   
elif selected_page == 'Introduction To MLOps':
    components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vSWeRutCayLmqVob30DjZ2FVEvANe8VnmwnuG0EYnqAfn6oeQA71NMGHFh3G4VRZadsE_ybSdh9lnwi/embed?start=false&loop=false&delayms=3000", height=460)
    