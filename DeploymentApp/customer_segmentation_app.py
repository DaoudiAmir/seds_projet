

# importing necessary libraries
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import os
#load the model
# Get the absolute path to the 'classifier.pkl' file
file_path = os.path.join(os.path.dirname(__file__), 'classifier.pkl')

# Load the classifier
classifier = pickle.load(open(file_path, 'rb'))


#page configuration
st.set_page_config(page_title = 'Customer Classification Web App', layout='centered')

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
        
