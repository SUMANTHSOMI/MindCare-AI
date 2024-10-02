from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import os
import cohere


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/')
def default_route():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/prognosis')
def prognosis():
    return render_template('solutions.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')


co = cohere.Client(api_key="oN4IpS2ynJ0w1Dsg3mXNwo20R3ulcogFiguuZkNw")

@app.route('/chat', methods=['POST'])
def chat():
    # Get the message from the frontend
    data = request.get_json()
    user_message = data.get('message')

    # Send the message to Cohere API and get the response
    response = co.chat(
        message=user_message,
        model="command"
    )
    
    # Return the chatbot response to the frontend
    return jsonify({'response': response.text})



@app.route('/blog')
def blog():
    return render_template('blog.html')


@app.route('/relaxation')
def privacypolicy():
    return render_template('privecy.html')


@app.route('/supporthub')
def supporthub():
    return render_template('case.html')


model = joblib.load('models/trained_model.pkl')



# Route to handle form submission
@app.route('/submit_prognosis_form', methods=['POST'])
def submit_prognosis_form():
    try:
        # Get form data from the 'prognosis' page in the frontend
        gender = request.form["gender"]
        # print(gender)
        age = request.form['age']
        # print(age)
        degree = request.form['degree']
        # print(degree)
        academicYear = request.form['academicYear']
        # print(academicYear)
        residentialStatus = request.form['residentialStatus']
        # print(residentialStatus)
        sportsEngagement = request.form['sportsEngagement']
        # print(sportsEngagement)
        averageSleep = request.form['averageSleep']
        # print(averageSleep)
        studySatisfaction = request.form['studySatisfaction']
        # print(studySatisfaction)
        academicWorkload = request.form['academicWorkload']
        # print(academicWorkload)
        academicPressure = request.form['academicPressure']
        # print(academicPressure)
        financialConcerns = request.form['financialConcerns']
        # print(financialConcerns)
        socialRelationships = request.form['socialRelationships']
        # print(socialRelationships)
        anxiety = request.form['anxiety']
        # print(anxiety)
        isolation = request.form['isolation']
        # print(isolation)
        futureInsecurity = request.form['futureInsecurity']
        # print(futureInsecurity)

        
        # print([gender,int(age),degree,academicYear,residentialStatus,sportsEngagement,averageSleep,int(studySatisfaction),int(academicWorkload),int(academicPressure),int(financialConcerns),int(socialRelationships),int(anxiety),int(isolation),int(futureInsecurity)])


    
        # Function to convert gender to 1 if Male and 0 if Female
        def convert_gender(gender):
            return 1 if gender.lower() == 'male' else 0
        # Function to convert degree_level
        def convert_degree_level(degree):
            return 1 if degree.lower()=='Undergraduate' else 0
        # Function to convert academic_year
        def convert_academic_year(academicYear):
            if academicYear == '2nd':
                return 1
            elif academicYear == '3rd':
                return 2
            elif academicYear == '1st':
                return 0
            elif academicYear == '4th':
                return 3
        # Function to convert residential_status
        def convert_residential_status(residentialStatus):
            return 1 if residentialStatus.lower()=='On-Campus' else 0

        # Function to convert sports_engagement
        def convert_sports_engagement(sportsEngagement):
            if sportsEngagement == '4-6 times':
                return 1
            elif sportsEngagement == '7+ times':
                return 2
            elif sportsEngagement == '1-3 times':
                return 0
            elif sportsEngagement == 'No sports':
                return 3
        # Function to convert average_sleep
        def convert_average_sleep(averageSleep):
            if averageSleep == '4-6':
                return 1
            elif averageSleep == '7+':
                return 2
            elif averageSleep == '2-4':
                return 0
        
        
        gender=convert_gender(gender)
        degree_level=convert_degree_level(degree)
        academic_year=convert_academic_year(academicYear)
        residential_status=convert_residential_status(residentialStatus)
        sports_engagement=convert_sports_engagement(sportsEngagement)
        average_sleep=convert_average_sleep(averageSleep)
        
        
        input_data=[gender,int(age),degree_level,academic_year,residential_status,sports_engagement,average_sleep,int(studySatisfaction),int(academicWorkload),int(academicPressure),int(financialConcerns),int(socialRelationships),int(anxiety),int(isolation),int(futureInsecurity)]
        print(input_data)


        input_data = np.array(input_data).reshape(1, -1)
        input_df = pd.DataFrame(input_data, columns=['gender', 'age', 'degree_level', 'academic_year', 'residential_status', 'sports_engagement', 'average_sleep','study_satisfaction', 'academic_workload ', 'academic_pressure','financial_concerns', 'social_relationships', 'anxiety','isolation', 'future_insecurity'])
        
        # Make a prediction using the loaded model
        prediction = model.predict(input_df)

        # print(prediction)

        # Remedies for each psychological breakdown level
        remedies = {
            1: [
                "Practice mindfulness and relaxation techniques.",
                "Engage in light physical activity.",
                "Maintain social connections and spend time with family or friends.",
                "Ensure proper sleep, nutrition, and hydration.",
                "Try journaling to reflect on your emotions.",
                "Sharing your feelings who can give you emotional support"
            ],
            2: [
                "Use cognitive behavioral therapy (CBT) techniques to challenge negative thoughts.",
                "Manage time effectively and break tasks into smaller parts.",
                "Engage in hobbies and leisure activities.",
                "Limit digital exposure and practice digital detox.",
                "Maintain a regular sleep schedule.",
                "Sharing your feelings who can give you emotional support"
            ],
            3: [
                "Consider talk therapy for professional guidance.",
                "Attend stress management workshops or use online resources.",
                "Limit caffeine and sugar intake to reduce anxiety.",
                "Practice mindfulness meditation regularly.",
                "Set clear work-life boundaries.",
                "Sharing your feelings who can give you emotional support"
            ],
            4: [
                "Seek regular therapy, including CBT or psychodynamic therapy.",
                "Join peer support groups to share experiences.",
                "Consult a psychiatrist for medication if necessary.",
                "Establish a structured daily routine to prevent overwhelm.",
                "Implement major lifestyle changes, such as improving sleep and diet.",
                "Sharing your feelings who can give you emotional support"
            ],
            5: [
                "Seek immediate psychiatric care from a professional.",
                "Use crisis helplines or consider hospitalization if needed.",
                "Engage in intensive therapy, such as dialectical behavior therapy (DBT).",
                "Follow a strict medication management plan under supervision.",
                "Involve close family or friends for support and focus on rest.",
                "Sharing your feelings who can give you emotional support"
            ]
        }

        print(prediction[0])

        # Select remedies based on prediction level
        suggested_remedies = remedies.get(prediction[0], ["No specific remedies available."])
        risk_percentage = "Had a risk of level " + str(prediction[0])

        return jsonify({
            'heart_risk_percentage': risk_percentage,
            'remedies': suggested_remedies
        })

    except Exception as e:
        print(f"Error processing form data: {str(e)}")
        return jsonify({'error': 'Error processing form data'}), 500  # Return an error response



if __name__ == '__main__':
    app.run(debug=True)
