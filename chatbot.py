import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath(r"C:\Users\DELL\Desktop\Implementation of ChatBot using NLP\intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("Intents of Chatbot using NLP")

    # Addidng image
    image_url = r"C:\Users\DELL\Desktop\Implementation of ChatBot using NLP\3455231.jpg"  
    st.image(image_url, caption="Welcome to the Chatbot")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Date and time
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Saving the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":

        st.header("Conversation History")

        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
 
        st.write("""
        EcoChatBot is designed to help users with information and suggestions related to environmental sustainability. 
        It provides guidance on various eco-friendly practices, energy conservation, and other green initiatives. 
        The chatbot uses Natural Language Processing (NLP) to analyze user input and provide tailored responses, such as 
        suggesting energy-saving tips, educating users about recycling, and offering insights into sustainable practices.
        """)

        # Example intents related to EcoChatBot
        st.subheader("EcoChatBot Intents:")

        st.write("""
        Some examples of intents that EcoChatBot can respond to include:
        -Greeting: 
          - User: "Hello"
          - Chatbot: "Hi! How can I help you today?"
          
        - Energy Conservation Tips: 
          - User: "How can I save energy at home?"
          - Chatbot: "You can save energy by switching off lights when not in use, using energy-efficient appliances, and unplugging electronics when not in use."
          
        - Recycling Information: 
          - User: "What can I recycle?"
          - Chatbot: "You can recycle paper, cardboard, glass bottles, and certain plastics. Make sure to check local recycling guidelines."
          
        - Eco-Friendly Products: 
          - User: "What are some eco-friendly products I can use?"
          - Chatbot: "You can try using biodegradable cleaning products, reusable bags, and water-saving faucets."
          
        - Sustainable Living: 
          - User: "How can I live more sustainably?"
          - Chatbot: "Living sustainably includes reducing waste, choosing renewable energy sources, and supporting eco-friendly businesses."
        """)

        st.subheader("Conclusion:")

        st.write("""
        In conclusion, EcoChatBot provides users with practical advice and knowledge on how to live an eco-friendly life. 
        By integrating NLP techniques and machine learning, EcoChatBot is constantly improving in providing accurate and helpful information to users on environmental topics.
        This chatbot can be extended by adding more eco-related intents, data, and even integrating with real-time data for energy consumption, weather forecasts, and sustainability tips.
        """)

if __name__ == '__main__':
    main()
