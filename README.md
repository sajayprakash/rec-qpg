# REC QPG (Question Paper Generator)
REC QPG is a web application designed to generate question papers based on uploaded PDF notes. It utilizes LangChain models and Groq API for natural language processing tasks.

> **Note:** This is just a prototype, I have added only a sample paper pattern. Modify the pattern before using for real exams.

## Features
- Upload PDF files containing study notes.
- Choose from different examination types (CAT-1, CAT-2, CAT-3, End Semester) for question paper generation.
- Customize the model used for question generation.
- Prototype model for generating question papers based on predefined patterns.
- Responsive UI with Streamlit.

## How to run
1. Clone the repo:
   
   ```bash
   git clone https://github.com/sajayprakash/rec-qpg
   ```
3. Install dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a .env file in the root directory.
   - Add your GROQ API key to the .env file:
     
     ```
     GROQ_API_KEY=your_groq_api_key
     ```
# Demo Video
https://github.com/sajayprakash/rec-qpg/assets/78424701/b8525639-54bd-46be-9e43-24a797548d55
