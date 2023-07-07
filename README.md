# Interview Analyzer
Repo containing all the code related to the functionality of software that transcribes and analyzes job interviews. The software takes in an audio file and transcribes it using the Whisper API. It then proceeds to analyze the call itself using OPenAI's GPT-4 model and outputs a PDF of the analysis along axes such as "confidence", "listening / talking ratio", and "Resume overview". This is useful for job/university interviewees such as me to understand what their communicative strengths and weaknesses are through the help of AI. Furthermore, after the analysis is generated, users can ask specific questions about how they can improve their performance by running the Chat.py file, which provides a ChatGPT-like interface of communication through the terminal.

Files include:

* Backend.py does the call transcription and analysis using OpenAI's Whisper by calling some functions in utils.py
* Terms.JSON contains a dictionary of terms to be used in the analysis
* pdf_style.css determines the style of the output pdf file
* pdf_template.html provides an HTML template for the output analysis, but one can modify it as needed
* Chat.py uses OpenAI's conversational API to allow the user to ask follow-up questions about the analysis generated

Note: This software stores the output of its predictions on a MongoDB database and offers support for both English and Portuguese calls.
