# Interview Analyzer
Repo containing all the code related to the functionality of a software that transcribes and analyzes job interviews.
This software offers support for both English and Portuguese calls. 

* Backend.py does the call transcription and analysis using OpenAI's Whisper by calling some functions in utils.py
* Terms.JSON contains a dictionary of terms to be used in the analysis
* pdf_style.css determines the style of the output pdf file
* pdf_template.html provides an HTML template for the output analysis, but one can modify it as needed
* Chat.py uses OpenAI's conversational API to allow the user to ask follow-up questions about the analysis generated
