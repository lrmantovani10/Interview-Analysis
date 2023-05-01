import json, os, pymongo
from dotenv import load_dotenv
from utils import *

# Load environment variables
dotenv_path = os.path.abspath(".env")
load_dotenv(dotenv_path)

#### VARIABLES ####
username = "USERNAME_HERE"
call_language = "english"

###################

# Translation
terms = json.load(open("Terms.json"))

# MongoDB client credentials
mongo_username = os.getenv("MONGO_USERNAME")
mongo_password = os.getenv("MONGO_PASSWORD")
mongo_cluster = os.getenv("MONGO_CLUSTER")

# Mongo cluster
cluster = pymongo.MongoClient(
    "mongodb+srv://{}:{}@{}.mongodb.net/?retryWrites=true&w=majority".format(
        mongo_username, mongo_password, mongo_cluster
    )
)
database = cluster["user_logs"]
collection = database["analysis_reports"]

# Allowing the interviewee to ask clarification questions
print(clean_response(terms[call_language]["follow_up"]))

# Iterate through the loop while there are potential follow-up questions
while True:

    # Ask next question
    current_question = input(clean_response(terms[call_language]["question_prompt"]))

    # If no more questions, break loop
    if current_question.lower() == "0" or len(current_question) == 0:
        break

    next_prompt = clean_response(terms[call_language]["next_prompt"]) + current_question

    # Input conversation
    conversation = [{"role": "system", "content": current_question}]

    # Ask GPT for answer to question
    question_output = makeCall(
        conversation, next_prompt, terms, call_language
    )[0]
    print(clean_response(terms[call_language]["answer"]))
    print(question_output)

    # Store answer in database
    collection.insert_one(
        {
            username: {
                "coaching": {"question": current_question, "answer": question_output}
            }
        }
    )
