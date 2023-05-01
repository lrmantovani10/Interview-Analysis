import math, nltk, subprocess, pymongo, json, copy, jinja2, pdfkit, base64
from utils import *

# Function to convert image to binary
def image_base64_data(filename):
    with open(filename, "rb") as image_file:
        return base64.b64encode(image_file.read())


############# User-defined variables #############
# Username of the current user
username = "CUSTOM_NAME_HERE"

# Filename transcribed
filename = "FILANEME_HERE"

# Language used, in lowercase
call_language = "english"

# Company name
company_name = "EXAMPLE_NAME"

##################################################

# Load environment variables
dotenv_path = os.path.abspath(".env")
load_dotenv(dotenv_path)

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

# Split filename into name and extension
splitted_filename = filename.split(".")

# If not .wav format, convert to .wav
if splitted_filename[-1] != "wav":

    # If not .wav format, define .wav output filename
    output_filename = splitted_filename[0] + ".wav"

    # Use existing .wav file
    if os.path.exists(output_filename):
        filename = output_filename
    else:
        # Convert file format
        subprocess.call(["ffmpeg", "-i", filename, output_filename])
        filename = output_filename

# Dictionary of terms in different languages
terms = json.load(open("Terms.json"))

# Name of file containing transcription
transcription_filename = filename + "_transcription.txt"

# Determine whether we need to conduct the transcription or whether the call has
# already been transcribed by checking if the transcription file exists in the directory
transcription_needed = not os.path.exists(transcription_filename)
if transcription_needed:

    # Transcribing audio using helper function
    try:
        transcription = transcribe(filename, "base", False)[0]
    except:
        print(clean_response(terms[call_language]["transcription_error"]))
        quit()

    # Writing transcription to .txt file
    with open(transcription_filename, "w", encoding="utf-8") as f:

        # Identifying the first speaker
        speaker = None
        for val in transcription.values():
            if "speaker" in val:
                speaker = val["speaker"]
                break

        # Iterating through transcription dictionary to generate text file
        # with speaker and their words
        for value in transcription.values():
            if "speaker" in value:
                speaker = value["speaker"]
            if "text" in value:
                f.write(speaker + ": " + value["text"] + "\n")

# Read dialogue transcribed to file
dialogue = ""
with open(transcription_filename, "r", encoding="utf-8") as f:
    dialogue = f.read()

# Message exchange provided to model
starting_prompt = clean_response(terms[call_language]["starting_prompt"])

# Store transcription in the database if it is not already there
if transcription_needed:
    collection.insert_one({username: {filename: {"transcription": dialogue}}})

conversation = [{"role": "system", "content": starting_prompt}]

# Maximum amount of tokens per request. Available at OpenAI's website
token_limit = 8192

# Split transcription into portions bounded by the maximum token number
tokenized_dialogue = nltk.word_tokenize(dialogue)

# Number of tokens in the dialogue
total_length = len(tokenized_dialogue)

# Provide the model with the transcription of the dialogue
conversation.append(
    {
        "role": "assistant",
        "content": clean_response(terms[call_language]["start_of_conversation"]),
    }
)

# Get number of tokens already in request
prompt_tokens = 0
for d in conversation:
    for value in d.values():
        prompt_tokens += len(nltk.word_tokenize(value))

# Clean ending message
ending_message = clean_response(terms[call_language]["end_of_conversation"])
prompt_tokens += len(nltk.word_tokenize(ending_message))

# Number of tokens available for dialogue. Needed because the sum of tokens in
# the request and response must be equal to the token limit
response_tokens = 2500
effective_tokens = token_limit - response_tokens - prompt_tokens

# Running token count
token_count = prompt_tokens
valid_range = math.ceil(total_length / effective_tokens)

# Ask for analysis along 5 axes of interest
output_filename = filename + "_analysis.txt"

# The context dicitonary stores the variables inserted in the pdf template
context = {
    "analysis": company_name
    + " "
    + clean_response(terms[call_language]["analysis"])
    + ":"
}

# Axes of call analysis
axes = [clean_response(term) for term in terms[call_language]["axes"]]

# alphabet
abc = ["a", "b", "c", "d", "e"]

# Iterate through dialogue and send to GPT-4
for i in range(valid_range):

    # Don't overwrite conversation
    temporary_conversation = copy.deepcopy(conversation)

    # 1-based indexing
    effective_i = i + 1

    # Range of tokens we are analyzing
    separated_text = tokenized_dialogue[
        i * effective_tokens : effective_i * effective_tokens
    ]
    joined_text = separated_text[0]

    # Decide on call segment based on index
    if i == 0:
        context["introduction"] = clean_response(terms[call_language]["intro_name"])
        chosen_segment = "introduction_text"
        context[chosen_segment] = ""

    elif 0 < i < valid_range - 1:
        chosen_segment = "middle_text"
        if i == 1:
            context["middle"] = clean_response(terms[call_language]["part_name"])
            context[chosen_segment] = ""

    else:
        context["conclusion"] = clean_response(terms[call_language]["end_name"])
        chosen_segment = "conclusion_text"
        context[chosen_segment] = ""

    # Add token to prompt
    for a in separated_text[1:]:
        if len(a) > 1:
            joined_text += " " + a
        else:
            joined_text += a

    temporary_conversation.append({"role": "assistant", "content": joined_text})
    temporary_conversation.append({"role": "assistant", "content": ending_message})


    # Write analysis along each axis to output file
    with open(output_filename, "a", encoding="utf-8") as f:

        # Iterate over axes
        for i, axis in enumerate(axes):

            # Building prompt
            current_prompt = (
                clean_response(terms[call_language]["axis_prompt1"])
                + axis
                + clean_response(terms[call_language]["axis_prompt2"])
            )

            # Ask GPT-4 for analysis
            current_response, temporary_conversation = makeCall(
                temporary_conversation, current_prompt, terms, call_language
            )
            
            # Write current insight to output file
            context[chosen_segment] += "<h3>"
            context[chosen_segment] += clean_response(abc[i] + ". " + axis)
            context[chosen_segment] += "</h3>"
            context[chosen_segment] += "<p>"
            context[chosen_segment] += clean_response(current_response)
            context[chosen_segment] += "</p>"

            # Store analysis in database
            collection.insert_one(
                {username: {(filename + str(effective_i)): {"axis": current_response}}}
            )

        # Other important insights from the call
        extra_prompts = [
            clean_response(term) for term in terms[call_language]["extra_prompts"]
        ]

        prompt_headers = [
            clean_response(term) for term in terms[call_language]["prompt_headers"]
        ]

        # Iterating over additional prompts
        for i in range(len(extra_prompts)):

            current_response, temporary_conversation = makeCall(
                temporary_conversation, extra_prompts[i], terms, call_language
            )

            # Write current insight to output file
            context[chosen_segment] += "<h3>"
            context[chosen_segment] += clean_response(prompt_headers[i])
            context[chosen_segment] += "</h3>"
            context[chosen_segment] += "<p>"
            context[chosen_segment] += clean_response(current_response)
            context[chosen_segment] += "</p>"

            # Store insight generated by GPT in the database
            collection.insert_one(
                {
                    username: {
                        (filename + str(effective_i)): {
                            prompt_headers[i].split(":")[0]: current_response
                        }
                    }
                }
            )

# Load template and render information stored in the
# context dictionary in pdf
template_loader = jinja2.FileSystemLoader(searchpath="./")
template_env = jinja2.Environment(loader=template_loader)
template = template_env.get_template("pdf_template.html")
output_text = template.render(context)

# Successfully export to PDF
config = pdfkit.configuration(
    wkhtmltopdf="PATH_TO_WKHTMLTOPDF"
)
pdfkit.from_string(
    output_text,
    "Analysis.pdf",
    configuration=config,
    css="pdf_style.css",
    options={"enable-local-file-access": ""},
)
