import whisper, openai, os
from pyannote.audio import Inference
from pyannote.audio import Model
from pyannote.audio import Pipeline
from time import sleep
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.abspath(".env")
load_dotenv(dotenv_path)

# Setting the API key for OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to remove bad encoding from GPT-3 response
def clean_response(response):
    # Replacing bad encoding with correct encoding
    response = response.replace("Ã¡", "á")
    response = response.replace("Ã£", "ã")
    response = response.replace("Ã¢", "â")
    response = response.replace("Ã§", "ç")
    response = response.replace("Ã¨", "è")
    response = response.replace("Ã©", "é")
    response = response.replace("Ãª", "ê")
    response = response.replace("Ã³", "ó")
    response = response.replace("Ã´", "ô")
    response = response.replace("Ãº", "ú")
    response = response.replace("Ãµ", "õ")
    response = response.replace("Ã±", "ñ")
    response = response.replace("Ã¼", "ü")
    response = response.replace("Ã¯", "ï")
    response = response.replace("Ã»", "û")
    response = response.replace("Ã®", "î")
    response = response.replace("Ã­", "í")
    response = response.replace("Ã ", "à")

    return response


# Function to transcribe the audio file
def transcribe(filename, model_name, embed=False):

    # Retrieving pyAnnote's pre-trained segmentation model
    pyannote_model = Model.from_pretrained("pytorch_model.bin")

    # pyannote's diarization pipeline
    pyannote_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=os.getenv("PYAUDIO_TOKEN")
    )

    # Loading Whisper model
    whisper_model = whisper.load_model(model_name)

    # Setting device to GPU for speedup
    # whisper_model.to(device="cuda")

    # Applying pyannote's pipeline for diarization
    diarization = pyannote_pipeline(filename)

    # Using pyannote's pretrained model for segmentation
    inference = Inference(pyannote_model, step=5)

    if embed:
        # Embedding of audio file
        embeddings = inference(filename)

    # Result of transcription
    whisper_result = whisper_model.transcribe(filename, verbose=False)

    # Dictionary storing dialogue data with timeframes as keys
    dialogue_data = {}

    # Iterate over speech segments given by Whisper
    for segment in whisper_result["segments"]:
        if segment["start"] not in dialogue_data:
            # Store text and tokens in dictionary
            dialogue_data[segment["start"]] = {"text": segment["text"]}

            # Add tokens if emebeddings is on
            if embed:
                dialogue_data[segment["start"]]["tokens"] = segment["tokens"]

    starting_index = 0
    dialogue_keys = list(dialogue_data.keys())

    # Add speaker label to each speech segment
    for turn, i, speaker in diarization.itertracks(yield_label=True):
        i = 0
        for key in dialogue_keys[starting_index:]:
            if turn.start <= float(key) <= turn.end:
                dialogue_data[key]["speaker"] = speaker
                starting_index = i
                i += 1
                break

    if embed:
        return [dialogue_data, embeddings]
    else:
        return [dialogue_data]


# Function to ask GPT-4 a question
def askGPT(conversation):
    response = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=conversation,
        temperature=1.3
    )
    return response.choices[0].message.content


# Function to make API calls
def makeCall(conversation, current_prompt, terms, call_language):

    # If current prompt is not null, add it to the conversation
    if current_prompt:
        conversation.append({"role": "user", "content": current_prompt})

    # Handling errros in the API calls
    done = False
    attempts = 0
    while not done:
        if attempts > 5:
            current_response = clean_response(terms[call_language]["error_message"])
            break
        try:
            # API call to GPT
            current_response = askGPT(conversation)
            done = True
        except Exception as e:
            print(e)
            sleep(0.5)

        attempts += 1

    # Encoding and decoding the response to avoid UTF-8 errors
    current_response = clean_response(current_response)
    return [current_response, conversation]
