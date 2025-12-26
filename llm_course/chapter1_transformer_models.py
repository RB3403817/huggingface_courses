# Import the pipeline function from the transformers library
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch


# Create a sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")

# Classify multiple sentences
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)

# Create a zero-shot classification pipeline
# Zero-shot classification allows you to classify text into labels that the model hasn't seen during training
# Its called zero-shot because you don't need to train the model on the labels you want to classify
classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
# Create a text generation pipeline
generator = pipeline("text-generation")
generator("In this course, we will teach you how to")

# Create a text generation pipeline with a specific model
# Here we use the SmolLM2-360M model, which is a smaller model suitable for text generation tasks
# Max_length specifies the maximum length of the generated text
# Num_return_sequences specifies how many different sequences to generate
generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

# TODO - Inference providers

# Create a mask-filling pipeline
# Mask-filling is a task where the model fills in the missing words in a sentence
# Top_k specifies how many different sequences to generate
unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)

# Named entity recognition
# Named entity recognition is a task where the model identifies named entities in a sentence
# We pass grouped_entities=True to group the named entities, for example, "New York" and "Brooklyn" are grouped together
ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn and I live in New York.")

# Question answering
# Question answering is a task where the model answers a question based on given context
# We pass context="Hugging Face was founded in 2016" to provide the context for the question
question_answerer = pipeline("question-answering")
question_answerer(
    question="When was Hugging Face founded?",
    context="Hugging Face was founded in 2016",
)
# Note that this pipeline works by extracting information from the provided context; it does not generate the answer.

# Summarization
# Summarization is the task of reducing a text into a shorter text while keeping all (or most) of the important aspects referenced in the text.
# Here’s an example:
summarizer = pipeline("summarization")
summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)

# Like with text generation, you can specify a max_length or a min_length for the result.

# Translation
# Translation is a task where the model translates text from one language to another.
# Here we translate from English to French.
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
translator("How old are you?")

# Image and audio pipelines
# Beyond text, you can also use pipelines for image and audio tasks.

# Image classification
# Image classification is a task where the model identifies the objects in an image.
# Here we use the Google vit base model

image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
result = image_classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
print(result)

# Speech recognition
# Speech recognition is a task where the model transcribes audio into text.
# Set device to CUDA GPU if available, otherwise use CPU for computation
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Use float16 precision on GPU for faster inference and lower memory usage, float32 on CPU for compatibility
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Specify the Whisper model ID from OpenAI (large-v3 version for high accuracy)
model_id = "openai/whisper-large-v3"

# Load the pre-trained Whisper model for speech-to-text conversion
# low_cpu_mem_usage=True optimizes memory usage during loading
# use_safetensors=True uses the safer tensor format for model weights
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
# Move the model to the specified device (GPU or CPU)
model.to(device)

# Load the processor which handles tokenization and feature extraction for audio
processor = AutoProcessor.from_pretrained(model_id)

# Create the automatic speech recognition pipeline
# This pipeline combines the model, tokenizer, and feature extractor for easy inference
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Run speech recognition on the audio file "mlk.flac"
result = pipe("/data/mlk.flac")
# Print the transcribed text from the audio file
print(result["text"])
