# Imports the Google Cloud client library
from google.cloud import translate
from tqdm import tqdm
import json

# Instantiates a client
translate_client = translate.Client()

def translate_txt(fp, source='en', targets=['ru'], output='output.json'):
    
    # read file
    input_lines = []
    
    with open(fp) as input_file:
        input_lines = input_file.readlines()
    
    # deduplicate (if any)
    input_lines = list(set([line.strip() for line in input_lines]))

    translate_client = translate.Client()
    output_dict = {}
    
    for target in targets:
        output_dict[target] = []
        
        for text in tqdm(input_lines, desc=target):
            translation = translate_client.translate(
                text,
                source_language=source
                target_language=target
            )
            output_dict[target].append(translation['translatedText'])
    
            json.dump(output_dict, open(output))

