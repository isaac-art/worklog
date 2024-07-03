import datetime
import os
import json
from openai import OpenAI

from settings import *

# OpenAI API setup
client = OpenAI(api_key=OPEN_AI_KEY)

def get_unique_filename(base_dir, base_name, extension):
    counter = 0
    while True:
        if counter == 0:
            file_name = f"{base_name}.{extension}"
        else:
            file_name = f"{base_name}_{chr(96 + counter)}.{extension}"
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            return file_path
        counter += 1

def create_diary_file():
    # Get today's date
    today = datetime.date.today()
    date_str = today.strftime('%Y_%m_%d')
    base_name = date_str
    extension = "md"
    file_name = f"{date_str}.md"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(script_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_path = get_unique_filename(log_dir, base_name, extension)
    
    # Get user input for each section
    today_content = input("Today: ")
    problems_content = input("Problems: ")
    findings_content = input("Findings: ")
    questions_content = input("Questions: ")
    
    # Create the content
    content = (
        f"# {date_str}\n\n"
        f"# Today\n\n{today_content}\n\n"
        f"# Problems\n\n{problems_content}\n\n"
        f"# Findings\n\n{findings_content}\n\n"
        f"# Questions\n\n{questions_content}\n"
    )
    
    # Write the content to the file
    with open(file_path, 'w') as file:
        file.write(content)
    
    print(f"Diary file '{file_path}' created successfully.")
    embeddings = generate_embeddings(content)
    embeddings_file_path = f"{file_path[:-3]}.json"
    with open(embeddings_file_path, 'w') as file:
        json.dump(embeddings, file)
    print(f"Embeddings file '{embeddings_file_path}' created successfully.")
    

def generate_embeddings(text):
    embeddings = client.embeddings.create(
        model=EMBEDDINGS_MODEL,
        input=text
    )
    return embeddings.data[0].embedding


# Run the function
if __name__ == "__main__":
    create_diary_file()