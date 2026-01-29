from langchain_community.document_loaders import TextLoader, GithubFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from fastapi.responses import FileResponse
import os
import base64
import itertools
import numpy as np
import pandas as pd
import requests
import math
from functools import reduce
from operator import mul
from dotenv import load_dotenv
from os import environ as env
import json
load_dotenv()

class RepoRequest(BaseModel):
    git_repo: str


class Main_file(BaseModel):
    columns: Dict[str, List[str]] = Field(
        description=(
                    """You are given a single code which has all the functions required to compute the various scores, the input variables are specified in the metadata file along with their data types which are used in the downstream modules.

                    Find the leaf nodes based on the various conditions applied on input variables in the code and for those leaf nodes find the values of input variables which satisfies the all conditional statements, 
                    including default value to trivial 'else' statement, to reach that node, for each leaf node produce a single observation of input variables as key value pairs, if variable is not applicable for that node give any value possible.

                    

                """
        )
    )
    

pydanticparser = PydanticOutputParser(pydantic_object=Main_file)

model = ChatOpenAI(model="gpt-5-mini", temperature=0)

app = FastAPI()

@app.post("/create")
def create_dataset(request: RepoRequest):
    git_repo = request.git_repo
    repo = git_repo.replace("https://github.com/", "")
    api_url = f"https://api.github.com/repos/{repo}/contents"
    token = env["GITHUB_PERSONAL_ACCESS_TOKEN"]

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }


    py_files = []

    urls_to_visit = [api_url]

    while urls_to_visit:
        current_url = urls_to_visit.pop()

        response = requests.get(current_url, headers=headers)
        response.raise_for_status()
        items = response.json()

        for item in items:
            name = item["name"]

            if name.startswith("."):
                continue

            if item["type"] == "dir":
                urls_to_visit.append(item["url"])

            elif item["type"] == "file" and name.endswith(".py"):
                if name != "__init__.py":
                    py_files.append(item["path"])

    py_file_contents = {}
    for file in py_files:
        loader = GithubFileLoader(
            repo=repo,
            branch="main",
            file_path=file,
            access_token=token,
            file_filter=None  
            )  
        name = file.split('/')[-1][:-3]
        try:
            docs = loader.load()
            py_file_contents[name] = docs[0].page_content
        except UnicodeDecodeError:
            response = requests.get(f"https://api.github.com/repos/{repo}/contents/{file}",
                                    headers={"Authorization": f"token {token}"})
            response.raise_for_status()
            content_encoded = response.json()["content"]
            content = base64.b64decode(content_encoded).decode("utf-8", errors="ignore")
            py_file_contents[name] = content

    whole_code = ""

    with open('metadata.json', 'r') as file_handle:
        meta_data = json.load(file_handle)

    whole_code += '\n\n------------------------ ' + 'metadata' + ' file ------------------------\n\n'
    whole_code += str(meta_data)

    for key in py_file_contents.keys():
        whole_code += '\n\n------------------------ ' + key + ' file ------------------------\n\n'
        whole_code += py_file_contents[key]

    prompt = """
        You are given the contents of a Python source file.

        Python file content:
        --------------------
        {file_content}
        --------------------

        {format_instruction}
    """
    temp = PromptTemplate(
        template=prompt,
        input_variables=['file_content'],
        partial_variables={'format_instruction': pydanticparser.get_format_instructions()},
        validate_template=True
    )
    chain1 = temp | model | pydanticparser

    result = chain1.invoke({"file_content": whole_code})

    file_path = "data.json"

    # Dump the dictionary to a JSON file
    try:
        with open(file_path, 'w') as json_file:
            json.dump(result.model_dump()['columns'], json_file, indent=4) # Using indent for pretty-printing
        print(f"Successfully dumped dictionary to {file_path}")
    except IOError as e:
        print(f"Error writing to file: {e}")


    data_observations = []
    for key in result.model_dump()['columns'].keys():
        observation = {}
        for item in result.model_dump()['columns'][key]:
            feature, val = item.split(':')
            observation[feature] = val.strip()
        if observation:
            data_observations.append(observation)

    print(pd.DataFrame(data_observations)[list[meta_data.keys()]])




 