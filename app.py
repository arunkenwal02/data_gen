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
load_dotenv()

class RepoRequest(BaseModel):
    git_repo: str


class Main_file(BaseModel):
    columns: Dict[str, List[str]] = Field(
        description=(
                    """Extract all input feature names that appear as keys in the application dictionary.
                For each feature, generate one representative value per conditional branch that strictly satisfies the corresponding if/elif/else condition by using only the literal threshold values explicitly compared in the code and selecting values that fall inside the valid comparison range (for example, for dti < 0.2, generate a value strictly less than 0.2, not 0.2 itself).

                If a feature appears in conditional logic and has an else or implicit default branch,
                include one representative value that satisfies the else condition (i.e., outside all preceding if/elif comparisons).

                If a feature is not used in any conditional comparison, return an empty list for it.
                Do not infer values from example inputs, test data, or any other sources.
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
    main_py_file = None

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
                if name == "main.py":
                    main_py_file = item["path"]
                elif name != "__init__.py":
                    py_files.append(item["path"])

    loader = GithubFileLoader(
            repo=repo,
            branch="main",
            file_path=main_py_file,
            access_token=token,
            file_filter=None  
            )  
    try:
        doc = loader.load()
        main_file_content = doc[0].page_content
    except UnicodeDecodeError:
        response = requests.get(f"https://api.github.com/repos/{repo}/contents/{main_py_file}", headers=headers)
        content_encoded = response.json()["content"]
        main_file_content = base64.b64decode(content_encoded).decode("utf-8", errors="ignore")

    py_file_contents = {}
    for file in py_files:
        loader = GithubFileLoader(
            repo=repo,
            branch="main",
            file_path=file,
            access_token=token,
            file_filter=None  
            )  
        name = file.split('/')[-1][:-3] + "_content"
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
    main_result = chain1.invoke({'file_content':main_file_content})
    data = dict(main_result)
    allowed_keys = data['columns'].keys()
    filtered_columns = {k: v for k, v in data['columns'].items() if v}

    dict_result = []

    for file_name, file_content in py_file_contents.items():
        result = chain1.invoke({"file_content": file_content})
        dict_result.append(
            result.model_dump()['columns']
        )

    filtered_list = [
    {k: v for k, v in d.items() if k in allowed_keys and v} 
    for d in dict_result
    ]
    filtered_list.append(filtered_columns)
    merged_dict = {}

    for d in filtered_list:
        for key, values in d.items():
            if key not in merged_dict:
                merged_dict[key] = []
            for v in values:
                if v not in merged_dict[key]:
                    merged_dict[key].append(v)

    EXCEL_MAX_ROWS = 1_048_576
    keys = list(merged_dict.keys())
    values = [merged_dict[k] for k in keys]
    total_rows = reduce(mul, (len(v) for v in values), 1)

    print(f"Total combinations: {total_rows}")

    if total_rows <= EXCEL_MAX_ROWS:
        all_combinations = list(itertools.product(*values))
        df = pd.DataFrame(all_combinations, columns=keys)
        df.to_excel("data.xlsx", index=False)

    else:
        all_combinations = itertools.islice(
            itertools.product(*values),
            EXCEL_MAX_ROWS
        )
        df = pd.DataFrame(all_combinations, columns=keys)
        df.to_excel("data.xlsx", index=False)

    print(f"Rows written: {len(df)}")
    return FileResponse(
        path="data.xlsx",
        filename="data.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    





 