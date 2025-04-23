import os
import re
import json
import orjson
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import random
#nltk.download('punkt')
#nltk.download('punkt_tab')

# Class to handle Azure OpenAI interactions
class AzureAIClient:
    def __init__(self, azure_endpoint, api_key, api_version):
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )

    def classify_user_query(self, query, few_shot_examples, model="Sashank"):
        """
        Function to classify the user query into one of the categories.
        """
        prompt = few_shot_examples + f"Statement: {query}\nClassification:"
        message_text = [{"role": "user", "content": f'{prompt} Now classify the following statements into their respective categories and present the output in the requested clean dictionary format: {query}'}]

        completion = self.client.chat.completions.create(
            model=model,
            messages=message_text,
            temperature=0.1,
            max_tokens=4096,
            top_p=0.95,
        )
        classification = completion.choices[0].message.content.strip()
        return classification

    def get_embedding(self, query, model="embedding-ada"):
        """
        Generate embeddings for a given query using Azure OpenAI embeddings API.
        """
        response = self.client.embeddings.create(input=[query], model=model)
        return response.data[0].embedding



# Class to handle user query processing
class UserQueryProcessor:
    def __init__(self, azure_client, few_shot_examples):
        self.azure_client = azure_client
        self.few_shot_examples = few_shot_examples

    def process_query(self, query):
        """
        Classifies the user query and returns a dictionary with the categorized sub-paragraphs.
        """
        output = self.azure_client.classify_user_query(query, self.few_shot_examples)
        categories, statements = [], []
        
        for entity in output.split('##'):
            if entity:
                categories.append(entity.split(':')[1])
                statements.append(entity.split(':')[0])

        result_dict = {}
        for cat, stat in zip(categories, statements):
            if cat not in result_dict:
                result_dict[cat] = stat
            else:
                result_dict[cat] += ', ' + stat

        return result_dict

    def create_embeddings(self, query_dict):
        """
        Creates embeddings for each statement in the categorized query dictionary.
        """
        embeddings = {}
        for category, statement in query_dict.items():
            embeddings[category] = self.azure_client.get_embedding(statement)
        return embeddings

# Class to handle cosine similarity and embedding comparisons
class EmbeddingComparer:
    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings.
        """
        embedding1, embedding2 = np.array(embedding1), np.array(embedding2)
        dot_product = np.dot(embedding1, embedding2)
        norm1, norm2 = np.linalg.norm(embedding1), np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)

    def compare_embeddings(self, user_embeddings, merged_embed_dir):
        """
        Compare user input embeddings with stored embeddings and return the top 5 closest matches.
        """
        file_similarity_scores = []
        for json_file_name in tqdm(os.listdir(merged_embed_dir)):
            json_file_path = os.path.join(merged_embed_dir, json_file_name)
            with open(json_file_path, 'rb') as file:
                merged_json_data_dict = orjson.loads(file.read())
            # with open(json_file_path, 'r') as file:
            #     merged_json_data_dict = json.load(file)
                # Find common categories between user input and database entry
                common_keys = set(user_embeddings.keys()) & set(merged_json_data_dict.keys())
                if not common_keys:
                    # No common categories, skip this file
                    continue
                similarity_scores_json_lst = []
                for key in common_keys:
                    similarity = self.cosine_similarity(merged_json_data_dict[key], user_embeddings[key])
                    similarity_scores_json_lst.append(similarity)
                # For this file, compute the average similarity
                avg_similarity = np.mean(similarity_scores_json_lst)
                file_similarity_scores.append((json_file_name.replace('.json', ''), avg_similarity))
        # Sort the files by similarity scores
        file_similarity_scores.sort(key=lambda x: x[1], reverse=True)
        # Return top 5 matches
        top_matches = file_similarity_scores[:5]
        return top_matches

class DocumentRanker:
    def __init__(self, tokenized_json_dir):
        """
        Initialize the DocumentRanker with the directory containing tokenized documents.
        Args:
            tokenized_json_dir (str): Path to the directory containing tokenized documents.
        """
        self.tokenized_json_dir = tokenized_json_dir
        self.tokenized_docs = self.load_tokenized_docs()

    def load_tokenized_docs(self):
        """
        Load and tokenize documents from the specified directory.
        Returns:
            list: A list of tokenized documents.
        """
        tokenized_docs = []
        for file_name in tqdm(os.listdir(self.tokenized_json_dir)):
            with open(os.path.join(self.tokenized_json_dir, file_name), 'r') as file:
                tokens = [line.strip() for line in file]
                tokenized_docs.append(tokens)
        return tokenized_docs

    def rank_documents(self, query):
        """
        Rank documents based on a query using the BM25 algorithm.
        Args:
            query (str): The search query.
        Returns:
            list: A list of the top 5 ranked documents.
        """
        # Initialize BM25
        bm25 = BM25Okapi(self.tokenized_docs)
        # Tokenize the query
        tokenized_query = word_tokenize(query.lower())
        # Get BM25 scores for each document
        doc_scores = bm25.get_scores(tokenized_query)
        # Sort documents by score
        ranked_docs = sorted(zip(doc_scores, self.tokenized_docs), reverse=True)
        # Display the top 5 documents
        top_5_docs = [doc[0] for score, doc in ranked_docs[:5]]
        return top_5_docs


class SOCCodeEvaluator:
    def __init__(self, azure_client):
        """
        Initialize the SOCCodeEvaluator with the Azure OpenAI client.
        """
        self.azure_client = azure_client

    def evaluate_soc_codes(self, soc_codes, user_query, model="Sashank"):
        """
        Function to evaluate the top 10 SOC codes using LLM and return the best match.
        Args:
            soc_codes (list): List of 10 SOC codes (from cosine similarity and BM25).
            user_query (str): The user query to match with SOC codes.
        Returns:
            str: The best-matching SOC code as determined by the LLM.
        """
        formatted_data = " "
        for soc_data in soc_codes:
            formatted_data += f"SOC Code: {soc_data['SocCode']}\n"
            formatted_data += f"Occupation: {soc_data['Occupation']}\n"
            formatted_data += f"Related Job Titles: {', '.join(soc_data['ReportedJobTitles'])}\n\n"

        prompt = f"""
        You are an expert in job role classifications. Based on the user's query, select the best matching SOC code from the provided list.
        
        User query: {user_query}
        
        Here are the SOC codes with related information:
        {formatted_data}
        
        Based on the user's query, which SOC code is the best match and why? Please provide the SOC code and a short explanation in 3-4 sentences.
        """
        
        # Send the prompt to Azure OpenAI
        message_text = [{"role": "user", "content": prompt}]
        
        completion = self.azure_client.client.chat.completions.create(
            model=model,
            messages=message_text,
            temperature=0.1,
            max_tokens=1000,
            top_p=0.95,
        )
        
        # Extract the best match SOC code from the LLM response
        best_match = completion.choices[0].message.content.strip().replace("**", "")
        # # Formatting Output
        # pattern = r'\b\d{2}-\d{4}\.\d{2}\b'
        # match = re.search(pattern, best_match)
        # soc_code = match.group()
        # matched_soc_code_file_path = f'MergedJSON/{soc_code}.json'
        # with open(matched_soc_code_file_path, 'r') as file:
        #     matched_soc_code_file = json.load(file)
        # similar_job_titles = matched_soc_code_file['ReportedJobTitles'].split(':')[1]
        # description = best_match
        # return f"SOC Code: {soc_code}\nSimilar Job Titles: {similar_job_titles}\nDescription: {description}"

        try:
            soc_code_part = best_match.split("SOC Code")[1].split(".")[0].strip()
            explanation_part = best_match.split(". ", 1)[1].strip()
            print(soc_code_part)
            print(explanation_part)
            response_templates = [
            f"Based on your experience and query, the most suitable SOC code is {soc_code_part} . {explanation_part}",
            f"After analyzing your input, the SOC code that fits best is {soc_code_part}. {explanation_part}",
            f"Considering the information provided, the recommended SOC code is {soc_code_part}. {explanation_part}",
            f"The most relevant SOC code matching your description is {soc_code_part}. {explanation_part}",
            f"Upon reviewing your query, I suggest the SOC code {soc_code_part} as the best match. {explanation_part}"
        ]
            response_message = random.choice(response_templates)
            return response_message

        except IndexError:
            soc_code_part = best_match  # In case it doesn't follow expected structure
            explanation_part = ""
            return soc_code_part

       

    def extract_data_for_soc_codes(self, soc_codes):
        """
        Iterate over SOC codes, fetch the relevant data from JSON files, and store the results in a list of dictionaries.
        Args:
            soc_codes (list): List of SOC codes to fetch data for.
        Returns:
            list: A list of dictionaries with extracted data for each SOC code.
        """
        soc_code_data_list = []
        
        for soc_code in soc_codes:
            json_file_path = os.path.join('JSON', f"MergedJSON/{soc_code}.json")        
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as file:
                    soc_data = json.load(file)
                    
                    # Extract relevant fields from the JSON (this assumes specific keys are available)
                    soc_code_data = {
                        "SocCode": soc_data.get("SocCode", "N/A"),
                        "Occupation": soc_data.get("Occupation", "N/A"),
                        "ReportedJobTitles": soc_data.get("ReportedJobTitles", []).lower()[len("sample of reported job titles:"):].strip(),
                        # Add other fields as necessary
                    }
                    soc_code_data_list.append(soc_code_data)
            else:
                print(f"JSON file for SOC code {soc_code} not found.")
        
        return soc_code_data_list


# Set up your Azure OpenAI client
azure_client = AzureAIClient(
    azure_endpoint="https://aoai-camp.openai.azure.com/",
    api_key="d3cf2354e3a34dae97c6520b3eeb2f5e",
    api_version="2024-02-15-preview"
)

# Few-shot examples for the classification
few_shot_examples = """
You are an AI expert. Your task is to split the given paragraph into sub-paragraphs based on given categories and return the output in an organized format.

Specified categories: Occupation, Description, ReportedJobTitles, Tasks, TechnologySkills, ToolsUsed, WorkActivities, DetailedWorkActivities, WorkContext, Skills, Knowledge, Abilities, WorkStyles.
Definitions:
- **Occupation**: The job title or profession.
- **Description**: A brief overview of the occupation’s main responsibilities.
- **ReportedJobTitles**: Common titles under the occupation.
- **Tasks**: Specific duties or tasks performed as part of the job.
- **TechnologySkills**: Technical skills or software used in the job.
- **ToolsUsed**: Physical or digital tools utilized in the role.
- **WorkActivities**: General activities performed at work.
- **DetailedWorkActivities**: Specific, detailed work actions.
- **WorkContext**: The environment or setting of the work.
- **Skills**: Skills needed to perform the job.
- **Knowledge**: Knowledge areas relevant to the job.
- **Abilities**: Abilities required to perform the job effectively.
- **WorkStyles**: Work styles or personality traits relevant to the job.

Instructions:
Step 1: Split the given paragraph into sub-paragraphs based on list of categories provided above. 
Step 2: Assign category for each of the sub-paragraphs. In the following format: sub-paragraph:category 
Step 3: all the sub-paragraphs and their respective category should be separated in the following format:##subparagraph:category##subparagraph:category##


User Input:
Supervise employees in geothermal power plants or well fields, Microsoft Excel - A spreadsheet software used for data analysis, Computers, Calculators.

Sample Output:
##Supervise employees in geothermal power plants or well fields:Tasks##Microsoft Excel - A spreadsheet software used for data analysis:TechnologySkills##Computers:ToolsUsed##Calculators:ToolsUsed##

Your output should look like the sample above, with all entries included and correctly organized. 
...
"""

# User query processing
query_processor = UserQueryProcessor(azure_client, few_shot_examples)

user_query = """
I have 8 years of experience as a DevOps Engineer, managing cloud infrastructure and automating deployment pipelines for web applications. My daily tasks include working with tools like Jenkins, Docker, Kubernetes, and Terraform to build and maintain CI/CD pipelines, ensuring smooth code deployment from development to production environments.
I have extensive experience with AWS services like EC2, S3, Lambda, and CloudFormation to architect scalable solutions, and I often collaborate with development and operations teams to resolve issues related to application performance and infrastructure stability.
Additionally, I monitor system performance using tools like Prometheus and Grafana, and I have been responsible for setting up automated alerts to ensure system uptime and quick resolution of incidents. I also manage configuration management tools such as Ansible and Chef to automate server provisioning and software updates.
In previous roles, I’ve worked on migrating legacy systems to cloud-based infrastructure and implemented container orchestration to improve deployment efficiency.
"""


## Process query and generate embeddings
classified_query = query_processor.process_query(user_query)
user_embeddings = query_processor.create_embeddings(classified_query)
#
## Compare embeddings
comparer = EmbeddingComparer()
merged_embed_dir = 'MergedEmbeddings'
top_matches = comparer.compare_embeddings(user_embeddings, merged_embed_dir)
top_matches_cosine_similarity = [match[0] for match in top_matches]
print(top_matches_cosine_similarity)

tokenized_json_dir = 'TokenizedJSON'
ranker = DocumentRanker(tokenized_json_dir)
top_documents = ranker.rank_documents(user_query)
print(top_documents)

combined_soc_codes = list(set(top_matches_cosine_similarity + top_documents))

soc_code_evaluator = SOCCodeEvaluator(azure_client)
#best_soc_code = soc_code_evaluator.evaluate_soc_codes(combined_soc_codes, user_query)

#print(f"The best SOC code match is: {best_soc_code}")

soc_code_data=soc_code_evaluator.extract_data_for_soc_codes(combined_soc_codes)
best_soc_code = soc_code_evaluator.evaluate_soc_codes(soc_code_data, user_query)
print(best_soc_code)

print(50*'*')
combined_soc_codes = list(set(top_matches_cosine_similarity[:2] + top_documents[:2]))
for soc_code in combined_soc_codes:
    file_path = f'MergedJSON/{soc_code}.json'
    with open(file_path, 'r') as file:
        top_files_data = json.load(file)
    print(f"SocCode: {top_files_data['SocCode']}, Occupation: {top_files_data['Occupation']}")
print(combined_soc_codes)
