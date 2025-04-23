import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
from tqdm import tqdm

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
        Compare user input embeddings with stored embeddings and return the closest match.
        """
        similarity_scores_db_lst = []
        for json_file_num in tqdm(range(len(os.listdir(merged_embed_dir)))):
            json_file_path = os.path.join(merged_embed_dir, os.listdir(merged_embed_dir)[json_file_num])
            with open(json_file_path, 'r') as file:
                merged_json_data_dict = json.load(file)
                similarity_scores_json_lst = []
                for key in user_embeddings:
                    similarity = self.cosine_similarity(merged_json_data_dict[key], user_embeddings[key])
                    similarity_scores_json_lst.append(similarity)
                similarity_scores_db_lst.append(np.mean(similarity_scores_json_lst))

        # Find the file with the highest similarity
        closest_file = os.listdir(merged_embed_dir)[np.argmax(similarity_scores_db_lst)].replace('.json', '')
        return closest_file



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
Step 2: Assign category for each of the sub-paragraphs. In the following format: sub-paragraph:categroy 
Step 3: all the sub-paragraphs and their respective category should be seperated in the following format :##subparagraph:category##subparagraph:category##


User Input:
Supervise employees in geothermal power plants or well fields, Microsoft Excel - A spreadsheet software used for data analysis, Computers, Calculators.


Sample Output:
##Supervise employees in geothermal power plants or well fields:Tasks##Microsoft Excel - A spreadsheet software used for data analysis:TechnologySkills##Computers:ToolsUsed##Calculators:ToolsUsed##


Your output should look like the sample above, with all entries included and correctly organized. 
...
"""

#User query processing
#user_query = 'Resolve customer complaints regarding problems, such as payout errors, Staffing Organizational Units -  Recruiting, interviewing, selecting, hiring, and promoting employees in an organization.'
## Process query and generate embeddings
#query_processor = UserQueryProcessor(azure_client, few_shot_examples)
#
#classified_query = query_processor.process_query(user_query)
#user_embeddings = query_processor.create_embeddings(classified_query)
## Compare embeddings
#
#comparer = EmbeddingComparer()
#merged_embed_dir = 'MergedEmbeddings'
#closest_file = comparer.compare_embeddings(user_embeddings, merged_embed_dir)
#print(f"File with highest similarity: {closest_file}")

