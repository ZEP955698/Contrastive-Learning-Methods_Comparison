from models import *
from config import *
from utils import *
import os 
import pandas as pd
import random


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

    def get_folder_names(directory_path):
        folder_names = [item for item in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, item))]
        return folder_names

    # Replace 'your_directory_path' with the actual path of the directory you want to get folder names from
    # directory_path = '/data/yf_center/Embeddings_demo2/outputs_QA'
    directory_path = '/data/yf_center/Embeddings_demo2/outputs'
    folders = get_folder_names(directory_path)

    all_data = pd.read_csv("./knowledge_base/csv/test_data_sts.csv").dropna()
    # all_data = all_data.sample(frac=1, random_state=42)
    queries = all_data.iloc[:500,:]["Positive"].to_list()
    answers = all_data.iloc[:500,:]["Anchor"].to_list()
    output_dir = "./knowledge_base/csv/results_sts.txt"
    with open(output_dir, "w",) as file:
        pass
    for model_name in sorted(folders):
        model, embeddings_dataset = csv_to_faiss(model_name, "test_data_sts.csv", "Positive")
        for topn in [3,2,1]:
            correct = 0
            total = len(queries)
            for i in range(total):
                samples_df = search_similar_text(model, queries[i], embeddings_dataset, topn=topn, threshold=0)
                if answers[i] in samples_df["Anchor"].tolist():
                    correct += 1
                else:
                    print(answers[i])
                    print(samples_df["Anchor"].tolist())
                    print("=====")
            with open(output_dir, "a",) as file:
                file.write(f"{model_name}||{topn}||{correct}||{total}||{correct/total}")
                file.write("\n")
                    
            
            
        


            