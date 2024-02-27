from models import *
from config import *
from utils import *
import os 


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

    def get_folder_names(directory_path):
        folder_names = [item for item in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, item))]
        return folder_names

    # Replace 'your_directory_path' with the actual path of the directory you want to get folder names from
    output_folder = "outputs_QA"
    directory_path = f'/data/yf_center/Embeddings_demo2/{output_folder}'
    folders = get_folder_names(directory_path)
    # 1. 或者从outputs文件夹中读取模型

    model_dir = f"./{output_folder}"
    for model_name in sorted(folders,):

        # model_name  = "consert-qa-unsup-snli-macbert-2-TripletLoss-2023-08-04_14-58-34"
        model_name_or_path = os.path.join(model_dir, model_name)

        model = SentenceTransformer(model_name_or_path=model_name_or_path)
        
        # DEVDATA_PATH     = "./datasets/STS-B/cnsd-sts-dev.txt"
        # TESTDATA_PATH    = "./datasets/STS-B/cnsd-sts-test.txt"
        DEVDATA_PATH     = "./datasets_QA/QA-sup-STS-dev.txt"
        TESTDATA_PATH    = "./datasets_QA/QA-sup-STS-test.txt"

        dev_samples   = load_sts_sup(DEVDATA_PATH)
        test_samples  = load_sts_sup(TESTDATA_PATH)

        dev_evaluator = init_evaluator(dev_samples, f'{model_name}-dev')
        test_evaluator = init_evaluator(test_samples, f'{model_name}-test')

        dev_evaluator(model)
        test_evaluator(model)
        print("=================================\n")
        
    


        