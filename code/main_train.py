from models import *


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                level=logging.INFO,
                handlers=[LoggingHandler()])

    train()
    # with open("./datasets_QA/QA-16k-sup-STS-train.txt", "r", encoding='UTF-8') as f:
    #     for i in f:
    #         i = i.replace("\n", "")
    #         print(i)
    #         data = i.split("||")
    #         print(len(data))
    #         break
