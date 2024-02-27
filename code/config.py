from datetime import datetime

dataset_dict = {
    "sup-snli":{
        "train":"./datasets/SNLI/cnsd_snli_v1.0.train.jsonl",
        "dev":"./datasets/STS-B/cnsd-sts-dev.txt",
        "test":"./datasets/STS-B/cnsd-sts-test.txt",
    },
    "unsup-snli":{
        "train":"./datasets/SNLI/train.txt",
        "dev":"./datasets/STS-B/cnsd-sts-dev.txt",
        "test":"./datasets/STS-B/cnsd-sts-test.txt",
    },
    "sup-sts":{
        "train":"./datasets/STS-B/cnsd-sts-train.txt",
        "dev":"./datasets/STS-B/cnsd-sts-dev.txt",
        "test":"./datasets/STS-B/cnsd-sts-test.txt",
    },
    "unsup-sts":{
        "train": "./datasets/STS-B/cnsd-sts-train_unsup.txt",
        "dev":"./datasets/STS-B/cnsd-sts-dev.txt",
        "test":"./datasets/STS-B/cnsd-sts-test.txt",
    },
    "qa-sup-snli":{
        "train":"./datasets_QA/QA-sup-SNLI-train.jsonl",
        "dev":"./datasets/STS-B/cnsd-sts-dev.txt",
        "test":"./datasets/STS-B/cnsd-sts-test.txt",
        "qa-dev":"./datasets_QA/QA-sup-STS-dev.txt",
        "qa-test":"./datasets_QA/QA-sup-STS-test.txt",
    },
    "qa-unsup-snli":{
        "train":"./datasets_QA/QA-unsup-SNLI-train.txt",
        "dev":"./datasets/STS-B/cnsd-sts-dev.txt",
        "test":"./datasets/STS-B/cnsd-sts-test.txt",
        "qa-dev":"./datasets_QA/QA-sup-STS-dev.txt",
        "qa-test":"./datasets_QA/QA-sup-STS-test.txt",
    },
    "qa-sup-sts":{
        "train":"./datasets_QA/QA-sup-STS-train.txt",
        "dev":"./datasets/STS-B/cnsd-sts-dev.txt",
        "test":"./datasets/STS-B/cnsd-sts-test.txt",
        "qa-dev":"./datasets_QA/QA-sup-STS-dev.txt",
        "qa-test":"./datasets_QA/QA-sup-STS-test.txt",
    },
    "qa-unsup-sts":{
        "train": "./datasets_QA/QA-unsup-STS-train.txt",
        "dev":"./datasets/STS-B/cnsd-sts-dev.txt",
        "test":"./datasets/STS-B/cnsd-sts-test.txt",
        "qa-dev":"./datasets_QA/QA-sup-STS-dev.txt",
        "qa-test":"./datasets_QA/QA-sup-STS-test.txt",
    }
}

bert_model_dict  = {
    "bert":"./models/chinese-bert-wwm-ext",
    "macbert":"./models/chinese-macbert-base",
    "roberta":"./models/chinese-roberta-wwm-ext",
    "rbt6":"./models/rbt6",
}

# 模型及数据集变量 
LEARNING_METHOD = "consert" # ["sbert", "simcse", "esimcse", "consert"]

BERT_MODEL = "macbert" # ["bert", "macbert", "roberta", "rbt6"]
BERT_PATH = bert_model_dict[BERT_MODEL]



# DATATYPE = "sup-snli"
# DATATYPE = "sup-sts" 
# DATATYPE = "unsup-snli" 
# DATATYPE = "unsup-sts" 

# DATATYPE = "qa-sup-snli"
# DATATYPE = "qa-sup-sts" 
# DATATYPE = "qa-unsup-snli" 
DATATYPE = "qa-unsup-sts" 


UNSUPERVISED = "unsup" in DATATYPE
SUPERVISED = not UNSUPERVISED 

TRAINDATA_PATH   = dataset_dict[DATATYPE]["train"]


if "qa-" in DATATYPE:
    DEVDATA_PATH     = dataset_dict[DATATYPE]["qa-dev"]
    TESTDATA_PATH    = dataset_dict[DATATYPE]["qa-test"]
else:
    DEVDATA_PATH     = dataset_dict[DATATYPE]["dev"]
    TESTDATA_PATH    = dataset_dict[DATATYPE]["test"]

POOLING_MODE = "cls"
POOLING_MODE_CLS_TOKEN = True

# 训练时变量
BATCH_SIZE = 256
NUM_EPOCHS = 2
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 64
WARMUP_RATIO = 0.1
EVALUATION_RATIO = 0.1
DEVICE = "cuda:3"
USE_AMP = True # 是否使用混合精度训练 (if GPU supports FP16 cores)


if LEARNING_METHOD in ["esimcse"]:
    LOSS_FUNCTION = "MultipleNegativesRankingLoss_embeddings" 
elif DATATYPE in ["sup-snli", "sup-sts", "qa-sup-snli", "qa-sup-sts"]:
    LOSS_FUNCTION = "CosineSimilarityLoss" 
    # LOSS_FUNCTION = "OnlineContrastiveLoss"
elif DATATYPE in ["unsup-snli","qa-unsup-snli"]:
    LOSS_FUNCTION = "TripletLoss" # ["TripletLoss"]
    TRIPLET_MARGIN = 0.5
elif DATATYPE in ["unsup-sts", "qa-unsup-sts"] and LEARNING_METHOD in ["sbert", "simcse", "consert"]:
    LOSS_FUNCTION = "MultipleNegativesRankingLoss" 
else:
    raise Exception("Unkown LOSS_FUNCTION :( ")

# 无监督训练SIMcse的参数
ATTENTION_PROBS_DROPOUT_PROB = 0.1 # attention probabilities的dropout概率
HIDDEN_DROPOUT_PROB = 0.1 # embeddings的全联接层,encoder和池化层的dropout概率

# 训练eSIMcse的参数
DUP_RATE = 0.32 # 随机采样的最大重复比例
Q_SIZE = 150 # 决定了将多少负样本堆叠起来

# 训练consert的参数
CUTOFF_RATE=0.15 # 随机移除多少比例的tokens或者features



# 保存模型
MODEL_NAME = "{}-{}-{}-{}-{}-{}".format(LEARNING_METHOD, DATATYPE, BERT_MODEL, NUM_EPOCHS, LOSS_FUNCTION, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
if "qa-" in DATATYPE:
    MODEL_PATH = f"./outputs_QA/{MODEL_NAME}"
else:
    MODEL_PATH = f"./outputs/{MODEL_NAME}"

# EMBEDDING_SAVE_FILE_PATH = f"./embeddings/{MODEL_SAVE_FILENAME}"




















