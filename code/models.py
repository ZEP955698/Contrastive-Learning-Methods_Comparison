import logging
import os
import math
import torch
from torch.utils.data import DataLoader
import pandas as pd

from datasets import Dataset, load_from_disk

from config import *
from utils import *


from sentence_transformers import (
    SentenceTransformer, LoggingHandler,
    models, losses
    )

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.losses import TripletDistanceMetric
from sentence_transformers.ConSERT import ConSERT
from sentence_transformers.ESimCSE import ESimCSE

# 训练模型
def train():
    # 1.a 创建 word embedding model 和 pooling model
    if LEARNING_METHOD in ["simcse", "esimcse"]:
        word_embedding_model = models.Transformer(BERT_PATH, 
                                                  max_seq_length=MAX_SEQ_LENGTH)
        pooling_model = models.Pooling(word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode=POOLING_MODE,
                                       pooling_mode_cls_token=POOLING_MODE_CLS_TOKEN)
        
    # 1.b 创建 model
    if LEARNING_METHOD in ["simcse"]:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=DEVICE)
        if UNSUPERVISED:
            model[0].auto_model.attention_probs_dropout_prob = ATTENTION_PROBS_DROPOUT_PROB
            model[0].auto_model.hidden_dropout_prob = HIDDEN_DROPOUT_PROB
    elif LEARNING_METHOD in ["esimcse"]:
        moco_encoder = SentenceTransformer(BERT_PATH, device=DEVICE).to(DEVICE)
        moco_encoder.__setattr__("max_seq_length", MAX_SEQ_LENGTH)

        model = ESimCSE(modules=[word_embedding_model, pooling_model], device=DEVICE, dup_rate=DUP_RATE, q_size=Q_SIZE)
    elif LEARNING_METHOD in ["sbert"]:
        model = SentenceTransformer(BERT_PATH, device=DEVICE)
        model.__setattr__("max_seq_length", MAX_SEQ_LENGTH)
    elif LEARNING_METHOD in ["consert"]:
        model = ConSERT(BERT_PATH, device=DEVICE, cutoff_rate=CUTOFF_RATE, close_dropout=True)
        model.__setattr__("max_seq_length", MAX_SEQ_LENGTH)
    else:
        raise Exception(f"Unsupport LEARNING_METHOD: {LEARNING_METHOD}")
    
    # 2. 读取数据集
    train_samples = load_data(DATATYPE, TRAINDATA_PATH)
    dev_samples   = load_sts_sup(DEVDATA_PATH)
    test_samples  = load_sts_sup(TESTDATA_PATH)

    # 3. 初始化评估器
    dev_evaluator = init_evaluator(dev_samples, f'{DATATYPE}-dev')
    test_evaluator = init_evaluator(test_samples, f'{DATATYPE}-test')

    # 4. 训练前准备
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
    warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * WARMUP_RATIO)  # 10% of train data for warm-up
    evaluation_steps = int(len(train_dataloader) * EVALUATION_RATIO)  # Evaluate every 10% of the data

    print("Training sentences: {}".format(len(train_samples)))
    print("Warmup-steps: {}".format(warmup_steps))
    print("Performance before training")
    dev_evaluator(model)


    if LOSS_FUNCTION == "OnlineContrastiveLoss":
        train_loss = losses.OnlineContrastiveLoss(model)
    elif LOSS_FUNCTION == "CosineSimilarityLoss":
        train_loss = losses.CosineSimilarityLoss(model)
    elif LOSS_FUNCTION == "TripletLoss":
        train_loss = losses.TripletLoss(model, distance_metric=TripletDistanceMetric.COSINE, triplet_margin=TRIPLET_MARGIN)
    elif LOSS_FUNCTION == "MultipleNegativesRankingLoss":
        train_loss = losses.MultipleNegativesRankingLoss(model)
    elif LOSS_FUNCTION == "MultipleNegativesRankingLoss_embeddings":
        train_loss = losses.MultipleNegativesRankingLoss_embeddings(model)
    else:
        raise Exception("Unkown loss function :( ")


    training_params = {"train_objectives":[(train_dataloader, train_loss)],
                       "evaluator":dev_evaluator,
                       "epochs":NUM_EPOCHS,
                       "evaluation_steps":evaluation_steps,
                       "warmup_steps":warmup_steps,
                       "show_progress_bar":False,
                       "output_path":MODEL_PATH,
                       "optimizer_params":{'lr': LEARNING_RATE},
                       "use_amp":USE_AMP} 
    if LEARNING_METHOD == "esimcse":
        training_params["moco_encoder"] = moco_encoder
    
    
    model.fit(**training_params)
    test_evaluator(model)
    return model



def init_evaluator(samples, name=""):
    return EmbeddingSimilarityEvaluator.from_input_examples(samples, batch_size=BATCH_SIZE,
                                                            name=name,
                                                            main_similarity=SimilarityFunction.COSINE)


# 在dev/text上测试模型（评判标准：spearman系数）
def evaluate(evaluator, model):
    evaluator(model, MODEL_PATH)


# 给定一句话，使其变成embedding
def get_embeddings(model, sentences):
    return model.encode(sentences, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)


def df_to_faiss(model_name, df, text_col, parent_dir=None):
    if "-qa-" in model_name:
        model_name_or_path = os.path.join("./outputs_QA", model_name)
    else:
        model_name_or_path = os.path.join("./outputs", model_name)
    model = SentenceTransformer(model_name_or_path=model_name_or_path)


    dataset = Dataset.from_pandas(df)
    embeddings_dataset = dataset.map(
        lambda x: {"embeddings": get_embeddings(model, x[text_col])}
    )
    
    # 创建保存datasets和index的目录
    if (parent_dir is not None) and (not os.path.exists(parent_dir)):
        os.makedirs(parent_dir)
    
    # 保存datasets
    if parent_dir is not None:
        embeddings_dataset.save_to_disk(os.path.join(parent_dir, model_name))
    
    # 在datasets中添加faiss index，并保存
    embeddings_dataset.add_faiss_index(column="embeddings")
    if parent_dir is not None:
        embeddings_dataset.save_faiss_index('embeddings', os.path.join(parent_dir, MODEL_NAME, 'index.faiss'))
        # embeddings_dataset.drop_index("embeddings")
    return model, embeddings_dataset



# 给定一组csv文档，使用模型将文档导入faiss库
def csv_to_faiss(model_name, filename, text_col):

    parent_dir = "./knowledge_base/csv/"
    file_path = os.path.join(parent_dir, filename)

    df = pd.read_csv(file_path).dropna()
    # df = df.sample(frac=1, random_state=42)
    df =  df.iloc[:500, :]
    dataset_dir = os.path.join(parent_dir, filename[:filename.rindex(".")])
    return df_to_faiss(model_name, df, text_col, dataset_dir)




# 给定一组txt文档，使用模型将文档导入faiss库
def txt_to_faiss(model_name, filename):
    parent_dir = "./knowledge_base/txt/"
    file_path = os.path.join(parent_dir, filename)
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            if line=="\n" or len(line.strip())<2:
                continue
            sentences.append(line.strip())
    df = pd.DataFrame(sentences, columns=['sentences'])
    dataset_dir = os.path.join(parent_dir, filename[:filename.rindex(".")])
    return df_to_faiss(model_name, df, "sentences", dataset_dir)


# 给定一组qa文档，使用模型将文档导入faiss库
def qa_to_faiss(model_name, filename, qa_splitter="？"):
    parent_dir = "./knowledge_base/qa/"
    file_path = os.path.join(parent_dir, filename)
    questions = []
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            if line=="\n" or len(line.strip())<2:
                continue
            try:
                line = line.strip()
                # 使用问号？将问题和答案分开
                question_ind = line.rindex(qa_splitter)+1
                questions.append(line[:question_ind])
                sentences.append(line)
            except:
                continue

    df = pd.DataFrame({'questions':questions, 'sentences':sentences})
    dataset_dir = os.path.join(parent_dir, filename[:filename.rindex(".")])
    return df_to_faiss(model_name, df, "questions", dataset_dir)





# 给定句子，在faiss库中找到最相近的n个
def search_similar_text(model, query, embeddings_dataset, topn=5, threshold=0):
    query_embedding = get_embeddings(model, [query])
    scores, samples = embeddings_dataset.get_nearest_examples(
        "embeddings", query_embedding, k=topn
    )
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=True, inplace=True)

    for _, row in samples_df.iterrows():
        # 如果遇到score>threshold的情况就退出循环
        if threshold!=0 and row.scores >threshold:
            break 
        # print("=" * 50)
        # print(f"scores: {row.scores}")
        # print(f"scores: {row.sentences}")
        # print(f"Anchor: {row.Anchor}")
        # print(f"Positive: {row.Positive}")
    return samples_df
    # return samples["sentences"]













































