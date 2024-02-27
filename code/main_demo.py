from models import *



if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

    # 1. 训练模型
    # model = train()
    # 1. 或者从outputs文件夹中读取模型
    model = SentenceTransformer(model_name_or_path="./outputs/sbert-sts-sup-macbert-2-CosineSimilarityLoss-2023-08-02_18-27-31")
    
    

    # 2. 创建faiss库，根据具体情况可以选择用txt_to_faiss还是qa_to_faiss
    # embeddings_dataset = txt_to_faiss(model, "test.txt")
    # embeddings_dataset = qa_to_faiss(model, "test.txt")
    # 2. 或者从txt文件夹中读取相关文件的faiss库
    embeddings_dataset = load_from_disk("/data/yf_center/Embeddings_demo/knowledge_base/qa/test/consert-snli-sup-macbert-2-CosineSimilarityLoss-2023-08-03_15-34-31")
    embeddings_dataset.load_faiss_index('embeddings', '/data/yf_center/Embeddings_demo/knowledge_base/qa/test/consert-snli-sup-macbert-2-CosineSimilarityLoss-2023-08-03_15-34-31/index.faiss')
    
    # 3. 从faiss库中找与query最相似的topn（=5）个问题
    query = "问题2：文本中提到的“物业管理区域”是什么意思？"
    topn_similar_sentences = search_similar_text(model, query, embeddings_dataset, topn=5, threshold=140)

