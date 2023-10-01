import os

import fasttext

from utils.pipeline_utils import grid_parameters, cv_param_2_df_column_title

def write_word2vec(df, word2vec_file_path):
    with open(word2vec_file_path, 'a') as f:
        word2vec_data = str(df.shape[0]) + " " + str(df.shape[1]) + "\n"
        for idx in df.index:
            line = idx + " "
            for col in df:
                line += str(df[col][idx])
                line += " "
            line = line.rstrip()
            line += "\n"
            word2vec_data += line
        f.write(word2vec_data)

def generate_model_file_path(embedding_root: str = None, embedding_for: str = None,
                             embedding_model='skipgram', embedding_wordNgrams: int = 1,
                             embedding_dim: int = 100, train_seq_len: int = 10, embedding_version: float = 1.0,
                             embedding_type='fasttext'):
    filename = f"{embedding_for}_{embedding_type}_{embedding_model}_" \
               f"{embedding_wordNgrams}wordNgram_{embedding_dim}dim_{train_seq_len}trainseq_v{embedding_version}.bin"
    return os.path.join(embedding_root, filename)

def generate_word2vec_model_file_path(embedding_root: str = None, embedding_for: str = None,
                             embedding_model='skipgram', embedding_wordNgrams: int = 1,
                             embedding_dim: int = 100, train_seq_len: int = 10, embedding_version: float = 1.0,
                             embedding_type='fasttext'):
    filename = f"{embedding_for}_{embedding_type}_{embedding_model}_" \
               f"{embedding_wordNgrams}wordNgram_{embedding_dim}dim_{train_seq_len}trainseq_v{embedding_version}.word2vec"
    return os.path.join(embedding_root, filename)

def fast_text_from_model_file(embedding_root: str = None, embedding_for: str = None,
                              embedding_model='skipgram', embedding_wordNgrams: int = 1,
                              embedding_dim: int = 100, train_seq_len: int = 10, embedding_version: float = 1.0):
    model_file_path = generate_model_file_path(embedding_root, embedding_for, embedding_model,
                                               embedding_wordNgrams, embedding_dim, train_seq_len,
                                               embedding_version, 'fasttext')
    model = fasttext.load_model(model_file_path)
    return model

def fast_text_from_word2vec_model_file(embedding_root: str = None, embedding_for: str = None,
                              embedding_model='skipgram', embedding_wordNgrams: int = 1,
                              embedding_dim: int = 100, train_seq_len: int = 10, embedding_version: float = 1.0):
    model_file_path = generate_word2vec_model_file_path(embedding_root, embedding_for, embedding_model,
                                               embedding_wordNgrams, embedding_dim, train_seq_len,
                                               embedding_version, 'fasttext')
    model = fasttext.load_model(model_file_path)
    return model

def outlier_event_template_reporter(df, df_labeled, cv_options):
    evt_template_series = df['event_template'].value_counts()

    # pretty print outlier messages
    tfidf_outliers = {}
    for cv_param in grid_parameters(cv_options):
        colname = cv_param_2_df_column_title(cv_param)
        outlier_templates = df_labeled[df_labeled[colname] == -1]['event_template'].tolist()
        if len(outlier_templates) > 0:
            tfidf_outliers[colname] = outlier_templates

    for key in tfidf_outliers.keys():
        modeltype, wordNgrams, vector_dim, seqlen, clustering_method, min_neigbhors = key.split("_")
        print(f"{clustering_method} clustering, Vector Dim={vector_dim}, TrainSeq={seqlen}, min neigbhors={min_neigbhors}     "
              f"{len(tfidf_outliers[key])} event templates marked as anomalies")
        # for item in tfidf_outliers[key]:
        #    print(f"\t{item}")
        print(evt_template_series.loc[tfidf_outliers[key]])
        print()
