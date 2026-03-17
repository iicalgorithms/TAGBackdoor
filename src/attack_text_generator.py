import os
import json
import pickle
import re
import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from tqdm import tqdm
import nltk
import transformers
import shap
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


# root_path = os.path.abspath(os.sep)
# try:
#     nltk.data.find(root_path+'/home/liuyang/nltk_data/averaged_perceptron_tagger_eng')
# except LookupError:
# nltk.download('averaged_perceptron_tagger_eng')


from src.models import ModelTrainer
from config import ExperimentConfig

class AttackWordSelector:
    """Selector for attack words based on importance and target class pools."""
    def __init__(self, config: ExperimentConfig, use_test: str = ''):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize stopwords and excluded POS tags
        self.stopwords = set(stopwords.words('english'))
        self.use_test= use_test
    
    def _preprocess_text(self, text: str) -> str:
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)

        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            if (token not in self.stopwords and 
                token.isalpha() and 
                len(token) > 2):  # At least 3 characters
                filtered_tokens.append(token)
        
        return ' '.join(filtered_tokens)
    
    def _build_keywords_pools(self, data) -> Dict[int,List[str]]:
        """Build keyword pools for all target classes using default TF-IDF method

        Returns:
            Dict of keywords for each target class
        """
        self.logger.info(f"Building keywords pools")

        # Check cache first
        filename = f"{self.config.dataset_name}_{self.config.word_importance_class}.json"
        file_path = os.path.join("saved", "keywords_pool", filename)
        keywords_pools = {}

        if os.path.exists(file_path):
            self.logger.info(f"Loading cached keywords pools from {file_path}")
            with open(file_path, 'r') as f:
                keywords_pools = json.load(f)
                keywords_pools = {int(k): v for k, v in keywords_pools.items()}
            return keywords_pools


        if self.config.word_importance_class == "logistic-regression":
            # TODO: 训练线性文本分类器（如 Logistic Regression）
            # 1用文本向量（BoW、TF-IDF、平均词向量）训练一个线性分类器。
            # 2每个类别对应一个权重向量：权重越高，表示该词对该类别贡献越大。
            # 3对每个类别取权重排序即可得到其最重要的词。
            pass

        else:
            # (default)Build keywords pools using TF-IDF
            class_texts = defaultdict(list)
            for node_idx, label in enumerate(data.y.tolist()):
                text = data.raw_texts[node_idx]
                processed_text = self._preprocess_text(text)
                if processed_text:
                    class_texts[label].append(processed_text)

            # Build TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=5000, #最多只保留 5000 个特征词（按重要性排序）
                stop_words='english', #去除英文停用词（如 "the", "is" 等）
                ngram_range=(1, 2), #考虑单个词（unigram）和两个词的词组（bigram）。
                min_df=2, # 一个词至少在 2 个文档中出现才会被保留。
                max_df=0.8 #如果一个词在 80% 以上的文档中出现，则会被忽略（认为它没区分度）
            )

            # Combine all texts for fitting vectorizer
            all_texts = []
            for texts in class_texts.values():
                all_texts.extend(texts)

            # Fit vectorizer on all texts
            vectorizer.fit(all_texts)
            feature_names = vectorizer.get_feature_names_out()

            # For each class, compute average TF-IDF scores
            for class_label, texts in class_texts.items():
                if not texts:
                    keywords_pools[class_label] = []
                    continue

                # Transform texts to TF-IDF matrix
                tfidf_matrix = vectorizer.transform(texts)

                # Compute mean TF-IDF scores for this class
                mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)

                # Get top words sorted by importance
                word_scores = list(zip(feature_names, mean_scores))
                word_scores.sort(key=lambda x: x[1], reverse=True)

                # Extract top words (filter out very low scores)
                top_words = [word for word, score in word_scores if score > 0.01]
                keywords_pools[class_label] = top_words[:self.config.num_keywords_per_class]

                self.logger.info(f"Class {class_label}: {len(keywords_pools[class_label])} keywords")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(keywords_pools, f, indent=2)

        self.logger.info(f"Built keywords pools saved in {file_path}")
        return keywords_pools

    def _keywords_in_text(self, data, poison_candidates):
        """Find the most important keywords in each candidate text using default SHAP.

        Args:
            data: Graph data object
            poison_candidates: List of poison candidate information

        Returns:
            Updated poison candidates with important keywords
        """
        self.logger.info("Finding important keywords in candidate texts")

        updated_candidates = [] # need to add  keywords


        if self.config.word_importance_text == "shap":
            try:
                # Use a more efficient approach for SHAP
                model_name = self.config.shap_model #, 'distilbert-base-uncased'
                
                # Initialize tokenizer and model
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
                model = transformers.AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=len(torch.unique(data.y))
                )
                
                # Move model to appropriate device
                device = torch.device(self.config.device)
                model = model.to(device)
                model.eval()

                # Build prediction pipeline with batch processing
                def predict_batch(texts):
                    """Batch prediction for efficiency"""
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    inputs = tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probabilities = torch.softmax(outputs.logits, dim=-1)
                    
                    return probabilities.cpu().numpy()

                # Initialize SHAP explainer
                explainer = shap.Explainer(predict_batch, tokenizer)

                # Process candidates in batches for efficiency
                batch_size = 8
                for i in range(0, len(poison_candidates), batch_size):
                    batch_candidates = poison_candidates[i:i + batch_size]
                    batch_texts = []
                    
                    for candidate in batch_candidates:
                        node_idx = candidate['node_idx']
                        node_text = data.raw_texts[node_idx]
                        preprocessed_text = self._preprocess_text(node_text)
                        batch_texts.append(preprocessed_text)
                    
                    try:
                        # Get SHAP values for the batch
                        shap_values = explainer(batch_texts)
                        
                        for j, candidate in enumerate(batch_candidates):
                            tokens = batch_texts[j].split()
                            
                            # Extract word importance scores
                            word_importance = {}
                            if hasattr(shap_values, 'values') and len(shap_values.values) > j:
                                values = shap_values.values[j]
                                for k, word in enumerate(tokens):
                                    if k < len(values) and len(word) > 2 and word.isalpha():
                                        word_importance[word] = float(np.abs(values[k]).sum())
                            
                            # Sort by importance
                            important_keywords = sorted(
                                word_importance.items(),
                                key=lambda x: x[1],
                                reverse=True
                            )[:self.config.num_keywords_per_class]
                            
                            # Update candidate
                            updated_candidate = candidate.copy()
                            updated_candidate['keywords'] = important_keywords
                            updated_candidates.append(updated_candidate)
                            
                    except Exception as e:
                        self.logger.warning(f"SHAP processing failed for batch {i//batch_size + 1}: {str(e)}")
                        # Fallback for this batch
                        for candidate in batch_candidates:
                            updated_candidate = candidate.copy()
                            updated_candidate['keywords'] = []
                            updated_candidates.append(updated_candidate)
                            
            except Exception as e:
                self.logger.error(f"SHAP keyword extraction failed: {str(e)}")
                # Fallback to frequency-based method
                return self._fallback_keyword_extraction(data, poison_candidates)

        else:
            # Enhanced frequency-based importance with TF-IDF weighting
            return self._fallback_keyword_extraction(data, poison_candidates)

        self.logger.info(f"Processed {len(updated_candidates)} candidates with keyword extraction")
        return updated_candidates

    def _fallback_keyword_extraction(self, data, poison_candidates):
        """Fallback keyword extraction using enhanced frequency-based approach with TF-IDF weighting.
        
        Args:
            data: Graph data object
            poison_candidates: List of poison candidate information
            
        Returns:
            Updated poison candidates with important keywords
        """
        self.logger.info("Using fallback keyword extraction method")
        
        updated_candidates = []
        
        try:
            # Use TF-IDF for better keyword extraction
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 1),
                min_df=1,
                max_df=0.95
            )
            
            # Preprocess all texts
            all_texts = [self._preprocess_text(text) for text in data.raw_texts]
            
            # Fit TF-IDF on all texts
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            for candidate in poison_candidates:
                try:
                    node_idx = candidate['node_idx']
                    
                    # Get TF-IDF scores for this document
                    doc_tfidf = tfidf_matrix[node_idx].toarray()[0]
                    
                    # Create word-score pairs
                    word_scores = list(zip(feature_names, doc_tfidf))
                    
                    # Filter and sort by TF-IDF score
                    important_keywords = [
                        (word, score) for word, score in word_scores
                        if score > 0 and len(word) > 2 and word.isalpha()
                    ]
                    important_keywords.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take top keywords
                    top_keywords = important_keywords[:self.config.num_keywords_per_class]
                    
                    # Update candidate
                    updated_candidate = candidate.copy()
                    updated_candidate['keywords'] = top_keywords
                    updated_candidates.append(updated_candidate)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process candidate {candidate.get('node_idx', 'unknown')} in fallback: {str(e)}")
                    # Simple frequency fallback
                    node_idx = candidate['node_idx']
                    node_text = data.raw_texts[node_idx]
                    tokens = self._preprocess_text(node_text).split()
                    
                    # Filter tokens
                    filtered_tokens = [
                        token for token in tokens
                        if len(token) > 2 and token.isalpha() and token.lower() not in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'way'}
                    ]
                    
                    word_freq = Counter(filtered_tokens)
                    important_keywords = [(word, freq) for word, freq in word_freq.most_common(self.config.num_keywords_per_class)]
                    
                    updated_candidate = candidate.copy()
                    updated_candidate['keywords'] = important_keywords
                    updated_candidates.append(updated_candidate)
                    
        except Exception as e:
            self.logger.error(f"Fallback keyword extraction failed: {str(e)}")
            # Ultimate fallback - just copy candidates without keywords
            for candidate in poison_candidates:
                updated_candidate = candidate.copy()
                updated_candidate['keywords'] = []
                updated_candidates.append(updated_candidate)
        
        return updated_candidates

    def _create_replace_plans(self, data, poison_candidates, keywords_pools: Dict[int, List[str]]) -> List[Dict[str, Any]]:
        """Create replace plan for poison candidates.

        Returns:
            List of replace plans
        """

        output_path = os.path.join("saved", "replace_plans",
                                   f"{self.config.dataset_name}_{self.config.surrogate_model}_{self.config.top_k_words}{self.use_test}.json")

        if os.path.exists(output_path):
            self.logger.info(f"Loading cached replace plans from {output_path}")
            return json.load(open(output_path, 'r'))

        self.logger.info("Creating replace plans for poison candidates")

        replace_plans = []

        for candidate in poison_candidates:
            node_idx = candidate['node_idx']
            true_label = candidate['true_label']
            target_label = int(candidate['target_label'])
            original_text = data.raw_texts[node_idx]
            keywords = candidate['keywords']

            # Get keyword pools for true and target labels
            target_keywords_pool = keywords_pools[target_label]

            # Create replace plan based on important keywords in the text
            tokens = original_text.lower().split()
            replaces = []
            
            # Use the important keywords from the text for replacement
            for i, keyword in enumerate(keywords[:self.config.top_k_words]):
                if i >= len(target_keywords_pool):
                    break
                
                # Find positions of the keyword in tokens
                positions = [j for j, token in enumerate(tokens) if token == keyword.lower()]
                if not positions:
                    continue

                # Select the first occurrence for replacement
                pos = positions[0]
                replace_word = target_keywords_pool[i].lower()
                
                replaces.append({
                    'position': int(pos),
                    'original_word': keyword.lower(),
                    'replace_word': replace_word
                })
                
                # Update token
                tokens[pos] = replace_word
            
            poisoned_text = " ".join(tokens)
            
            replace_plan = {
                'node_idx': int(node_idx),
                'true_label': true_label,
                'target_label': target_label,
                'keywords': keywords,
                'original_text': original_text,
                'poisoned_text': poisoned_text,
                'num_replaces': len(replaces),
                'replaces': replaces,
            }
            replace_plans.append(replace_plan)

        # Save replace plans to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(replace_plans, f, indent=2)

        self.logger.info(f"Created {len(replace_plans)} replace plans saved to {output_path}")
        return replace_plans


    def run_attack_word_selection_pipeline(self, data, poison_candidates) -> List[Dict[str, Any]]:
        """Run the complete attack word selection pipeline.
        
        Args:
            data: Graph data object
            poison_candidates: List of poison candidate information
            
        Returns:
            List of replace plans for poison candidates
        """
        self.logger.info("Starting attack word selection pipeline")

        # 1. Build keyword pools for all target classes
        keywords_pools = self._build_keywords_pools(data)

        # 2. Find important keywords in each candidate text
        updated_candidates = self._keywords_in_text(data, poison_candidates)

        # 3. Create replace plans by mapping important keywords to target class keywords
        replace_plans = self._create_replace_plans(data, updated_candidates, keywords_pools)

        self.logger.info(f"Attack word selection pipeline completed")
        return replace_plans




class RestrictProcessor(LogitsProcessor):
    def __init__(self, tokenizer, non_target_tokens):
        self.tokenizer = tokenizer
        self.non_target_tokens = non_target_tokens
        all_specified_and_non_specified = set(non_target_tokens)
        self.stopwords = [i for i in range(tokenizer.vocab_size) if i not in all_specified_and_non_specified]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        scores[:, self.non_target_tokens] = -float('Inf')
        return scores


def load_bow_config(filename):
    cora_config = {
        'max_tokens': 512,
        'max_words': 300,
        'num_classes': 7,
        'category_names': ["Rule Learning", "Neural Networks", "Case Based", "Genetic Algorithms", "Theory",
                           "Reinforcement Learning", "Probabilistic Methods"]
    }

    citeseer_config = {
        'max_tokens': 512,
        'max_words': 300,
        'num_classes': 6,
        'category_names': ["Agents", "Machine Learning", "Information Retrieval", "Database",
                           "Human Computer Interaction", "Artificial Intelligence"]
    }

    pubmed_config = {
        'max_tokens': 550,
        'max_words': 400,
        'num_classes': 3,
        'category_names': ['Diabetes Mellitus Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
    }

    arxiv_config = {
        'max_tokens': 512,
        'max_words': 350,
        'num_classes': 40,
        'category_names': ['Artificial Intelligence', 'Computation and Language', 'Computational Complexity',
                           'Computational Engineering, Finance, and Science', 'Computational Geometry',
                           'Computer Science and Game Theory', 'Computer Vision and Pattern Recognition',
                           'Computers and Society', 'Cryptography and Security', 'Data Structures and Algorithms',
                           'Databases', 'Digital Libraries', 'Discrete Mathematics',
                           'Distributed, Parallel, and Cluster Computing', 'Emerging Technologies',
                           'Formal Languages and Automata Theory', 'General Literature', 'Graphics',
                           'Hardware Architecture', 'Human-Computer Interaction', 'Information Retrieval',
                           'Information Theory', 'Logic in Computer Science', 'Machine Learning',
                           'Mathematical Software', 'Multiagent Systems', 'Multimedia',
                           'Networking and Internet Architecture', 'Neural and Evolutionary Computing',
                           'Numerical Analysis', 'Operating Systems', 'Other Computer Science', 'Performance',
                           'Programming Languages', 'Robotics', 'Social and Information Networks',
                           'Software Engineering', 'Sound', 'Symbolic Computation', 'Systems and Control']
    }

    reddit_config = {
        'max_tokens': 550,
        'max_words': 400,
        'num_classes': 0,
        'category_names': []
    }

    if 'cora' in filename:
        return cora_config['max_tokens'], cora_config['max_words'], cora_config['num_classes'], cora_config[
            'category_names']
    elif 'citeseer' in filename:
        return citeseer_config['max_tokens'], citeseer_config['max_words'], citeseer_config['num_classes'], \
        citeseer_config['category_names']
    elif 'pubmed' in filename:
        return pubmed_config['max_tokens'], pubmed_config['max_words'], pubmed_config['num_classes'], pubmed_config[
            'category_names']
    elif 'arxiv' in filename:
        return arxiv_config['max_tokens'], arxiv_config['max_words'], arxiv_config['num_classes'], arxiv_config[
            'category_names']
    elif 'reddit' in filename:
        return reddit_config['max_tokens'], reddit_config['max_words'], reddit_config['num_classes'], reddit_config[
            'category_names']


def generate_attack_texts(data, dataset, filename, llm): # 生成攻击文本的主函数
    """生成攻击文本的主函数
    
    Args:
        data: 包含图数据的对象
        dataset: 数据集名称
        filename: 文件名
        llm: 使用的大语言模型类型
    
    Returns:
        raw_texts: 包含原始文本和生成攻击文本的列表
    """
    # 从文件名中提取目录名和文件前缀
    dir_name = filename.split(f'{dataset}')[0]  # 获取目录路径
    file = filename.split(f'{dataset}')[1].split(".pt")[0]  # 获取文件名（去除.pt扩展名）
    file = dataset + file  # 重新组合文件名

    def save_input(data, dataset):
        """保存输入数据，提取攻击节点的特征并分析词汇使用情况
        
        Args:
            data: 图数据对象
            dataset: 数据集名称
        """
        ori_node_num = data.y.shape[0]  # 获取原始节点数量
        features_attack = data.x.to_dense()[ori_node_num:, ]  # 提取攻击节点的特征（密集格式）
        vectorizer_path = os.path.join("./bow_cache/", f"{dataset}.pkl")  # 构建词袋模型文件路径

        # 加载预训练的词袋向量化器
        with open(vectorizer_path, 'rb') as f:
            vec =  pickle.load(f)  # 加载词袋模型
        words = vec.get_feature_names_out()  # 获取词汇表中的所有词汇

        used_words = []  # 存储每个攻击节点使用的词汇
        not_used_words = []  # 存储每个攻击节点未使用的词汇
        # 遍历每个攻击节点的特征向量
        for doc in features_attack:
            # 提取特征值为1的词汇（使用的词汇）
            used = [words[i] for i in range(len(words)) if doc[i] == 1]
            # 提取特征值为0的词汇（未使用的词汇）
            not_used = [words[i] for i in range(len(words)) if doc[i] == 0]
            used_words.append(used)  # 添加到使用词汇列表
            not_used_words.append(not_used)  # 添加到未使用词汇列表
        # 转换为numpy数组，便于保存
        used_words = np.array(used_words, dtype=object)
        not_used_words = np.array(not_used_words, dtype=object)
        # 创建保存目录（如果不存在）
        if not os.path.exists(f"{dir_name}raw"):
            os.makedirs(f"{dir_name}raw")
        # 保存使用和未使用的词汇到文件
        np.save(f"{dir_name}raw/{file}_used.npy", used_words)
        np.save(f"{dir_name}raw/{file}_not_used.npy", not_used_words)

    def clear_text(raw_text):
        """清理文本，移除标题和摘要标签以及格式字符
        
        Args:
            raw_text: 原始文本字符串
            
        Returns:
            清理后的文本字符串
        """
        # 使用正则表达式移除"title:"标签（忽略大小写）
        raw_text = re.sub(r"\btitle:\s*", "", raw_text, flags=re.IGNORECASE)
        # 移除单独的"title"词汇（忽略大小写）
        raw_text = re.sub(r"\btitle\b", "", raw_text, flags=re.IGNORECASE)
        # 移除"abstract:"标签（忽略大小写）
        raw_text = re.sub(r"\babstract:\s*", "", raw_text, flags=re.IGNORECASE)
        # 移除单独的"abstract"词汇（忽略大小写）
        raw_text = re.sub(r"\babstract\b", "", raw_text, flags=re.IGNORECASE)
        # 将换行符替换为空格
        raw_text = raw_text.replace("\n", " ")
        # 移除引号
        raw_text = raw_text.replace('"', "")
        return raw_text

    def extract_number(filename):
        """从文件名中提取数字，例如从'result_12.txt'中提取12
        
        Args:
            filename: 文件名字符串
            
        Returns:
            提取的数字，如果没有找到则返回0
        """
        match = re.search(r'(\d+)', filename)  # 使用正则表达式查找数字
        return int(match.group(1)) if match else 0  # 返回找到的数字或0

    def load_LLM_output(directory):
        """加载LLM生成的输出文件并提取内容
        
        Args:
            directory: 包含输出文件的目录路径
            
        Returns:
            extracted_content: 提取的文本内容列表
        """
        extracted_content = []  # 存储提取的内容
        if os.path.exists(directory):  # 检查目录是否存在
            # 筛选出所有.txt文件
            txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
            # 按文件名中的数字部分排序
            txt_files.sort(key=extract_number)

            # 遍历每个文本文件
            for i, filename in enumerate(txt_files):
                filepath = os.path.join(directory, filename)  # 构建完整文件路径
                # 读取文件内容
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()  # 读取整个文件内容
                    # 查找相关部分的开始和结束位置
                    start_index = content.lower().rfind("title")  # 查找最后一个"title"的位置
                    end_index = content.find("=========")  # 查找分隔符的位置
                    if start_index != -1:  # 如果找到了"title"
                        # 提取内容并去除多余的空白字符
                        section = content[start_index:end_index].strip()
                        if dataset != 'pubmed':  # 对于非pubmed数据集
                            section = clear_text(section)  # 清理文本
                        else:  # 对于pubmed数据集
                            section = section.replace("\n", " ")  # 只替换换行符
                        extracted_content.append(section)  # 添加到结果列表
                    else:  # 如果没有找到"title"
                        section = content[:end_index].strip()  # 提取分隔符之前的内容
                        if dataset != 'pubmed':  # 对于非pubmed数据集
                            section = clear_text(section)  # 清理文本
                        else:  # 对于pubmed数据集
                            section = section.replace("\n", " ")  # 只替换换行符
                        extracted_content.append(section)  # 添加到结果列表
                if i < 10:  # 调试：打印前10个文件的内容
                    print(filename, extracted_content[-1])
        return extracted_content

    def calculate_usage_rates(text, should_use_words, should_not_use_words):
        """计算文本中词汇的使用率
        
        Args:
            text: 要分析的文本
            should_use_words: 应该使用的词汇列表
            should_not_use_words: 不应该使用的词汇列表
            
        Returns:
            should_use_rate: 应该使用词汇的使用率
            should_not_use_rate: 不应该使用词汇的使用率
            non_use: 未使用的应该使用的词汇列表
        """
        text = text.lower().split()  # 将文本转换为小写并分割成词汇列表
        # 处理连字符分隔的词汇，将其拆分
        text_words = [subpart for part in text for subpart in part.split('-')]
        non_use = []  # 存储未使用的应该使用的词汇

        should_use_count = 0  # 应该使用且实际使用的词汇计数
        should_not_use_count = 0  # 不应该使用但实际使用的词汇计数

        # 检查应该使用的词汇
        for word in should_use_words:
            if word in text_words:  # 如果词汇在文本中
                should_use_count += 1  # 增加使用计数
            else:
                non_use.append(word)  # 添加到未使用列表
        # 检查不应该使用的词汇
        for word in should_not_use_words:
            if word in text_words:  # 如果不应该使用的词汇出现在文本中
                should_not_use_count += 1  # 增加错误使用计数

        # 计算应该使用词汇的使用率（百分比）
        should_use_rate = (should_use_count / len(should_use_words)) * 100 if len(should_use_words) > 0 else 0
        # 计算不应该使用词汇的错误使用率（百分比）
        should_not_use_rate = (should_not_use_count / len(should_not_use_words)) * 100 if len(
            should_not_use_words) > 0 else 0

        return should_use_rate, should_not_use_rate, non_use

    def generate_response_gpt(input_text, max_tokens):
        """使用GPT模型生成响应
        
        Args:
            input_text: 输入的消息列表
            max_tokens: 最大生成token数
            
        Returns:
            生成的文本内容
        """
        client = OpenAI(api_key="f")  # 创建OpenAI客户端（注意：这里的API密钥需要替换为真实的）
        # 调用GPT模型生成响应
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",  # 指定模型名称
            messages=input_text,  # 输入消息
            max_tokens=max_tokens  # 最大生成token数
        )

        return response.choices[0].message.content  # 返回生成的内容

    def generate_response_llama(input_text, max_tokens, model, tokenizer, logits_processor, terminators):
        """使用Llama模型生成响应
        
        Args:
            input_text: 输入的消息列表
            max_tokens: 最大生成token数
            model: Llama模型
            tokenizer: 分词器
            logits_processor: logits处理器
            terminators: 终止符列表
            
        Returns:
            生成的文本内容
        """
        # 应用聊天模板并转换为tensor
        input_ids = tokenizer.apply_chat_template(
            input_text,  # 输入消息
            add_generation_prompt=True,  # 添加生成提示
            return_tensors="pt"  # 返回PyTorch张量
        ).to(model.device)  # 移动到模型设备

        # 使用模型生成文本
        outputs = model.generate(
            input_ids,  # 输入token IDs
            max_new_tokens=max_tokens,  # 最大新生成token数
            eos_token_id=terminators,  # 结束符token ID
            pad_token_id=128001,  # 填充token ID
            do_sample=True,  # 启用采样
            temperature=0.6,  # 温度参数，控制随机性
            top_p=0.9,  # nucleus采样参数
            logits_processor=logits_processor  # logits处理器
        )
        # 提取新生成的部分（去除输入部分）
        response = outputs[0][input_ids.shape[-1]:]
        # 解码为文本
        text = tokenizer.decode(response, skip_special_tokens=True)

        return text

    def generate(dir_name, llm):
        """生成攻击文本的核心函数
        
        Args:
            dir_name: 目录名称
            llm: 使用的大语言模型类型
        """
        if 'llama' in llm:  # 如果使用llama模型
            model_path = "saved/model"  # 模型保存路径
            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                model_path,  # 模型路径
                torch_dtype=torch.bfloat16,  # 数据类型
                device_map="auto",  # 自动设备映射
                # attn_implementation="flash_attention_2"  # 注释掉的flash attention
            )
            # 设置分词器的填充token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
            # 设置终止符
            terminators = [
                tokenizer.eos_token_id,  # 结束符token ID
                tokenizer.convert_tokens_to_ids("<|eot_id|>")  # 对话结束token ID
            ]

        folder_path = f'{dir_name}raw'  # 原始数据文件夹路径
        file_pairs = {}  # 存储文件对的字典

        # 遍历文件夹中的所有文件
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)  # 构建完整文件路径
            if os.path.isfile(file_path):  # 如果是文件
                if file.endswith('not_used.npy'):  # 如果是未使用词汇文件
                    suffix = 'not_used'  # 设置后缀
                    prefix = file[:-len('not_used.npy') - 1]  # 提取文件前缀
                elif file.endswith('used.npy'):  # 如果是使用词汇文件
                    suffix = 'used'  # 设置后缀
                    prefix = file[:-len('used.npy') - 1]  # 提取文件前缀

                # 将文件按前缀分组
                if prefix not in file_pairs:
                    file_pairs[prefix] = {}
                file_pairs[prefix][suffix] = file_path

        # 处理每个文件对
        for prefix, files in file_pairs.items():
            # 创建输出目录
            if not os.path.exists(f'{dir_name}{llm}/{prefix}'):
                os.makedirs(f'{dir_name}{llm}/{prefix}')
            use_rates = []  # 存储使用率
            not_use_rates = []  # 存储未使用率
            word_counts = []  # 存储词汇计数
            not_used_file = files['not_used']  # 未使用词汇文件
            used_file = files['used']  # 使用词汇文件
            print(f"Processing File Pairs: {not_used_file} and {used_file}")
            # 加载词汇数据
            used_words = np.load(used_file, allow_pickle=True)
            not_used_words = np.load(not_used_file, allow_pickle=True)
            # 加载配置信息
            max_tokens, max_words, num_classes, category_names = load_bow_config(used_file)
            # 遍历每个词汇对
            for id, (used_word, not_used_word) in tqdm(enumerate(zip(used_words, not_used_words))):
                # 如果结果文件已存在，跳过
                if os.path.exists(f'{dir_name}{llm}/{prefix}/result_{id}.txt'):
                    continue
                max_rate = 0  # 最大使用率
                final_use_rate = 0  # 最终使用率
                final_not_use_rate = 0  # 最终未使用率
                final_word_count = 0  # 最终词汇计数
                Results = ''  # 结果字符串
                # 根据LLM类型构建不同的提示消息
                if 'topic' in llm:  # 如果包含主题信息
                    messages = [
                        {"role": "system",
                         "content": "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."},
                        {"role": "user",
                         "content": "There are " + f"{num_classes} types of paper, which are " + ", ".join(
                             category_names) + ".\n" + "Generate a title and an abstract for paper belongs to one of the given categories.\nEnsure the generated content explicitly contains the following words: " + ", ".join(
                             f"'{word}'" for word in
                             used_word) + ".\n" + "These words should appear as specified, without using synonyms, plural forms, or other variants.\n" + f"Length limit: {max_words} words." + "\nOutput the TITLE and ABSTRACT without explanation.\nTITLE:...\nABSTRACT:..."}
                    ]
                else:  # 普通模式
                    messages = [
                        {"role": "system",
                         "content": "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."},
                        {"role": "user",
                         "content": "Generate a title and an abstract for an academic article.\n" + "Ensure the generated content explicitly contains the following words: " + ", ".join(
                             f"'{word}'" for word in
                             used_word) + ".\n" + "These words should appear as specified, without using synonyms, plural forms, or other variants.\n" + f"Length limit: {max_words} words." + "\nOutput the TITLE and ABSTRACT without explanation.\nTITLE:...\nABSTRACT:..."}
                    ]
                # 第一轮：初始请求
                if 'llama' in llm:  # 如果使用llama模型
                    # 处理不应使用的词汇，包括大写形式
                    Cap = [word.capitalize() for word in not_used_word]
                    not_used_word = np.append(not_used_word, Cap)
                    # 获取不应使用词汇的token ID
                    not_used_tokens = [tokenizer.encode(word)[1] for word in not_used_word]
                    the_not_used_tokens = [tokenizer.encode(f"the {word}")[2] for word in not_used_word]
                    not_used_tokens.extend(the_not_used_tokens)
                    # 根据是否包含mask设置限制处理器
                    if 'mask' in llm:
                        custom_processor = RestrictProcessor(tokenizer, not_used_tokens)
                    else:  # 无限制
                        custom_processor = RestrictProcessor(tokenizer, [])
                    logits_processor = LogitsProcessorList([custom_processor])
                    # 生成响应
                    response = generate_response_llama(messages, max_tokens, model, tokenizer, logits_processor,
                                                       terminators)
                elif 'gpt' in llm:  # 如果使用GPT模型
                    response = generate_response_gpt(messages, max_tokens)
                # 计算使用率
                use_rate1, use_rate2, missing_words = calculate_usage_rates(response, used_word, not_used_word)
                print("Initial Use Rate: {:.2f}%".format(use_rate1))
                print("Initial Not Use Rate: {:.2f}%".format(use_rate2))
                messages.append({"role": "assistant", "content": response})
                # 如果使用率达到最大值，保存结果
                if use_rate1 >= max_rate:
                    max_rate = use_rate1
                    final_use_rate = use_rate1
                    final_not_use_rate = use_rate2
                    Results = response
                    Results += '\n\n====================================\n\n'
                    Results += "Should Use Rate: {:.2f}%\n".format(use_rate1)
                    Results += "Should Not Use Rate: {:.2f}%\n".format(use_rate2)
                    Results += f"Word Count: {len(response.split())}"
                    final_word_count = len(response.split())
                    # 保存结果到文件
                    with open(f'{dir_name}{llm}/{prefix}/result_{id}.txt', 'w') as f:
                        f.write(Results)

                # 第2-n轮：用户反馈和助手修正
                for _ in range(3):  # 最多进行3轮修正
                    # 构建反馈消息
                    feedback = f"You forgot to use " + ', '.join(f"'{word}'" for word in
                                                                 missing_words) + ".\n" + "Output the corrected TITLE and ABSTRACT without explanation.\nTITLE:...\nABSTRACT:..."
                    messages.append({"role": "user", "content": feedback})
                    # 生成修正后的响应
                    if 'llama' in llm:
                        response = generate_response_llama(messages, max_tokens, model, tokenizer,
                                                           logits_processor, terminators)
                    elif 'gpt' in llm:
                        response = generate_response_gpt(messages, max_tokens)
                    # 重新计算使用率
                    use_rate1, use_rate2, missing_words = calculate_usage_rates(response, used_word,
                                                                                not_used_word)
                    messages.append({"role": "assistant", "content": response})
                    # 如果使用率提高，更新最佳结果
                    if use_rate1 >= max_rate:
                        max_rate = use_rate1
                        final_use_rate = use_rate1
                        final_not_use_rate = use_rate2
                        Results = response
                        Results += '\n\n====================================\n\n'
                        Results += "Should Use Rate: {:.2f}%\n".format(use_rate1)
                        Results += "Should Not Use Rate: {:.2f}%\n".format(use_rate2)
                        Results += f"Word Count: {len(response.split())}"
                        final_word_count = len(response.split())
                        # 保存更新的结果
                        with open(f'{dir_name}{llm}/{prefix}/result_{id}.txt', 'w') as f:
                            f.write(Results)
                # 打印当前ID的处理结果
                print(f'Finish id {id}. Use rate is: {max_rate}. Word count is: {final_word_count}.')
                # 记录统计信息
                use_rates.append(final_use_rate)
                not_use_rates.append(final_not_use_rate)
                word_counts.append(final_word_count)
            # 打印平均统计信息
            print(f'{prefix} Avg Use Rate: {np.mean(use_rates)}')
            print(f'{prefix} Avg Not Use Rate: {np.mean(not_use_rates)}')
            print(f'{prefix} Avg Word Count: {np.mean(word_counts)}')

    raw_texts = data.raw_texts
    texts = []

    # 1. Save input to used.npy / not_used.npy
    save_input(data, dataset)
    # Save to dir_name/raw/xxx_used.npy, dir_name/raw/xxx_not_used.npy

    # 2. Use LLM to generate raw text
    generate(dir_name, llm)
    # Save to dir_name/llm/file

    # 3. Load results
    texts = load_LLM_output(f"{dir_name}{llm}/{file}")
    if len(texts) > 0:
        if dataset.lower() == 'cora':
            assert len(texts) == 60, "Missing content"
        elif dataset.lower() == 'citeseer':
            assert len(texts) == 90, "Missing content"
        elif dataset.lower() == 'pubmed':
            assert len(texts) == 400, "Missing content"
    raw_texts.extend(texts)

    return raw_texts





