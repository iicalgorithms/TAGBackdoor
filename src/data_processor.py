import gc
import os
import numpy as np
import torch
import transformers
from torch import Tensor
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import logging
import vec2text
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer

from collections import Counter
from copy import deepcopy
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM

from torch_geometric.utils import to_undirected

from TAGLAS import get_dataset
from config import ExperimentConfig


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm import tqdm


class SentenceEncoder:

    def __init__(self, device, batch_size):
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Mean Pooling - Take attention mask into account for correct averaging


    def get_embeds(self, texts: List[str]) -> torch.Tensor:
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                      min=1e-9)
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Perform pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            batch_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)


            embeddings.append(batch_embeddings)
        return torch.cat(embeddings, dim=0).to(self.device)





class Vec2TextAutoencoder:

    # https://github.com/vec2text/vec2text

    # https://github.com/vec2text/vec2text
    def __init__(self, device='cpu', batch_size: int = 16):
        self.device = device
        self.batch_size = batch_size

        self.model=SentenceTransformer("sentence-transformers/gtr-t5-base")

        self.encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

        self.corrector = vec2text.load_pretrained_corrector("gtr-base")


    @torch.no_grad()
    def get_embeds(self, texts: List[str]):
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors='pt').to(self.device)

            with torch.no_grad():
                model_output = self.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                hidden_state = model_output.last_hidden_state
                batch_embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])

            # batch_embeddings =self.model.encode(batch_texts, convert_to_tensor=True)
            embeddings.extend(batch_embeddings)
        return torch.stack(embeddings)

    @torch.no_grad()
    def get_texts(self, embeddings: torch.FloatTensor) -> List[str]:
        texts = []

        for i in range(0, embeddings.size(0), self.batch_size):
            batch_embeddings = embeddings[i:i + self.batch_size]

            batch_texts = vec2text.invert_embeddings(
                embeddings=batch_embeddings.to(self.device),
                corrector=self.corrector,
                num_steps=20,
            )
            texts.extend(batch_texts)

        return texts

    @torch.no_grad()
    def get_texts2(self, embeddings):
        """Convert features back to text using decoder
            List of reconstructed texts
        """
        # Only for gtr-t5-base model
        # See https://github.com/jxmorris12/vec2text
        # https://github.com/jxmorris12/vec2text/issues/28


        inversion_model = vec2text.models.InversionModel.from_pretrained(
            "ielabgroup/vec2text_gtr-base-st_inversion"
        )
        correct_model = vec2text.models.CorrectorEncoderModel.from_pretrained(
            "ielabgroup/vec2text_gtr-base-st_corrector"
        )

        inversion_trainer = vec2text.trainers.InversionTrainer(
            model=inversion_model,
            train_dataset=None,
            eval_dataset=None,
            data_collator=transformers.DataCollatorForSeq2Seq(
                inversion_model.tokenizer,
                label_pad_token_id=-100,
            ),
        )

        correct_model.config.dispatch_batches = None
        corrector = vec2text.trainers.Corrector(
            model=correct_model,
            inversion_trainer=inversion_trainer,
            args=None,
            data_collator=vec2text.collator.DataCollatorForCorrection(
                tokenizer=inversion_trainer.model.tokenizer
            ),
        )


        # Batch processing for text reconstruction
        batch_size = 16
        res = []

        for i in range(0, len(embeddings), batch_size):
            batch_end = min(i + batch_size, len(embeddings))
            batch_embeddings = embeddings[i:batch_end]
            batch_reconstructed = vec2text.invert_embeddings(
                embeddings=batch_embeddings,
                corrector=corrector,
                num_steps=20,
            )
            res.extend(batch_reconstructed)

            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return res



class BottleneckT5TextAutoencoder:
    # https://colab.research.google.com/drive/1CF5Lr1bxoAFC_IPX5I0azu4X8UDz_zp-?usp=sharing#scrollTo=kOmfwesXEw5w
    def __init__(self, device='cpu', batch_size: int = 16):
        self.device = 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained("thesephist/contra-bottleneck-t5-xl-wikipedia")
        self.model = AutoModelForCausalLM.from_pretrained("thesephist/contra-bottleneck-t5-xl-wikipedia", trust_remote_code=True).to(self.device)
        self.model.eval()
        self.batch_size=batch_size

    @torch.no_grad()
    def get_embed(self, text: str) -> torch.FloatTensor:
        inputs = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
        decoder_inputs = self.tokenizer('', return_tensors='pt').to(self.device)
        return self.model(
            **inputs,
            decoder_input_ids=decoder_inputs['input_ids'],
            encode_only=True,
        )[0]

    @torch.no_grad()
    def get_text(self, latent: torch.FloatTensor) -> str:
        dummy_text = '.'
        dummy = self.get_embed(dummy_text)
        perturb_vector = latent - dummy
        self.model.perturb_vector = perturb_vector

        input_ids = self.tokenizer(dummy_text, return_tensors='pt').to(self.device).input_ids
        output = self.model.generate(
            input_ids=input_ids,
            max_length=512,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=False
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    @torch.no_grad()
    def get_embeds(self, texts: List[str]):
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                embed = self.get_embed(text)
                batch_embeddings.append(embed)

            embeddings.extend(batch_embeddings)
        
        # 连接所有批次的嵌入
        return torch.stack(embeddings)
    
    @torch.no_grad()
    def get_texts(self, latents: torch.FloatTensor) -> List[str]:
        texts = []
        
        for i in range(0, latents.size(0), self.batch_size):
            batch_latents = latents[i:i + self.batch_size]
            
            for latent in batch_latents:
                text = self.get_text(latent)
                texts.append(text)
        return texts


class SonarTextAutoencoder:
    def __init__(self, device, batch_size: int = 16, text_poison_mode: str="overwriting", max_seq_length_overwriting: int=1024, max_seq_length_appending: int=512):
        # https://github.com/facebookresearch/SONAR
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
        from fairseq2.generation import TopPSampler, TopKSampler

        self.device = device
        self.batch_size = batch_size
        self.max_seq_len=1024

        if text_poison_mode == "overwriting":
            self.max_seq_len=max_seq_length_overwriting
        else: #appending
            self.max_seq_len=max_seq_length_appending

        self.vec2text_model = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=torch.device(device),
            dtype=torch.float32,
        )

        self.text2vec_model = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder"
        )

    @torch.no_grad()
    def get_embeds(self, texts: List[str], progress_bar=False):
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.text2vec_model.predict(batch_texts, progress_bar=progress_bar, source_lang="eng_Latn")
            embeddings.extend(batch_embeddings)
        return torch.stack(embeddings).to(self.device)

    @torch.no_grad()
    def get_texts(self, latents: torch.FloatTensor) -> List[str]:
        texts = []
        for i in range(0, latents.size(0), self.batch_size):
            batch_latents = latents[i:i + self.batch_size]

            batch_texts=self.vec2text_model.predict(
                batch_latents,
                target_lang="eng_Latn",
                progress_bar=True,
                # sampler=TopPSampler(0.99),
                max_seq_len=self.max_seq_len
            )

            texts.extend(batch_texts)
        return texts


class BowAutoencoder:
    def __init__(self, device, batch_size, raw_texts):
        self.device = device
        self.batch_size = batch_size
        self.vocab_size = 1024

        self.vocabulary = {}
        self.reverse_vocabulary = {}
        self._build_vocabulary(raw_texts)

        
    def _build_vocabulary(self, texts: List[str]):
        # 简单的文本预处理和分词
        all_words = []
        for text in texts:
            # 转换为小写，移除标点符号，分词
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        # 统计词频并选择最常见的词
        word_counts = Counter(all_words)
        most_common_words = word_counts.most_common(self.vocab_size - 1)  # 保留一个位置给未知词
        
        # 构建词汇表
        self.vocabulary = {'<UNK>': 0}  # 未知词
        self.reverse_vocabulary = {0: '<UNK>'}
        
        for i, (word, _) in enumerate(most_common_words, 1):
            self.vocabulary[word] = i
            self.reverse_vocabulary[i] = word

        
    def _get_embed(self, text: str) -> torch.Tensor:
        """将单个文本转换为词袋向量"""
        # 文本预处理和分词
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 创建词袋向量
        bow_vector = torch.zeros(len(self.vocabulary), dtype=torch.float32)
        
        for word in words:
            word_id = self.vocabulary.get(word, 0)  # 未知词使用0
            bow_vector[word_id] += 1.0
            
        return bow_vector

    def _get_text(self, bow_vector) -> str:
            
        # 获取非零元素的索引和值
        bow_vector = bow_vector.cpu()
        words = []
        
        for word_id, count in enumerate(bow_vector):
            if count > 0 and word_id in self.reverse_vocabulary:
                word = self.reverse_vocabulary[word_id]
                if word != '<UNK>':
                    # 根据词频重复添加词（简单的重构策略）
                    words.extend([word] * int(count.item()))
                    
        # 随机打乱词序并连接成文本
        import random
        random.shuffle(words)
        return ' '.join(words) if words else '<UNK>'

    def get_embeds(self, texts: List[str]) -> torch.Tensor:
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                bow_vector = self._get_embed(text)
                batch_embeddings.append(bow_vector)
                
            # 堆叠批次嵌入
            if batch_embeddings:
                batch_tensor = torch.stack(batch_embeddings, dim=0)
                embeddings.append(batch_tensor)
                
        # 连接所有批次的嵌入
        return torch.cat(embeddings, dim=0).to(self.device)


    def get_texts(self, embeddings: torch.FloatTensor) -> List[str]:
        texts = []
        
        for i in range(0, embeddings.size(0), self.batch_size):
            batch_embeddings = embeddings[i:i + self.batch_size]
            
            for embedding in batch_embeddings:
                text = self._get_text(embedding)
                texts.append(text)
                
        return texts


class TFIDFAutoencoder:
    def __init__(self, device, batch_size, raw_texts):
        self.device = device
        self.batch_size = batch_size

        self.max_features = 1024
        self.vectorizer = None
        self.feature_names = None
        self._fit_vectorizer(raw_texts)


    def _preprocess_text(self, text: str) -> str:
        # 转换为小写，保留字母和数字
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _fit_vectorizer(self, raw_texts):
        processed_texts = [self._preprocess_text(text) for text in raw_texts]
        
        # 初始化TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',  # 移除英文停用词
            ngram_range=(1, 2),    # 使用1-gram和2-gram
            min_df=2,              # 忽略出现次数少于2的词
            max_df=0.90,           # 忽略出现在90%以上文档中的词
            sublinear_tf=True      # 使用次线性TF缩放
        )
        
        # 训练向量化器
        self.vectorizer.fit(processed_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()



    def _get_embed(self, text: str) -> torch.Tensor:
        # 预处理文本
        processed_text = self._preprocess_text(text)
        
        # 转换为TF-IDF向量
        tfidf_vector = self.vectorizer.transform([processed_text])
        
        # 转换为密集矩阵并创建torch张量
        dense_vector = tfidf_vector.toarray()[0]
        tensor = torch.tensor(dense_vector, dtype=torch.float32).to(self.device)
        
        return tensor

    def _get_text(self, tfidf_vector: torch.FloatTensor) -> str:
        # 将张量转换为numpy数组
        vector_np = tfidf_vector.cpu().numpy()
        
        # 获取非零元素的索引和值
        nonzero_indices = vector_np.nonzero()[0]
        
        if len(nonzero_indices) == 0:
            return "<EMPTY>"
            
        # 根据TF-IDF值排序，选择最重要的词
        sorted_indices = sorted(nonzero_indices, key=lambda i: vector_np[i], reverse=True)
        
        # 选择前N个最重要的词（可调整）
        top_n = min(20, len(sorted_indices))
        selected_indices = sorted_indices[:top_n]
        
        # 构建词汇列表
        words = []
        for idx in selected_indices:
            if idx < len(self.feature_names):
                feature = self.feature_names[idx]
                # 根据TF-IDF值决定词的重复次数
                weight = vector_np[idx]
                repeat_count = max(1, int(weight * 10))  # 缩放因子可调整
                
                # 处理n-gram特征
                if ' ' in feature:  # 2-gram
                    words.extend([feature] * repeat_count)
                else:  # 1-gram
                    words.extend([feature] * repeat_count)
        
        # 随机打乱并连接
        random.shuffle(words)
        
        # 清理和连接文本
        reconstructed_text = ' '.join(words)
        
        # 简单的后处理：移除重复的n-gram
        words_clean = []
        seen = set()
        for word in words:
            if word not in seen or len(seen) < 10:  # 允许少量重复
                words_clean.append(word)
                seen.add(word)
                
        return ' '.join(words_clean) if words_clean else "<EMPTY>"



    # def get_embeds(self, texts: List[str]):
    #     embeddings = []

    #     for i in range(0, len(texts), self.batch_size):
    #         batch_texts = texts[i:i + self.batch_size]
    #         batch_embeddings = []

    #         for text in batch_texts:
    #             vector = self._get_embed(text)
    #             batch_embeddings.append(vector)

    #         if batch_embeddings:
    #             batch_tensor = torch.stack(batch_embeddings, dim=0)
    #             embeddings.append(batch_tensor)

    #     return torch.cat(embeddings, dim=0).to(self.device)


    def get_embeds(self, texts: List[str]) -> torch.Tensor:
        all_tensors = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            processed_texts = [self._preprocess_text(text) for text in batch_texts]
            tfidf_vectors = self.vectorizer.transform(processed_texts)
            dense_vectors = tfidf_vectors.toarray()
            tensors = torch.tensor(dense_vectors, dtype=torch.float32).to(self.device)
            all_tensors.append(tensors)
        return torch.cat(all_tensors, dim=0)

    def get_texts(self, embeddings: torch.FloatTensor) -> List[str]:
        texts = []

        for i in range(0, embeddings.size(0), self.batch_size):
            batch_embeddings = embeddings[i:i + self.batch_size]

            for embedding in batch_embeddings:
                text = self._get_text(embedding)
                texts.append(text)

        return texts


class TAGDataProcessor:
    """Data processor for Text-Attributed Graph datasets."""
    
    def __init__(self, config: ExperimentConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.dataset=get_dataset(self.config.dataset_name, config.root)


        #这些数据集数据集的label处理上有区别
        label = list(self.dataset.label)
        if self.config.dataset_name in ["arxiv", "cora", "pubmed"]:
            for l in ["No", "Yes"]:
                if l in label:
                    label.remove(l)
        # if self.config.dataset_name == "products":
        #     for l in ["Buy a Kindle", "Furniture & decoration", "#508510"]:
        #         if l in label:
        #             label.remove(l)
        self.label = label


        # Determine batch size: use config value or auto-detect based on GPU memory
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if gpu_memory_gb > 40:  # High-end GPU
                batch_size = 64
            elif gpu_memory_gb > 20:  # Mid-range GPU
                batch_size = 32
            else:  # Low-end GPU
                batch_size = 16
        else:
            batch_size = 8  # CPU fallback

        self.batch_size=batch_size


        embedding_model_name = self.config.embedding_model_name.lower()


        if "bottleneck-t5" in embedding_model_name:
            self.text_autoencoder=BottleneckT5TextAutoencoder(self.config.device, batch_size)
        elif "sonar" in embedding_model_name:
            self.text_autoencoder=SonarTextAutoencoder(self.config.device, batch_size, self.config.text_poison_mode, self.config.max_seq_length_overwriting, self.config.max_seq_length_appending)
        elif "vec2text" in embedding_model_name:
            self.text_autoencoder=Vec2TextAutoencoder(self.config.device, batch_size)
        elif "tfidf" in embedding_model_name:
            self.text_autoencoder=TFIDFAutoencoder(self.config.device, batch_size, self.dataset.x)
        elif "bow" in embedding_model_name:
            self.text_autoencoder=BowAutoencoder(self.config.device, batch_size, self.dataset.x)
        elif "mini" in embedding_model_name:
            self.text_autoencoder=SentenceEncoder(self.config.device, batch_size)
        # else:
        #     self.text_autoencoder=None



    def _generate_text_embeddings(self) -> torch.Tensor:
        """Generate text embeddings using sentence transformer.
            
        Returns:
            Tensor of text embeddings
        """
        # Create embedding cache path based on text encoder and dataset
        encoder_name = self.config.embedding_model_name.replace('/', '_').replace('-', '_')
        embedding_cache_dir = os.path.join('dataset', self.config.dataset_name, 'embeddings')
        os.makedirs(embedding_cache_dir, exist_ok=True)
        embedding_cache_path = os.path.join(embedding_cache_dir, f"{encoder_name}.pt")

        # Load saved embeddings
        if os.path.exists(embedding_cache_path):
            self.logger.info(f"Loading embeddings from {embedding_cache_path}")

            # embeddings = torch.load(embedding_cache_path)
            embeddings = torch.load(embedding_cache_path, map_location='cpu')
            # embeddings = embeddings.to(device)

            self.logger.info(f" Embeddings shape: {embeddings.shape}")
            return embeddings


        # Generate embeddings with progress tracking and batch processing
        texts = self.dataset.x

        embeddings=self.text_autoencoder.get_embeds(texts)
        self.logger.info(f"Generated embeddings shape: {embeddings.shape}")

        torch.save(embeddings, embedding_cache_path)
        self.logger.info(f"Save text embeddings to {embedding_cache_path}")
        return embeddings

    
    def _create_inductive_splits(self) :
        """Create inductive splits for the dataset.
        labeled_ratio: Ratio of labeled nodes in training graph
        """
        train_ratio = self.config.train_ratio  # Ratio of labeled nodes in training graph
        val_ratio=self.config.val_ratio
        test_ratio=self.config.test_ratio
        unlabeled_ratio = 1 - train_ratio - val_ratio - test_ratio

        self.logger.info(f"Creating splits - train: {train_ratio:.2f}, val: {val_ratio:.2f}, test: {test_ratio:.2f}")

        num_nodes =len(self.dataset.x)  # Total number of nodes
        indices = self.dataset.node_map
        np.random.seed(self.config.random_seed) # Set random seed for reproducibility
        
        # Determine stratification
        stratify_labels = self.dataset.label_map

        # First split: separate test nodes
        train_unlabeled_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=self.config.random_seed,
            stratify=stratify_labels
        )
        # Second split: separate validation from training
        train_unlabeled_indices, val_indices = train_test_split(
            train_unlabeled_val_indices,
            test_size=val_ratio / (1 - test_ratio),
            random_state=self.config.random_seed,
            stratify=stratify_labels[train_unlabeled_val_indices]
        )

        train_indices, unlabeled_indices = train_test_split(
            train_unlabeled_indices,
            test_size=unlabeled_ratio/(1-test_ratio-val_ratio),
            random_state=self.config.random_seed,
            stratify=stratify_labels[train_unlabeled_indices]
        )

        # test_poison_indices, test_clean_indices=train_test_split(
        #     test_indices,
        #     test_size=0.5,
        #     random_state=self.config.random_seed,
        #     stratify=stratify_labels[test_indices]
        # )


        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        unlabeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
        except_test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # test_poison_mask = torch.zeros(num_nodes, dtype=torch.bool)
        # test_clean_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        unlabeled_mask[unlabeled_indices] = True
        except_test_mask[train_unlabeled_val_indices] = True

        #
        # test_poison_mask[test_poison_indices] = True
        # test_clean_mask[test_clean_indices] = True

        # Log split statistics
        self.logger.info(f" Created inductive splits for {len(indices)} nodes:")
        self.logger.info(f"  Training nodes: {train_mask.sum().item()}")
        self.logger.info(f"  Validation nodes: {val_mask.sum().item()}")
        self.logger.info(f"  Unlabeled nodes: {unlabeled_mask.sum().item()}")
        self.logger.info(f"  Test nodes: {test_mask.sum().item()}")
        # self.logger.info(f"   Test clean nodes: {test_clean_mask.sum().item()}")
        # self.logger.info(f"   Test poison nodes: {test_poison_mask.sum().item()}")
        
        return train_mask, val_mask, test_mask, unlabeled_mask, except_test_mask

    def run_process_dataset_pipeline(self):
        """
            This is the main entry point called by ExperimentRunner.
        Returns:
            Tuple of (processed_data, splits, text_encoder)
        """
        self.logger.info("Processing dataset pipeline...")

        if self.text_autoencoder:
            text_embeddings = self._generate_text_embeddings()
        else:
            text_embeddings = self.dataset.x_original # Use original features if no encoder is specified

        num_classes = len(self.label)
        num_nodes = len(self.dataset.x)
        num_edges = self.dataset.edge_index.shape[1]
        num_features = text_embeddings.shape[1]

        train_mask, val_mask, test_mask, unlabeled_mask, except_test_mask =self._create_inductive_splits() # Create data splits

        num_train_nodes=train_mask.sum().item()
        num_val_nodes=val_mask.sum().item()
        num_test_nodes=test_mask.sum().item()
        num_unlabeled_nodes = unlabeled_mask.sum().item()

        # Edge processing for inductive learning
        edge_index = to_undirected(self.dataset.edge_index)

        # all edges except test edges are training edges
        except_test_edge_mask = except_test_mask[edge_index[0]] & except_test_mask[edge_index[1]]
        except_test_edge_index = edge_index[:, except_test_edge_mask]

        test_edge_mask = test_mask[edge_index[0]] & test_mask[edge_index[1]]
        test_edge_index = edge_index[:, test_edge_mask]

        # Create PyG Data object
        data = Data(
            raw_texts=self.dataset.x,  # Original texts
            edge_index=edge_index,
            x=text_embeddings,
            y=self.dataset.label_map,   # label index to text map，在输出类别总数的时候需要用到，不用label
            label=self.label,  # text labels list 这个就是类别的文本列表，总长度为类别数,

            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            unlabeled_mask=unlabeled_mask,
            except_test_mask=except_test_mask,  # train+val+unlabeled
            # test_poison_mask=test_poison_mask,
            # test_clean_mask=test_clean_mask,

            train_edge_index=except_test_edge_index,  # Edges used for backdoor model training
            test_edge_index=test_edge_index,         # Edges used for backdoor testing

            num_classes=num_classes,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_features=num_features,
            num_train_nodes=num_train_nodes,
            num_val_nodes=num_val_nodes,
            num_test_nodes=num_test_nodes,
            num_unlabeled_nodes=num_unlabeled_nodes,
        )
        self.logger.info(f"Dataset: {self.dataset.name}: {data}.")

        gc.collect()

        # return data, self.encoder, self.tokenizer
        return data, self.text_autoencoder




if __name__ == '__main__':


    print(f"Running TAG Data Processor on GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'CPU')}")

    from config import ExperimentConfig

    datasets=["cora", "wikics", "pubmed", "arxiv"]
    for dataset in datasets:
        config = ExperimentConfig()
        config.root = "../dataset"
        config.embedding_model_name = "none"
        config.dataset_name = dataset
        data, text_autoencoder = TAGDataProcessor(config).run_process_dataset_pipeline()


        raw_texts = data.raw_texts
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        num_classes = data.num_classes
        avg_words = np.mean([len(text.split()) for text in raw_texts])
        max_words = max([len(text.split()) for text in raw_texts])

        print(f"Dataset: {dataset}")
        print(f"  Nodes: {num_nodes}")
        print(f"  Edges: {num_edges}")
        print(f"  Classes: {num_classes}")
        print(f"  Avg words per node: {avg_words:.2f}")
        print(f"  Max words : {max_words:.2f}")
        print("-"*20)

        save_path = os.path.join(config.root, dataset)
        os.makedirs(save_path, exist_ok=True)
        torch.save(data, os.path.join(save_path, f"{dataset}.pt"))






