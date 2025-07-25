import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from pythainlp.tokenize import word_tokenize, sent_tokenize
from pythainlp.util import normalize
import re
from collections import defaultdict
import streamlit as st
import hashlib


class ExperimentalThaiTextProcessor:
    """Handle Thai text preprocessing with multiple chunking methods"""

    @staticmethod
    def normalize_thai_text(text: str) -> str:
        """Normalize Thai text to handle vowel rendering issues"""
        # Remove zero-width characters
        text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')

        # Fix common Thai character encoding issues
        replacements = {
            'ํา': 'ำ',  # Fix sara am
            'เเ': 'แ',  # Fix repeated sara e
            '  ': ' ',  # Fix double spaces
            '\t': ' ',  # Replace tabs with spaces
            '\xa0': ' ',  # Replace non-breaking spaces
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text.strip()

    @staticmethod
    def extract_title(text: str) -> str:
        """Extract title from document text after 'เรื่อง'"""
        pattern = r'เรื่อง\s+([^\n-]+?)(?:[-]{3,}|\n|$)'
        match = re.search(pattern, text)

        if match:
            title = match.group(1).strip()
            title = re.sub(r'[-\s]+$', '', title)
            title = re.sub(r'\s+\d+\s*$', '', title)
            return title

        return "ไม่ระบุหัวข้อ"

    def chunk_by_whitespace(self, text: str, title: str, chunk_size: int = 300, overlap_ratio: float = 0.2) -> List[
        Dict]:
        """Chunk text by whitespace (simple splitting)"""
        chunks = []

        # Split by whitespace/newlines
        lines = text.split('\n')

        overlap_size = int(chunk_size * overlap_ratio)
        current_chunk = []
        current_length = 0

        for line in lines:
            line = line.strip()
            if not line:
                # Keep empty lines as separators
                if current_chunk:
                    current_chunk.append('')
                continue

            line_length = len(line)

            # If adding this line exceeds chunk size
            if current_length + line_length > chunk_size and current_chunk:
                # Create chunk
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'title': title,
                    'method': 'whitespace',
                    'chunk_size': chunk_size
                })

                # Handle overlap
                if overlap_size > 0:
                    overlap_lines = []
                    overlap_length = 0
                    for j in range(len(current_chunk) - 1, -1, -1):
                        line_len = len(current_chunk[j])
                        if overlap_length + line_len <= overlap_size:
                            overlap_lines.insert(0, current_chunk[j])
                            overlap_length += line_len
                        else:
                            break
                    current_chunk = overlap_lines
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(line)
            current_length += line_length

        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'title': title,
                'method': 'whitespace',
                'chunk_size': chunk_size
            })

        return chunks


class SimpleEmbedder:
    """Simple embedder using TF-IDF or hash-based embeddings as fallback"""

    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.vocab = {}
        self.idf = {}

    def _hash_text_to_vector(self, text: str) -> np.ndarray:
        """Create a deterministic vector from text using hashing"""
        # Create multiple hash values to fill the embedding dimension
        vector = np.zeros(self.embedding_dim)

        # Use different hash functions/seeds
        for i in range(self.embedding_dim):
            # Create a unique hash for each dimension
            hash_input = f"{i}:{text}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            # Convert to float between -1 and 1
            vector[i] = (hash_value % 10000) / 10000.0 * 2 - 1

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def encode(self, text: str, normalize_embeddings: bool = True, show_progress_bar: bool = False) -> np.ndarray:
        """Encode text to vector"""
        if isinstance(text, list):
            return np.array([self.encode(t, normalize_embeddings, show_progress_bar) for t in text])

        # Simple approach: use hash-based embeddings
        vector = self._hash_text_to_vector(text)

        if normalize_embeddings:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        return vector


class ExperimentalVectorDatabase:
    """Vector database for experimental evaluation with metadata support"""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.db_path = 'data/vector_db.json'
        self.model = self._load_model()
        self.data = []
        self.title_groups = defaultdict(list)
        self.source_to_metadata = {}
        self.keyword_index = defaultdict(list)
        self.department_index = defaultdict(list)
        self.type_index = defaultdict(list)

    def _load_model(self):
        """Load the embedding model"""
        try:
            # Try to load sentence transformers
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        except:
            # Fallback to simple embedder
            st.warning("Using simple embedder as sentence-transformers failed to load")
            return SimpleEmbedder()

    def clear_database(self):
        """Clear existing database"""
        self.data = []
        self.title_groups = defaultdict(list)
        self.source_to_metadata = {}
        self.keyword_index = defaultdict(list)
        self.department_index = defaultdict(list)
        self.type_index = defaultdict(list)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def embed_documents(self, documents: List[Dict]) -> None:
        """Embed documents with enhanced metadata"""
        progress_bar = st.progress(0)

        for idx, doc in enumerate(documents):
            # Update progress
            progress_bar.progress((idx + 1) / len(documents))

            # Create embeddings
            text_embedding = self.model.encode(doc['text'],
                                               normalize_embeddings=True,
                                               show_progress_bar=False)

            title_embedding = self.model.encode(doc['title'],
                                                normalize_embeddings=True,
                                                show_progress_bar=False)

            # Create keyword embedding
            keywords_text = ' '.join(doc['keywords'])
            keyword_embedding = self.model.encode(keywords_text,
                                                  normalize_embeddings=True,
                                                  show_progress_bar=False) if keywords_text else None

            # Create combined embedding with metadata
            metadata_text = f"{doc['title']} {doc['department']} {doc['type']} {keywords_text}"
            combined_text = f"{metadata_text} {doc['text']}"
            combined_embedding = self.model.encode(combined_text,
                                                   normalize_embeddings=True,
                                                   show_progress_bar=False)

            # Store document with embeddings and metadata
            doc_entry = {
                'id': doc['id'],
                'text_value': doc['text'],
                'text_embedding': text_embedding.tolist() if hasattr(text_embedding,
                                                                     'tolist') else text_embedding.tolist(),
                'title': doc['title'],
                'title_embedding': title_embedding.tolist() if hasattr(title_embedding,
                                                                       'tolist') else title_embedding.tolist(),
                'keyword_embedding': keyword_embedding.tolist() if keyword_embedding is not None and hasattr(
                    keyword_embedding, 'tolist') else None,
                'combined_embedding': combined_embedding.tolist() if hasattr(combined_embedding,
                                                                             'tolist') else combined_embedding.tolist(),
                'source': doc['source'],
                'department': doc['department'],
                'type': doc['type'],
                'date': doc['date'],
                'keywords': doc['keywords'],
                'doc_number': doc['doc_number'],
                'chunk_index': doc['chunk_index'],
                'total_chunks': doc['total_chunks'],
                'chunk_method': doc['chunk_method'],
                'chunk_size': doc['chunk_size']
            }

            self.data.append(doc_entry)
            index = len(self.data) - 1

            # Update indices
            self.title_groups[doc['title']].append(index)
            self.department_index[doc['department']].append(index)
            self.type_index[doc['type']].append(index)

            # Update keyword index
            for keyword in doc['keywords']:
                self.keyword_index[keyword.lower()].append(index)

            # Store source metadata mapping
            if doc['source'] not in self.source_to_metadata:
                self.source_to_metadata[doc['source']] = {
                    'title': doc['title'],
                    'department': doc['department'],
                    'type': doc['type'],
                    'keywords': doc['keywords'],
                    'date': doc['date']
                }

        progress_bar.empty()

    def save(self) -> None:
        """Save database with metadata indices"""
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)

        save_data = {
            'experiment_name': self.experiment_name,
            'documents': self.data,
            'title_groups': dict(self.title_groups),
            'source_to_metadata': self.source_to_metadata,
            'keyword_index': dict(self.keyword_index),
            'department_index': dict(self.department_index),
            'type_index': dict(self.type_index),
            'version': 'streamlit_1.0'
        }

        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """Load database from file"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r', encoding='utf-8') as f:
                save_data = json.load(f)

            self.experiment_name = save_data.get('experiment_name', 'streamlit_app')
            self.data = save_data.get('documents', [])
            self.title_groups = defaultdict(list, save_data.get('title_groups', {}))
            self.source_to_metadata = save_data.get('source_to_metadata', {})
            self.keyword_index = defaultdict(list, save_data.get('keyword_index', {}))
            self.department_index = defaultdict(list, save_data.get('department_index', {}))
            self.type_index = defaultdict(list, save_data.get('type_index', {}))

    def get_available_sources(self) -> List[str]:
        """Get list of available document sources"""
        return list(self.source_to_metadata.keys())

    def search_by_source_with_metadata(self, query: str, source: str, k: int = 8) -> List[Dict]:
        """Enhanced search within a specific source using metadata"""
        if source not in self.source_to_metadata:
            return []

        # Get metadata for this source
        source_metadata = self.source_to_metadata[source]
        title = source_metadata['title']
        source_indices = self.title_groups[title]

        if not source_indices:
            return []

        # Embed query
        query_embedding = self.model.encode(query,
                                            normalize_embeddings=True,
                                            show_progress_bar=False)

        # Check if query contains any keywords
        query_lower = query.lower()
        keyword_boost_indices = set()
        for keyword in source_metadata['keywords']:
            if keyword.lower() in query_lower:
                # Get all chunks that have this keyword
                keyword_boost_indices.update(self.keyword_index.get(keyword.lower(), []))

        # Calculate similarities with metadata boosting
        chunk_scores = []
        for idx in source_indices:
            doc = self.data[idx]

            # Use combined embedding for better matching
            combined_sim = cosine_similarity([query_embedding],
                                             [np.array(doc['combined_embedding'])])[0][0]

            text_sim = cosine_similarity([query_embedding],
                                         [np.array(doc['text_embedding'])])[0][0]

            # Check keyword similarity if available
            keyword_sim = 0.0
            if doc['keyword_embedding']:
                keyword_sim = cosine_similarity([query_embedding],
                                                [np.array(doc['keyword_embedding'])])[0][0]

            # Calculate weighted score
            base_score = 0.5 * combined_sim + 0.3 * text_sim + 0.2 * keyword_sim

            # Apply keyword boost if this chunk's document contains matching keywords
            if idx in keyword_boost_indices:
                base_score *= 1.2  # 20% boost for keyword matches

            chunk_scores.append({
                'idx': idx,
                'chunk_index': doc['chunk_index'],
                'score': float(base_score)
            })

        # Sort by score
        chunk_scores.sort(key=lambda x: x['score'], reverse=True)

        # Get top chunks and neighbors
        selected_indices = set()
        for i, item in enumerate(chunk_scores[:k // 2]):
            chunk_idx = item['chunk_index']
            selected_indices.add(item['idx'])

            # Add neighboring chunks
            for neighbor_idx in source_indices:
                neighbor_doc = self.data[neighbor_idx]
                neighbor_chunk_idx = neighbor_doc['chunk_index']

                if abs(neighbor_chunk_idx - chunk_idx) <= 1:
                    selected_indices.add(neighbor_idx)

        # Build results with full metadata
        results = []
        for idx in selected_indices:
            doc = self.data[idx]
            similarity = cosine_similarity([query_embedding],
                                           [np.array(doc['combined_embedding'])])[0][0]

            results.append({
                'id': doc['id'],
                'text': doc['text_value'],
                'title': doc['title'],
                'source': doc['source'],
                'department': doc['department'],
                'type': doc['type'],
                'keywords': doc['keywords'],
                'similarity': float(similarity),
                'chunk_index': doc['chunk_index'],
                'total_chunks': doc['total_chunks'],
                'date': doc['date'],
                'doc_number': doc['doc_number']
            })

        # Sort by chunk index
        results.sort(key=lambda x: x['chunk_index'])

        return results[:k]


# Keep the rest of the ExperimentalCAMTLlamaRAG class as is...
class ExperimentalCAMTLlamaRAG:
    """RAG system for experiments with metadata awareness"""

    def __init__(self, vector_db: ExperimentalVectorDatabase):
        self.vector_db = vector_db
        self.thai_processor = ExperimentalThaiTextProcessor()

    def create_enhanced_prompt(self, query: str, context: List[Dict], source: str) -> str:
        """Create enhanced prompt with metadata context"""
        # Sort context by chunk index
        context.sort(key=lambda x: x.get('chunk_index', 0))

        # Build context string
        context_parts = []
        for i, doc in enumerate(context):
            chunk_text = f"[ส่วนที่ {doc['chunk_index'] + 1}/{doc.get('total_chunks', '?')}]\n{doc['text']}"
            context_parts.append(chunk_text)

        context_str = "\n\n---\n\n".join(context_parts)

        # Get document metadata
        if context:
            first_doc = context[0]
            title = first_doc.get('title', 'ไม่ระบุ')
            department = first_doc.get('department', 'ไม่ระบุ')
            doc_type = first_doc.get('type', 'ไม่ระบุ')
            date = first_doc.get('date', 'ไม่ระบุ')
            keywords = first_doc.get('keywords', [])
            keywords_str = ', '.join(keywords) if keywords else 'ไม่ระบุ'
        else:
            title = department = doc_type = date = keywords_str = 'ไม่ระบุ'

        prompt = f"""คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับระเบียบและแนวปฏิบัติของวิทยาลัยศิลปะ สื่อ และเทคโนโลยี มหาวิทยาลัยเชียงใหม่

ข้อมูลเอกสาร:
- ชื่อเอกสาร: {title}
- แผนก: {department}
- ประเภท: {doc_type}
- วันที่: {date}
- คำสำคัญ: {keywords_str}
- ไฟล์: {source}

ข้อมูลจากเอกสาร:
{context_str}

คำถาม: {query}

กรุณาตอบคำถามโดย:
1. ใช้ข้อมูลจากเอกสารที่ให้มาเท่านั้น
2. หากเป็นคำถามเกี่ยวกับรายการหรือขั้นตอน ให้ตอบครบทุกข้อ
3. รวบรวมข้อมูลจากทุกส่วนที่เกี่ยวข้อง
4. ตอบเป็นภาษาไทย ชัดเจน กระชับ
5. หากไม่มีข้อมูล ให้ระบุว่า "ไม่พบข้อมูลในเอกสาร"

คำตอบ:"""

        return prompt

    def generate_answer(self, prompt: str) -> str:
        """Generate answer based on context (without LLM)"""
        # Extract context from prompt
        context_match = re.search(r'ข้อมูลจากเอกสาร:\n(.*?)\n\nคำถาม:', prompt, re.DOTALL)
        query_match = re.search(r'คำถาม: (.*?)\n\nกรุณาตอบคำถาม', prompt, re.DOTALL)

        if context_match and query_match:
            context = context_match.group(1)
            query = query_match.group(1)

            # Simple keyword-based answer generation
            query_lower = query.lower()
            context_lower = context.lower()

            # Extract numbers and amounts
            amounts = re.findall(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:บาท|เดือน|วัน|%)', context)

            # Common question patterns
            if any(word in query_lower for word in ['วงเงิน', 'จำนวน', 'เท่าไหร่', 'กี่บาท']):
                if amounts:
                    return f"จากข้อมูลในเอกสาร พบว่ามีการกล่าวถึงจำนวนเงิน {', '.join(amounts[:3])}"

            elif any(word in query_lower for word in ['วิธี', 'ขั้นตอน', 'อย่างไร']):
                # Extract steps or procedures
                steps = re.findall(r'(\d+\..*?)(?=\d+\.|$)', context, re.DOTALL)
                if steps:
                    return "ขั้นตอนที่พบในเอกสาร:\n" + "\n".join(steps[:5])

            elif any(word in query_lower for word in ['เงื่อนไข', 'คุณสมบัติ', 'หลักเกณฑ์']):
                # Extract conditions
                conditions = re.findall(r'[-•]\s*(.*?)(?=[-•]|\n\n|$)', context)
                if conditions:
                    return "เงื่อนไขที่พบในเอกสาร:\n" + "\n".join([f"- {c.strip()}" for c in conditions[:5]])

            # Default: return relevant sentences
            sentences = context.split('.')
            relevant_sentences = []
            for sent in sentences:
                if any(word in sent.lower() for word in query_lower.split()):
                    relevant_sentences.append(sent.strip())

            if relevant_sentences:
                return "จากเอกสารพบข้อมูลที่เกี่ยวข้อง: " + ". ".join(relevant_sentences[:3]) + "."
            else:
                return "ไม่พบข้อมูลที่ตรงกับคำถามในเอกสารที่ให้มา"

        return "ไม่สามารถประมวลผลคำถามได้"

    def ask_with_enhanced_retrieval(self, query: str, expected_source: str = None) -> Dict:
        """Enhanced RAG pipeline with metadata"""
        # Normalize query
        query = self.thai_processor.normalize_thai_text(query)

        # Search within the expected source with metadata enhancement
        if expected_source and expected_source in self.vector_db.source_to_metadata:
            relevant_chunks = self.vector_db.search_by_source_with_metadata(
                query, expected_source, k=8
            )
            selected_source = expected_source
        else:
            # Return empty if no source
            return {
                'query': query,
                'answer': "ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามของคุณในเอกสาร",
                'sources': [],
                'selected_source': None,
                'metadata': None
            }

        if not relevant_chunks:
            return {
                'query': query,
                'answer': "ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามของคุณในเอกสาร",
                'sources': [],
                'selected_source': None,
                'metadata': None
            }

        # Extract metadata from first chunk
        metadata = {
            'title': relevant_chunks[0].get('title'),
            'department': relevant_chunks[0].get('department'),
            'type': relevant_chunks[0].get('type'),
            'keywords': relevant_chunks[0].get('keywords', [])
        }

        # Create prompt
        prompt = self.create_enhanced_prompt(query, relevant_chunks, selected_source)

        # Generate answer
        answer = self.generate_answer(prompt)

        return {
            'query': query,
            'answer': answer,
            'sources': relevant_chunks,
            'selected_source': selected_source,
            'selected_title': relevant_chunks[0]['title'] if relevant_chunks else None,
            'metadata': metadata
        }