"""
🚀 Enhanced Chunking Module - Cải tiến phương pháp tách đoạn cho RAG
Implements hybrid chunking strategy với performance optimization
"""

import hashlib
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import time


class ChunkingConfig:
    """Configuration class cho chunking strategies"""
    
    def __init__(self):
        # Hybrid Strategy Settings
        self.strategy = "hybrid"  # "semantic", "fixed", "hybrid"
        self.semantic_threshold = 0.8
        self.use_semantic_fallback = True
        
        # Fixed-size Settings (for hybrid/fallback)
        self.fixed_chunk_size = 1000
        self.fixed_overlap = 100
        
        # Semantic Settings
        self.semantic_buffer_size = 1
        self.semantic_threshold_type = "percentile"
        self.semantic_threshold_amount = 85  # Giảm từ 95 để nhanh hơn
        
        # General Settings
        self.min_chunk_size = 200
        self.max_chunk_size = 1500
        self.max_workers = 4
        
        # Caching Settings
        self.enable_cache = True
        self.cache_dir = ".chunking_cache"
        
        # Progress Settings
        self.show_progress = True


class ChunkingCache:
    """Smart caching system cho embeddings và chunks"""
    
    def __init__(self, cache_dir: str = ".chunking_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_path(self, content_hash: str, config_hash: str) -> str:
        """Get cache file path using both content and config hash"""
        combined_hash = content_hash + "_" + config_hash
        return os.path.join(self.cache_dir, f"{combined_hash}_chunks.pkl")
    
    def get_chunks(self, content_hash: str, config_hash: str) -> Optional[List[Document]]:
        """Lấy cached chunks nếu có"""
        cache_path = self._get_cache_path(content_hash, config_hash)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    chunks = pickle.load(f)
                # Validate chunks structure
                if isinstance(chunks, list) and all(isinstance(chunk, Document) for chunk in chunks):
                    return chunks
            except Exception as e:
                # Remove corrupted cache file
                try:
                    os.remove(cache_path)
                except:
                    pass
        return None
    
    def save_chunks(self, content_hash: str, config_hash: str, chunks: List[Document]):
        """Lưu chunks vào cache"""
        cache_path = self._get_cache_path(content_hash, config_hash)
        
        try:
            # Ensure directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Save with atomic write
            temp_path = cache_path + ".tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(chunks, f)
            
            # Atomic rename
            os.rename(temp_path, cache_path)
            
        except Exception as e:
            # Clean up temp file if exists
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
    
    def clear_cache(self):
        """Xóa toàn bộ cache"""
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.pkl') or file.endswith('.tmp'):
                    os.remove(os.path.join(self.cache_dir, file))
        except:
            pass
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
            total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in cache_files)
            
            return {
                "cache_files": len(cache_files),
                "total_size_mb": total_size / (1024 * 1024),
                "cache_dir": self.cache_dir
            }
        except:
            return {"cache_files": 0, "total_size_mb": 0, "cache_dir": self.cache_dir}


class EnhancedChunker:
    """Enhanced chunker với hybrid strategy và optimizations"""
    
    def __init__(self, embeddings: HuggingFaceEmbeddings, config: ChunkingConfig = None):
        self.embeddings = embeddings
        self.config = config or ChunkingConfig()
        self.cache = ChunkingCache(self.config.cache_dir) if self.config.enable_cache else None
        
        # Initialize splitters
        self._init_splitters()
        
    def _init_splitters(self):
        """Khởi tạo các text splitters"""
        # Fixed-size splitter (fast fallback)
        self.fixed_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.fixed_chunk_size,
            chunk_overlap=self.config.fixed_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Semantic splitter (high quality)
        try:
            self.semantic_splitter = SemanticChunker(
                embeddings=self.embeddings,
                buffer_size=self.config.semantic_buffer_size,
                breakpoint_threshold_type=self.config.semantic_threshold_type,
                breakpoint_threshold_amount=self.config.semantic_threshold_amount,
                min_chunk_size=self.config.min_chunk_size,
                add_start_index=True
            )
        except Exception as e:
            st.warning(f"⚠️ Không thể tạo SemanticChunker: {e}")
            self.semantic_splitter = None
    
    def _get_config_hash(self) -> str:
        """Tạo hash cho configuration để cache với proper normalization"""
        # Normalize values để cache stable hơn
        strategy = self.config.strategy
        chunk_size = self.config.fixed_chunk_size
        overlap = self.config.fixed_overlap
        semantic_threshold = self.config.semantic_threshold_amount
        
        # Round values để tránh minor differences
        chunk_size = round(chunk_size / 100) * 100  # Round to nearest 100
        overlap = round(overlap / 25) * 25  # Round to nearest 25
        semantic_threshold = round(semantic_threshold)
        
        config_str = f"{strategy}_{chunk_size}_{overlap}_{semantic_threshold}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _get_content_hash(self, content: str) -> str:
        """Tạo normalized content hash"""
        # Normalize content để stable cache
        normalized_content = content.strip().replace('\r\n', '\n').replace('\r', '\n')
        # Remove extra whitespace but preserve structure
        lines = []
        for line in normalized_content.split('\n'):
            stripped_line = line.strip()
            if stripped_line:  # Only add non-empty lines
                lines.append(stripped_line)
        normalized_content = '\n'.join(lines)
        return hashlib.md5(normalized_content.encode()).hexdigest()
    
    def _chunk_with_fixed_splitter(self, documents: List[Document]) -> List[Document]:
        """Chunking bằng fixed-size splitter (nhanh)"""
        chunks = []
        for doc in documents:
            doc_chunks = self.fixed_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
        return chunks
    
    def _chunk_with_semantic_splitter(self, documents: List[Document]) -> List[Document]:
        """Chunking bằng semantic splitter (chất lượng cao)"""
        if not self.semantic_splitter:
            return self._chunk_with_fixed_splitter(documents)
        
        try:
            return self.semantic_splitter.split_documents(documents)
        except Exception as e:
            st.warning(f"⚠️ Semantic chunking failed, fallback to fixed: {e}")
            return self._chunk_with_fixed_splitter(documents)
    
    def _apply_overlap_strategy(self, chunks: List[Document]) -> List[Document]:
        """Thêm overlap giữa các chunks để preserve context"""
        if len(chunks) <= 1:
            return chunks
        
        enhanced_chunks = []
        overlap_size = self.config.fixed_overlap
        
        for i, chunk in enumerate(chunks):
            content = chunk.page_content
            
            # Thêm context từ chunk trước
            if i > 0:
                prev_content = chunks[i-1].page_content
                prev_suffix = prev_content[-overlap_size:] if len(prev_content) > overlap_size else prev_content
                content = prev_suffix + "\n" + content
            
            # Thêm context từ chunk sau
            if i < len(chunks) - 1:
                next_content = chunks[i+1].page_content
                next_prefix = next_content[:overlap_size] if len(next_content) > overlap_size else next_content
                content = content + "\n" + next_prefix
            
            # Tạo chunk mới với metadata
            enhanced_chunk = Document(
                page_content=content,
                metadata={
                    **chunk.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "enhanced": True
                }
            )
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def _filter_chunks_by_size(self, chunks: List[Document]) -> List[Document]:
        """Lọc chunks theo kích thước min/max"""
        filtered_chunks = []
        
        for chunk in chunks:
            chunk_size = len(chunk.page_content)
            
            if chunk_size < self.config.min_chunk_size:
                # Merge với chunk trước nếu quá nhỏ
                if filtered_chunks:
                    last_chunk = filtered_chunks[-1]
                    merged_content = last_chunk.page_content + "\n" + chunk.page_content
                    if len(merged_content) <= self.config.max_chunk_size:
                        last_chunk.page_content = merged_content
                        continue
            
            elif chunk_size > self.config.max_chunk_size:
                # Chia nhỏ nếu quá lớn
                sub_chunks = self.fixed_splitter.split_documents([chunk])
                filtered_chunks.extend(sub_chunks)
                continue
            
            filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def chunk_documents(self, documents: List[Document]) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Main chunking method với hybrid strategy
        Returns: (chunks, metadata)
        """
        start_time = time.time()
        
        # Prepare content and config
        raw_content = "\n".join([doc.page_content for doc in documents])
        content_hash = self._get_content_hash(raw_content)
        config_hash = self._get_config_hash()
        
        # Debug info
        if self.config.show_progress:
            st.write(f"🔍 Content hash: {content_hash[:8]}...")
            st.write(f"⚙️ Config hash: {config_hash[:8]}...")
        
        # Check cache
        if self.cache:
            cached_chunks = self.cache.get_chunks(content_hash, config_hash)
            
            if cached_chunks:
                if self.config.show_progress:
                    st.success("✅ Cache HIT! Sử dụng cached chunks (instant)")
                
                return cached_chunks, {
                    "strategy": "cached",
                    "num_chunks": len(cached_chunks),
                    "processing_time": 0.1,
                    "cache_hit": True,
                    "content_hash": content_hash[:8],
                    "config_hash": config_hash[:8]
                }
            else:
                if self.config.show_progress:
                    st.info("💾 Cache MISS - Sẽ process và cache mới")
        
        # Progress tracking
        if self.config.show_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("🚀 Bắt đầu chunking...")
        
        chunks = []
        strategy_used = self.config.strategy
        
        try:
            # Step 1: Choose chunking strategy
            if self.config.show_progress:
                progress_bar.progress(20)
                status_text.text("📊 Phân tích strategy...")
            
            if self.config.strategy == "hybrid":
                # Thử semantic trước, fallback sang fixed nếu cần
                if self.semantic_splitter and len(documents) <= 5:  # Chỉ dùng semantic cho docs nhỏ
                    try:
                        if self.config.show_progress:
                            status_text.text("🧠 Đang sử dụng semantic chunking...")
                        chunks = self._chunk_with_semantic_splitter(documents)
                        strategy_used = "semantic"
                    except Exception as e:
                        if self.config.show_progress:
                            status_text.text("⚡ Fallback sang fixed chunking...")
                        chunks = self._chunk_with_fixed_splitter(documents)
                        strategy_used = "fixed_fallback"
                else:
                    if self.config.show_progress:
                        status_text.text("⚡ Đang sử dụng fixed chunking...")
                    chunks = self._chunk_with_fixed_splitter(documents)
                    strategy_used = "fixed"
            
            elif self.config.strategy == "semantic":
                if self.config.show_progress:
                    status_text.text("🧠 Đang sử dụng semantic chunking...")
                chunks = self._chunk_with_semantic_splitter(documents)
                strategy_used = "semantic"
            
            else:  # fixed
                if self.config.show_progress:
                    status_text.text("⚡ Đang sử dụng fixed chunking...")
                chunks = self._chunk_with_fixed_splitter(documents)
                strategy_used = "fixed"
            
            # Step 2: Apply enhancements
            if self.config.show_progress:
                progress_bar.progress(60)
                status_text.text("🔗 Đang thêm overlap context...")
            
            chunks = self._apply_overlap_strategy(chunks)
            
            # Step 3: Filter by size
            if self.config.show_progress:
                progress_bar.progress(80)
                status_text.text("📏 Đang tối ưu kích thước chunks...")
            
            chunks = self._filter_chunks_by_size(chunks)
            
            # Step 4: Cache results
            if self.cache:
                if self.config.show_progress:
                    progress_bar.progress(90)
                    status_text.text("💾 Đang lưu cache...")
                try:
                    self.cache.save_chunks(content_hash, config_hash, chunks)
                    if self.config.show_progress:
                        st.success("💾 Cache saved successfully!")
                except Exception as e:
                    if self.config.show_progress:
                        st.warning(f"⚠️ Cache save failed: {e}")
            
            # Complete
            processing_time = time.time() - start_time
            
            if self.config.show_progress:
                progress_bar.progress(100)
                status_text.text(f"✅ Hoàn thành! {len(chunks)} chunks trong {processing_time:.1f}s")
                time.sleep(1)  # Show completion message
                progress_bar.empty()
                status_text.empty()
            
            # Metadata
            metadata = {
                "strategy": strategy_used,
                "num_chunks": len(chunks),
                "processing_time": processing_time,
                "cache_hit": False,
                "content_hash": content_hash[:8],
                "config_hash": config_hash[:8],
                "config": {
                    "chunk_size": self.config.fixed_chunk_size,
                    "overlap": self.config.fixed_overlap,
                    "min_size": self.config.min_chunk_size,
                    "max_size": self.config.max_chunk_size
                }
            }
            
            return chunks, metadata
            
        except Exception as e:
            if self.config.show_progress:
                progress_bar.empty()
                status_text.empty()
            
            st.error(f"❌ Lỗi chunking: {str(e)}")
            
            # Emergency fallback
            chunks = self._chunk_with_fixed_splitter(documents)
            
            return chunks, {
                "strategy": "emergency_fallback",
                "num_chunks": len(chunks),
                "processing_time": time.time() - start_time,
                "error": str(e)
            }


# Utility functions để integrate với existing code

def create_enhanced_chunker(embeddings: HuggingFaceEmbeddings, 
                          strategy: str = "hybrid",
                          enable_cache: bool = True,
                          show_progress: bool = True) -> EnhancedChunker:
    """Factory function để tạo enhanced chunker"""
    config = ChunkingConfig()
    config.strategy = strategy
    config.enable_cache = enable_cache
    config.show_progress = show_progress
    
    return EnhancedChunker(embeddings, config)


def get_chunking_stats(metadata: Dict[str, Any]) -> str:
    """Format chunking statistics for display"""
    stats = []
    stats.append(f"📊 **Strategy**: {metadata.get('strategy', 'unknown')}")
    stats.append(f"📦 **Chunks**: {metadata.get('num_chunks', 0)}")
    stats.append(f"⏱️ **Time**: {metadata.get('processing_time', 0):.1f}s")
    
    if metadata.get('cache_hit'):
        stats.append("💾 **Cache**: Hit ✅")
    else:
        stats.append("💾 **Cache**: Miss ❌")
    
    # Add hash info for debugging
    if 'content_hash' in metadata:
        stats.append(f"🔍 **Content**: {metadata['content_hash']}")
    if 'config_hash' in metadata:
        stats.append(f"⚙️ **Config**: {metadata['config_hash']}")
    
    if 'config' in metadata:
        config = metadata['config']
        stats.append(f"📏 **Size**: {config.get('chunk_size', 0)} chars")
        stats.append(f"🔗 **Overlap**: {config.get('overlap', 0)} chars")
    
    return "\n".join(stats) 