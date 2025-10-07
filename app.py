import streamlit as st
import torch
import tempfile
import os
import gc
from datetime import datetime

# LangChain and HuggingFace imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Transformers imports
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# Enhanced chunking imports
from enhanced_chunking import (
    create_enhanced_chunker, 
    get_chunking_stats, 
    ChunkingConfig,
    EnhancedChunker,
    ChunkingCache
)

# =================================================================
# 1. CẤU HÌNH TRANG VÀ SESSION STATE
# =================================================================

st.set_page_config(
    page_title="🤖 AI RAG Assistant - Enhanced", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Khởi tạo session state
for key in ["rag_chain", "models_loaded", "embeddings", "llm", "messages", "chat_history", "current_chat_id", "chunking_settings"]:
    if key not in st.session_state:
        if key == "messages":
            st.session_state[key] = []
        elif key == "chat_history":
            st.session_state[key] = []
        elif key == "current_chat_id":
            st.session_state[key] = None  # None = chat mới chưa lưu
        elif key == "chunking_settings":
            st.session_state[key] = {
                "strategy": "hybrid",
                "chunk_size": 1000,
                "overlap": 100,
                "enable_cache": True,
                "show_progress": True
            }
        else:
            st.session_state[key] = None

# =================================================================
# 2. CÁC HÀM TẢI MODEL (TỐI ỮU CHO COLAB GPU T4)
# =================================================================

@st.cache_resource
def load_embeddings():
    """Tải model embedding được tối ưu cho tiếng Việt."""
    model_name = "bkai-foundation-models/vietnamese-bi-encoder"
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

@st.cache_resource
def load_llm():
    """Tải Large Language Model (Vicuna 7B) với quantization để tiết kiệm tài nguyên."""
    # Cấu hình quantization tối ưu cho T4
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    MODEL_NAME = "lmsys/vicuna-7b-v1.5"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=nf4_config,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # Đảm bảo có pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id,
            device_map="auto",
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

        # Dọn dẹp bộ nhớ
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return HuggingFacePipeline(pipeline=model_pipeline)
    
    except Exception as e:
        return None

# =================================================================
# 3. HÀM XỬ LÝ PDF VÀ TẠO RAG CHAIN (ENHANCED VERSION)
# =================================================================

def process_pdf(uploaded_file):
    """Xử lý file PDF được tải lên và tạo RAG chain với Enhanced Chunking."""
    
    try:
        # Tạo file tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load tài liệu
        st.info("📄 Đang đọc file PDF...")
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Lấy settings từ session state
        settings = st.session_state.get("chunking_settings", {
            "strategy": "hybrid",
            "chunk_size": 1000,
            "overlap": 100,
            "enable_cache": True,
            "show_progress": True
        })

        # Tạo custom config cho enhanced chunker
        config = ChunkingConfig()
        config.strategy = settings["strategy"]
        config.fixed_chunk_size = settings["chunk_size"]
        config.fixed_overlap = settings["overlap"]
        config.enable_cache = settings["enable_cache"]
        config.show_progress = settings["show_progress"]

        # Tạo enhanced chunker
        enhanced_chunker = EnhancedChunker(st.session_state.embeddings, config)

        # Chunking với metadata
        st.info("✂️ Đang chia nhỏ tài liệu với Enhanced Chunking...")
        docs, chunking_metadata = enhanced_chunker.chunk_documents(documents)

        # Hiển thị thống kê chunking
        with st.expander("📊 Chunking Statistics", expanded=True):
            st.markdown(get_chunking_stats(chunking_metadata))
            
            # Thêm thông tin chi tiết
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📦 Total Chunks", chunking_metadata.get('num_chunks', 0))
            with col2:
                st.metric("⏱️ Processing Time", f"{chunking_metadata.get('processing_time', 0):.1f}s")
            with col3:
                cache_status = "✅ Hit" if chunking_metadata.get('cache_hit') else "❌ Miss"
                st.metric("💾 Cache", cache_status)

        st.info("🔍 Đang tạo cơ sở dữ liệu vector...")
        
        # Tạo Vector Database
        vector_db = Chroma.from_documents(documents=docs, embedding=st.session_state.embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})

        # Tạo prompt template tối ưu cho tiếng Việt
        RAG_PROMPT_TEMPLATE = """Bạn là một trợ lý AI thông minh. Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Hướng dẫn:
- Trả lời bằng tiếng Việt một cách rõ ràng và chính xác
- Chỉ sử dụng thông tin từ ngữ cảnh được cung cấp
- Nếu không tìm thấy thông tin, hãy nói rằng bạn không có đủ thông tin

Trả lời:"""

        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | st.session_state.llm
            | StrOutputParser()
            | (lambda text: text.split("Trả lời:")[-1].strip() if "Trả lời:" in text else text.strip())
        )

        # Dọn dẹp
        os.unlink(tmp_file_path)
        del loader, documents, vector_db
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return rag_chain, len(docs), chunking_metadata
        
    except Exception as e:
        st.error(f"❌ Lỗi xử lý PDF: {str(e)}")
        return None, 0, None

# =================================================================
# 4. HÀM QUẢN LÝ LỊCH SỬ CHAT (UNCHANGED)
# =================================================================

def auto_save_current_chat():
    """Tự động lưu chat hiện tại (ghi đè nếu đã có, tạo mới nếu chưa)."""
    try:
        if st.session_state.messages and len(st.session_state.messages) > 0:
            # Tạo title dựa trên câu hỏi đầu tiên
            first_question = st.session_state.messages[0]["content"][:30] + "..." if len(st.session_state.messages[0]["content"]) > 30 else st.session_state.messages[0]["content"]
            
            chat_session = {
                "timestamp": datetime.now().strftime("%H:%M %d/%m"),
                "messages": st.session_state.messages.copy(),
                "title": first_question
            }
            
            if (st.session_state.current_chat_id is not None and 
                0 <= st.session_state.current_chat_id < len(st.session_state.chat_history)):
                # Ghi đè chat đã có
                st.session_state.chat_history[st.session_state.current_chat_id] = chat_session
            else:
                # Tạo chat mới
                st.session_state.chat_history.append(chat_session)
                st.session_state.current_chat_id = len(st.session_state.chat_history) - 1
            return True
    except Exception as e:
        # Nếu có lỗi, vẫn tạo chat mới
        try:
            if st.session_state.messages and len(st.session_state.messages) > 0:
                first_question = st.session_state.messages[0]["content"][:30] + "..."
                chat_session = {
                    "timestamp": datetime.now().strftime("%H:%M %d/%m"),
                    "messages": st.session_state.messages.copy(),
                    "title": first_question
                }
                st.session_state.chat_history.append(chat_session)
                st.session_state.current_chat_id = len(st.session_state.chat_history) - 1
                return True
        except:
            pass
    return False

def create_new_chat():
    """Tạo chat mới và lưu chat hiện tại."""
    # Lưu chat hiện tại nếu có
    auto_save_current_chat()
    
    # Tạo chat mới
    st.session_state.messages = []
    st.session_state.current_chat_id = None  # Chat mới chưa lưu
    return True

def load_chat_from_history(index):
    """Tải cuộc trò chuyện từ lịch sử."""
    try:
        if 0 <= index < len(st.session_state.chat_history):
            # Lưu chat hiện tại trước khi chuyển
            auto_save_current_chat()
            
            # Load chat được chọn
            st.session_state.messages = st.session_state.chat_history[index]["messages"].copy()
            st.session_state.current_chat_id = index
            return True
    except (IndexError, KeyError):
        # Nếu có lỗi, reset về chat mới
        st.session_state.messages = []
        st.session_state.current_chat_id = None
    return False

def delete_chat_from_history(index):
    """Xóa một chat khỏi lịch sử."""
    try:
        if 0 <= index < len(st.session_state.chat_history):
            # Lưu current_chat_id cũ để so sánh
            old_current_id = st.session_state.current_chat_id
            
            # Xóa chat khỏi lịch sử trước
            st.session_state.chat_history.pop(index)
            
            # Cập nhật current_chat_id sau khi xóa
            if old_current_id == index:
                # Nếu xóa chat đang active, chuyển về chat mới
                st.session_state.messages = []
                st.session_state.current_chat_id = None
            elif old_current_id is not None and old_current_id > index:
                # Nếu xóa chat có index nhỏ hơn current_chat_id, giảm current_chat_id đi 1
                st.session_state.current_chat_id = old_current_id - 1
            
            # Đảm bảo current_chat_id không vượt quá số lượng chat còn lại
            if (st.session_state.current_chat_id is not None and 
                st.session_state.current_chat_id >= len(st.session_state.chat_history)):
                st.session_state.current_chat_id = None
                st.session_state.messages = []
            
            return True
    except Exception as e:
        # Xử lý lỗi và reset về trạng thái an toàn
        st.session_state.current_chat_id = None
        st.session_state.messages = []
        return False
    return False

def clear_chat_history():
    """Xóa toàn bộ lịch sử chat."""
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.session_state.current_chat_id = None
    st.session_state['refresh_needed'] = True

# =================================================================
# 5. GIAO DIỆN NGƯỜI DÙNG (ENHANCED VERSION)
# =================================================================

# Header chính
st.title("🚀 Ứng dụng RAG Enhanced - Hỏi đáp tài liệu PDF")
st.markdown("*Trò chuyện thông minh với tài liệu của bạn bằng tiếng Việt - Với Enhanced Chunking*")

# === SIDEBAR ===
with st.sidebar:
    st.header("📂 Quản lý tài liệu")
    
    # Upload file
    uploaded_file = st.file_uploader(
        "Chọn file PDF", 
        type="pdf",
        help="Hỗ trợ file PDF có kích thước tối đa 200MB"
    )
    
    # === ENHANCED CHUNKING SETTINGS ===
    st.markdown("---")
    st.subheader("⚙️ Enhanced Chunking Settings")

    # Chunking strategy selection
    strategy = st.selectbox(
        "📊 Strategy",
        ["hybrid", "semantic", "fixed"],
        index=0,
        help="• Hybrid: Balance tốc độ vs chất lượng\n• Semantic: Chất lượng cao (chậm)\n• Fixed: Tốc độ cao (nhanh)",
        key="chunking_strategy"
    )

    # Advanced settings trong expander
    with st.expander("🔧 Advanced Settings"):
        chunk_size = st.slider("📏 Chunk Size", 500, 2000, 1000, 100, 
                              help="Kích thước chunk (characters)")
        overlap = st.slider("🔗 Overlap", 0, 300, 100, 50,
                          help="Overlap giữa chunks để preserve context")
        enable_cache = st.checkbox("💾 Enable Cache", True,
                                 help="Cache chunks để tăng tốc lần sau")
        show_progress = st.checkbox("📊 Show Progress", True,
                                  help="Hiển thị progress bars")
        
        # Clear cache button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Cache", help="Xóa toàn bộ chunking cache"):
                try:
                    cache = ChunkingCache()
                    cache.clear_cache()
                    st.success("Cache cleared!")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col2:
            # Show cache info
            if st.button("ℹ️ Cache Info"):
                try:
                    cache = ChunkingCache()
                    cache_info = cache.get_cache_info()
                    st.info(f"📦 {cache_info['cache_files']} cached documents\n💾 {cache_info['total_size_mb']:.1f} MB used")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Lưu settings vào session state
    st.session_state.chunking_settings = {
        "strategy": strategy,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "enable_cache": enable_cache,
        "show_progress": show_progress
    }

    # Process PDF button
    if uploaded_file and st.session_state.models_loaded:
        if st.button("🚀 Xử lý PDF với Enhanced Chunking", type="primary"):
            if st.session_state.embeddings and st.session_state.llm:
                # Lưu chat hiện tại trước khi bắt đầu chat mới với tài liệu mới
                auto_save_current_chat()
                
                # Reset về chat mới
                st.session_state.messages = []
                st.session_state.current_chat_id = None
                
                rag_chain, num_chunks, chunking_metadata = process_pdf(uploaded_file)
                if rag_chain and num_chunks > 0:
                    st.session_state.rag_chain = rag_chain
                    
                    # Enhanced success message with stats
                    strategy_used = chunking_metadata.get('strategy', 'unknown')
                    processing_time = chunking_metadata.get('processing_time', 0)
                    cache_hit = chunking_metadata.get('cache_hit', False)
                    
                    success_msg = f"✅ Thành công! {num_chunks} chunks ({strategy_used})"
                    if cache_hit:
                        success_msg += " - Cache Hit! ⚡"
                    else:
                        success_msg += f" trong {processing_time:.1f}s"
                    
                    st.success(success_msg)
                    st.info("💬 Bây giờ bạn có thể bắt đầu trò chuyện!")
                else:
                    st.error("❌ Không thể xử lý file PDF. Vui lòng thử file khác.")
            else:
                st.error("❌ Models chưa sẵn sàng!")
    elif uploaded_file and not st.session_state.models_loaded:
        st.warning("⏳ Vui lòng chờ hệ thống tải models xong.")

    st.markdown("---")
    
    # Quản lý chat - chỉ hiển thị khi có RAG chain
    if st.session_state.rag_chain:
        st.header("💬 Quản lý cuộc trò chuyện")
        
        # Hiển thị trạng thái chat hiện tại
        try:
            if (st.session_state.current_chat_id is not None and 
                0 <= st.session_state.current_chat_id < len(st.session_state.chat_history)):
                current_title = st.session_state.chat_history[st.session_state.current_chat_id]["title"]
                st.info(f"📝 Đang chỉnh sửa: {current_title}", icon="✏️")
            else:
                st.info("🆕 Chat mới (chưa lưu)", icon="💬")
        except (IndexError, KeyError):
            # Nếu có lỗi, reset về chat mới
            st.session_state.current_chat_id = None
            st.info("🆕 Chat mới (chưa lưu)", icon="💬")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 Mở chat mới", help="Lưu chat hiện tại và tạo chat mới", key="new_chat"):
                if create_new_chat():
                    st.session_state['refresh_needed'] = True
        
        with col2:
            if st.button("🗑️ Xóa", help="Xóa cuộc trò chuyện hiện tại", key="clear_current"):
                st.session_state.messages = []
                st.session_state.current_chat_id = None
                st.session_state['refresh_needed'] = True
        
        # Lịch sử chat
        if st.session_state.chat_history:
            st.subheader("📚 Lịch sử")
            
            # Hiển thị tối đa 5 chat gần nhất
            display_count = min(5, len(st.session_state.chat_history))
            start_index = len(st.session_state.chat_history) - display_count
            
            for i in range(display_count):
                chat_index = start_index + i
                
                # Kiểm tra bounds an toàn
                if chat_index >= len(st.session_state.chat_history):
                    continue
                    
                try:
                    chat = st.session_state.chat_history[chat_index]
                    
                    # Highlight chat đang active
                    is_current = (st.session_state.current_chat_id == chat_index)
                    button_type = "primary" if is_current else "secondary"
                    icon = "📝" if is_current else "📖"
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Sử dụng unique key cho mỗi button
                        safe_timestamp = chat['timestamp'].replace('/', '_').replace(' ', '_').replace(':', '_')
                        button_key = f"load_chat_{chat_index}_{safe_timestamp}"
                        if st.button(f"{icon} {chat['title']}", key=button_key, help=f"{chat['timestamp']} {'(Đang chỉnh sửa)' if is_current else ''}", type=button_type):
                            if not is_current:  # Chỉ load nếu không phải chat hiện tại
                                load_chat_from_history(chat_index)
                                st.session_state['refresh_needed'] = True
                    
                    with col2:
                        delete_key = f"delete_chat_{chat_index}_{safe_timestamp}"
                        if st.button("🗑️", key=delete_key, help="Xóa chat này"):
                            delete_chat_from_history(chat_index)
                            st.session_state['refresh_needed'] = True
                            
                except (IndexError, KeyError):
                    # Bỏ qua chat bị lỗi
                    continue
            
            if len(st.session_state.chat_history) > 5:
                st.caption(f"Hiển thị {display_count}/{len(st.session_state.chat_history)} chat gần nhất")
            
            if st.button("🗑️ Xóa tất cả", type="secondary", key="clear_all_chats"):
                clear_chat_history()

# === MAIN CONTENT ===

# === MAIN CONTENT - TẢI MODELS ===
if not st.session_state.models_loaded:
    st.info("🔄 Đang khởi tạo hệ thống AI... Vui lòng chờ trong giây lát")
    
    # Hiển thị thông tin GPU
    if torch.cuda.is_available():
        st.success(f"✅ Phát hiện GPU: {torch.cuda.get_device_name()}")
    else:
        st.warning("⚠️ Chỉ sử dụng CPU - có thể chậm hơn")
    
    # Container để hiển thị tiến trình
    progress_container = st.container()
    
    # Tải embedding
    with progress_container:
        st.write("📥 Đang tải Embedding Model...")
        try:
            if not st.session_state.embeddings:
                st.session_state.embeddings = load_embeddings()
            st.write("✅ Embedding Model đã sẵn sàng")
        except Exception as e:
            st.error(f"❌ Lỗi tải Embedding: {str(e)}")
            st.stop()
    
    # Tải LLM
    with progress_container:
        st.write("🤖 Đang tải Large Language Model...")
        try:
            if not st.session_state.llm:
                st.session_state.llm = load_llm()
            
            if st.session_state.llm:
                st.write("✅ LLM đã sẵn sàng")
                st.session_state.models_loaded = True
                progress_container.success("🎉 Hệ thống đã sẵn sàng! Hãy tải file PDF ở sidebar.")
                st.balloons()
            else:
                st.error("❌ Không thể tải LLM")
                st.stop()
        except Exception as e:
            st.error(f"❌ Lỗi tải LLM: {str(e)}")
            st.stop()

# === MAIN CONTENT - CHAT INTERFACE ===

# Giao diện chat chính
if st.session_state.rag_chain:
    st.header("💬 Trò chuyện với tài liệu")
    
    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input chat
    if prompt := st.chat_input("💭 Hãy hỏi tôi bất cứ điều gì về tài liệu..."):
        # Thêm tin nhắn người dùng
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Tạo và hiển thị phản hồi
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.info("🤔 Đang suy nghĩ...")
            
            try:
                response = st.session_state.rag_chain.invoke(prompt)
                response_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Tự động lưu chat sau khi có phản hồi
                auto_save_current_chat()
                
            except Exception as e:
                error_msg = f"❌ Có lỗi xảy ra khi xử lý câu hỏi. Vui lòng thử lại."
                response_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

elif st.session_state.models_loaded:
    # Hướng dẫn sử dụng khi models đã tải
    st.info("🚀 **Hệ thống đã sẵn sàng!** Hãy tải file PDF ở sidebar để bắt đầu.")
    
    # Thông tin hệ thống
    with st.expander("🔧 Thông tin hệ thống"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**GPU:**", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
            st.write("**Models:**", "✅ Đã tải" if st.session_state.models_loaded else "❌ Chưa tải")
        with col2:
            st.write("**Embedding:**", "✅ Sẵn sàng" if st.session_state.embeddings else "❌ Chưa tải")
            st.write("**RAG:**", "✅ Sẵn sàng" if st.session_state.rag_chain else "❌ Chưa có tài liệu")
    
    # Enhanced chunking info
    with st.expander("🚀 Enhanced Chunking Features"):
        st.markdown("""
        ### ✨ Cải tiến mới:
        - **🔥 Hybrid Strategy**: Balance tốc độ vs chất lượng
        - **💾 Smart Caching**: Cache chunks để load instant
        - **📊 Progress Tracking**: Real-time status updates
        - **🔗 Context Overlap**: Preserve context giữa chunks
        - **⚙️ Configurable**: Customize theo needs
        - **🛡️ Robust Fallback**: Auto-fallback nếu có lỗi
        """)
        
        # Current settings display
        settings = st.session_state.get("chunking_settings", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Strategy", settings.get("strategy", "hybrid"))
        with col2:
            st.metric("📏 Chunk Size", f"{settings.get('chunk_size', 1000)} chars")
        with col3:
            st.metric("🔗 Overlap", f"{settings.get('overlap', 100)} chars")
            
    # Hướng dẫn chi tiết
    st.markdown("""
    ### 📋 Hướng dẫn sử dụng:
    1. **⚙️ Cấu hình**: Chọn chunking strategy ở sidebar
    2. **📂 Tải tài liệu**: Chọn file PDF ở sidebar bên trái
    3. **🚀 Xử lý**: Nhấn nút "Xử lý PDF với Enhanced Chunking"
    4. **💬 Trò chuyện**: Bắt đầu hỏi đáp về nội dung tài liệu
    5. **💾 Quản lý**: Lưu, tải lại hoặc xóa các cuộc trò chuyện
    
    ### 🎯 Performance Tips:
    - **Hybrid strategy**: Tốt nhất cho hầu hết documents
    - **Enable cache**: Lần đầu chậm, lần sau instant
    - **Fixed strategy**: Dùng cho documents lớn (>20 pages)
    - **Semantic strategy**: Dùng cho documents quan trọng cần độ chính xác cao
    """)

# =================================================================
# XỬ LÝ REFRESH GIAO DIỆN
# =================================================================

# Kiểm tra và thực hiện refresh nếu cần
if st.session_state.get('refresh_needed', False):
    st.session_state['refresh_needed'] = False
    st.rerun() 