import sys
import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class ScreenKnowledge:
    """
    Structured Knowledge representation of a scanned window or document.
    """
    content_type: str  # 'dataset' | 'document' | 'web_page' | 'generic_ui'
    scope: str         # 'visible_region' | 'entire_source'
    source_app: str
    window_title: str
    extracted_text: str
    dataframe: Optional[pd.DataFrame] = None
    schema_info: Optional[Dict[str, Any]] = None
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_type": self.content_type,
            "scope": self.scope,
            "source_app": self.source_app,
            "window_title": self.window_title,
            "row_count": len(self.dataframe) if self.dataframe is not None else 0,
            "col_count": len(self.dataframe.columns) if self.dataframe is not None else 0,
            "columns": list(self.dataframe.columns) if self.dataframe is not None else [],
            "chunk_count": len(self.chunks),
            "summary": self.summary,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp
        }


class KnowledgeContextManager:
    """
    Multi-Context Stack Manager.
    Preserves active dataset, active document, web context, and generic UI context
    without destructive overwrites.
    """
    def __init__(self):
        self.dataset_context: Optional[ScreenKnowledge] = None
        self.document_context: Optional[ScreenKnowledge] = None
        self.web_context: Optional[ScreenKnowledge] = None
        self.generic_ui_context: Optional[ScreenKnowledge] = None
        self.history: List[ScreenKnowledge] = []

    def set_context(self, knowledge: ScreenKnowledge):
        if knowledge.content_type == 'dataset':
            self.dataset_context = knowledge
        elif knowledge.content_type == 'document':
            self.document_context = knowledge
        elif knowledge.content_type == 'web_page':
            self.web_context = knowledge
        else:
            self.generic_ui_context = knowledge
        self.history.append(knowledge)

    def detach(self, content_type: str):
        if content_type == 'dataset':
            self.dataset_context = None
        elif content_type == 'document':
            self.document_context = None
        elif content_type == 'web_page':
            self.web_context = None
        elif content_type == 'generic_ui':
            self.generic_ui_context = None
        elif content_type == 'all':
            self.dataset_context = None
            self.document_context = None
            self.web_context = None
            self.generic_ui_context = None

    def get_active_dataframe(self) -> Optional[pd.DataFrame]:
        if self.dataset_context and self.dataset_context.dataframe is not None:
            return self.dataset_context.dataframe
        return None

    def get_active_document_chunks(self) -> List[Dict[str, Any]]:
        if self.document_context:
            return self.document_context.chunks
        return []

    def get_active_summary_list(self) -> List[Dict[str, Any]]:
        active = []
        for ctx in [self.dataset_context, self.document_context, self.web_context, self.generic_ui_context]:
            if ctx is not None:
                active.append(ctx.to_dict())
        return active


class KnowledgeExtractor:
    """
    Extracted Content Classification & Ingestion Engine.
    """

    @staticmethod
    def classify_content_type(app_name: str, window_title: str, text: str, df: Optional[pd.DataFrame]) -> str:
        """Classifies content into 'dataset', 'document', 'web_page', or 'generic_ui'."""
        app_l = app_name.lower()
        title_l = window_title.lower()
        
        # 1. Dataset / Table detection
        if df is not None and len(df) > 1 and len(df.columns) > 1:
            return 'dataset'
        if any(ext in title_l for ext in ['.xlsx', '.csv', '.tsv', '.ods', 'excel']) or any(kw in app_l for kw in ['excel', 'sheets', 'calc']):
            return 'dataset'
        
        # 2. Document / PDF / Word detection
        if any(ext in title_l for ext in ['.pdf', '.docx', '.doc', '.txt', '.md', '.rtf']) or any(kw in app_l for kw in ['acrobat', 'word', 'notion', 'reader']):
            return 'document'
        
        # 3. Web Page detection
        if any(kw in app_l for kw in ['chrome', 'edge', 'firefox', 'browser', 'arc', 'brave']):
            return 'web_page'
        
        # 4. Fallback to document if long structured text, else generic UI
        if len(text) > 1500 or text.count('\n') > 25:
            return 'document'
        
        return 'generic_ui'

    @staticmethod
    def extract_grid_from_text(text: str) -> Optional[pd.DataFrame]:
        """Attempts to parse tabular grid structure (CSV/Tabular/Delimiter) from text."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) < 3:
            return None

        comma_counts = [l.count(',') for l in lines[:10]]
        tab_counts = [l.count('\t') for l in lines[:10]]
        pipe_counts = [l.count('|') for l in lines[:10]]

        delimiter = None
        if len(set(comma_counts)) == 1 and comma_counts[0] > 0:
            delimiter = ','
        elif len(set(tab_counts)) == 1 and tab_counts[0] > 0:
            delimiter = '\t'
        elif len(set(pipe_counts)) == 1 and pipe_counts[0] > 0:
            delimiter = '|'

        if delimiter:
            try:
                import io
                df = pd.read_csv(io.StringIO(text), sep=delimiter)
                if len(df.columns) >= 2 and len(df) >= 2:
                    return df
            except Exception:
                pass
        return None

    @staticmethod
    def chunk_document_semantically(text: str, heading_title: str = "Document Section") -> List[Dict[str, Any]]:
        """
        Chunks text semantically by headings, paragraphs, and blank lines.
        Preserves section index, heading, and char positions for RAG retrieval.
        """
        if not text:
            return []

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        chunks = []
        current_chunk = []
        current_len = 0
        section_idx = 1
        current_heading = heading_title

        for p in paragraphs:
            if len(p) < 80 and not p.endswith('.') and (p.isupper() or p.istitle() or p.startswith('#')):
                current_heading = p.lstrip('#').strip()
                continue

            current_chunk.append(p)
            current_len += len(p)

            if current_len >= 600:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "section_index": section_idx,
                    "heading": current_heading,
                    "text": chunk_text,
                    "char_length": len(chunk_text)
                })
                section_idx += 1
                current_chunk = []
                current_len = 0

        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append({
                "section_index": section_idx,
                "heading": current_heading,
                "text": chunk_text,
                "char_length": len(chunk_text)
            })

        return chunks

    @staticmethod
    def retrieve_document_chunks(query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs TF-IDF / term-overlap scoring to retrieve top-K relevant chunks for document Q&A."""
        if not chunks or not query:
            return []

        query_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower()))
        if not query_words:
            return chunks[:top_k]

        scored_chunks = []
        for c in chunks:
            text_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', c['text'].lower()))
            overlap = len(query_words.intersection(text_words))
            heading_words = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', c.get('heading', '').lower()))
            heading_overlap = len(query_words.intersection(heading_words))
            score = overlap + (heading_overlap * 2)
            scored_chunks.append((score, c))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_chunks[:top_k]]

    @staticmethod
    def clean_system_ui_noise(text: str) -> str:
        """Removes window chrome buttons, scrollbar controls, and system titlebar noise."""
        if not text:
            return ""
        
        noise_exact = {
            'minimize', 'maximize', 'restore', 'close', 'system', 'titlebar', 'sysmenu', 
            'system menu bar', 'close tab', 'new tab', 'more options', 'vertical', 
            'horizontal', 'vertical small decrease', 'vertical large decrease', 
            'vertical small increase', 'vertical large increase', 'horizontal small decrease', 
            'horizontal large decrease', 'horizontal small increase', 'horizontal large increase', 
            'scrollbar', 'thumb', 'desktopwindowxamlsource', 'applicationframewindow', 
            'chrome_widgetwin_1', 'systemitem', 'menu bar', 'toolbar', 'status bar'
        }

        clean_lines = []
        for line in text.split('\n'):
            l_strip = line.strip()
            l_lower = l_strip.lower()
            if not l_strip or len(l_strip) <= 1:
                continue
            if l_lower in noise_exact:
                continue
            if any(kw in l_lower for kw in ['small decrease', 'large decrease', 'small increase', 'large increase', 'desktopwindowxamlsource', 'close tab', 'new tab', 'sysmenu']):
                continue
            clean_lines.append(l_strip)

        return "\n".join(clean_lines)

    @classmethod
    def build_knowledge(cls, app_name: str, window_title: str, text: str, grid_df: Optional[pd.DataFrame] = None) -> ScreenKnowledge:
        """
        Factory method to construct a fully populated ScreenKnowledge instance.
        """
        text = cls.clean_system_ui_noise(text)
        if grid_df is None:
            grid_df = cls.extract_grid_from_text(text)


        content_type = cls.classify_content_type(app_name, window_title, text, grid_df)
        scope = "visible_region"  # UIA screen scans capture visible region

        schema_info = None
        chunks = []

        if content_type == 'dataset' and grid_df is not None:
            schema_info = {
                "columns": list(grid_df.columns),
                "row_count": len(grid_df),
                "column_count": len(grid_df.columns),
                "dtypes": {str(col): str(dtype) for col, dtype in grid_df.dtypes.items()},
                "null_counts": {str(col): int(grid_df[col].isnull().sum()) for col in grid_df.columns}
            }
            summary = f"✓ Read visible table — {len(grid_df):,} rows · {len(grid_df.columns)} columns in '{window_title}'"
            recommendations = ["📊 Run Auto-EDA", "📈 Calculate Column Metrics", "🔍 Filter Top Rows"]
        
        elif content_type == 'document':
            chunks = cls.chunk_document_semantically(text, heading_title=window_title)
            word_count = len(text.split())
            summary = f"✓ Read document — {len(chunks)} sections · {word_count:,} words in '{window_title}'"
            recommendations = ["📝 Summarize Full Document", "❓ Answer Questions on Document", "🔍 Key Takeaways"]
        
        elif content_type == 'web_page':
            chunks = cls.chunk_document_semantically(text, heading_title=window_title)
            summary = f"✓ Read web page — '{window_title}' ({len(text):,} chars parsed)"
            recommendations = ["📝 Summarize Page", "📊 Extract Page Tables", "🔍 Key Highlights"]
        
        else:
            summary = f"✓ Read workspace window — '{window_title}' ({len(text):,} chars parsed)"
            recommendations = ["📝 Summarize Screen Text", "🛠️ Explain Screen Context", "🔍 Profile Text"]

        return ScreenKnowledge(
            content_type=content_type,
            scope=scope,
            source_app=app_name,
            window_title=window_title,
            extracted_text=text,
            dataframe=grid_df,
            schema_info=schema_info,
            chunks=chunks,
            summary=summary,
            recommendations=recommendations
        )
