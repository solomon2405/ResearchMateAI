"""
Project-Scoped Persistent FAISS Vector Store.
Ensures strict multi-user and multi-project isolation by storing separate
FAISS indexes, metadata, and ID mappings per project.
Features cross-process/thread concurrency locking, atomic file publishing,
dimension/model validation, and consistency reconciliation.
"""

import os
import json
import logging
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import faiss
from django.conf import settings

logger = logging.getLogger(__name__)

# Process-level thread locks per project
_PROJECT_THREAD_LOCKS: Dict[int, threading.RLock] = {}
_GLOBAL_LOCK_REGISTRY = threading.Lock()


def get_project_thread_lock(project_id: int) -> threading.RLock:
    """Retrieve or create an in-process re-entrant lock for a specific project."""
    with _GLOBAL_LOCK_REGISTRY:
        if project_id not in _PROJECT_THREAD_LOCKS:
            _PROJECT_THREAD_LOCKS[project_id] = threading.RLock()
        return _PROJECT_THREAD_LOCKS[project_id]


_LOCK_OWNERSHIP = threading.local()


class ProjectFileLock:
    """
    Cross-process/thread filesystem lock for serialized index mutations.
    Re-entrant per thread and safe on both Windows and POSIX environments.
    """
    def __init__(self, lock_file_path: Path):
        self.lock_file_path = lock_file_path
        self._file_handle = None
        self._is_nested = False

    def __enter__(self):
        if not hasattr(_LOCK_OWNERSHIP, 'held_locks'):
            _LOCK_OWNERSHIP.held_locks = set()

        lock_key = str(self.lock_file_path.resolve())
        if lock_key in _LOCK_OWNERSHIP.held_locks:
            self._is_nested = True
            return self

        self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = open(self.lock_file_path, 'a+')
        try:
            if os.name == 'nt':
                import msvcrt
                self._file_handle.seek(0)
                msvcrt.locking(self._file_handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_EX)
            _LOCK_OWNERSHIP.held_locks.add(lock_key)
        except Exception as e:
            logger.warning(f"Could not acquire OS file lock for {self.lock_file_path}: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._is_nested:
            return
        lock_key = str(self.lock_file_path.resolve())
        if hasattr(_LOCK_OWNERSHIP, 'held_locks') and lock_key in _LOCK_OWNERSHIP.held_locks:
            _LOCK_OWNERSHIP.held_locks.remove(lock_key)
        if self._file_handle:
            try:
                if os.name == 'nt':
                    import msvcrt
                    self._file_handle.seek(0)
                    msvcrt.locking(self._file_handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            finally:
                self._file_handle.close()
                self._file_handle = None


class ProjectVectorStore:
    def __init__(self, project_id: int):
        self.project_id = project_id
        self.index_dir = Path(getattr(settings, 'VECTOR_INDEX_DIR', settings.BASE_DIR / 'vector_indexes'))
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.index_dir / f"project_{project_id}.index"
        self.mapping_path = self.index_dir / f"project_{project_id}_map.json"
        self.lock_path = self.index_dir / f".project_{project_id}.lock"
        
        self.dimension = 384
        self.expected_model = getattr(settings, 'EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunk_ids: List[int] = []
        self.metadata: Dict[str, Any] = {}
        self._load_or_create()

    def get_lock(self):
        """Context manager combining in-memory RLock and file lock."""
        thread_lock = get_project_thread_lock(self.project_id)
        file_lock = ProjectFileLock(self.lock_path)

        class CombinedLock:
            def __enter__(self):
                thread_lock.acquire()
                file_lock.__enter__()
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                try:
                    file_lock.__exit__(exc_type, exc_val, exc_tb)
                finally:
                    thread_lock.release()

        return CombinedLock()

    def _load_or_create(self):
        """Load index and ID mapping from disk, or initialize fresh index."""
        if self.index_path.exists() and self.mapping_path.exists():
            try:
                loaded_index = faiss.read_index(str(self.index_path))
                with open(self.mapping_path, 'r', encoding='utf-8') as f:
                    raw_mapping = json.load(f)

                # Support both schema v2.0 dict and legacy list format
                if isinstance(raw_mapping, dict):
                    self.metadata = raw_mapping
                    self.chunk_ids = raw_mapping.get('chunk_ids', [])
                    stored_model = raw_mapping.get('embedding_model')
                    if stored_model and stored_model != self.expected_model:
                        logger.warning(
                            f"Model mismatch in project {self.project_id}: index uses '{stored_model}', "
                            f"configured is '{self.expected_model}'."
                        )
                else:
                    self.chunk_ids = raw_mapping
                    self.metadata = {'schema_version': '1.0', 'chunk_ids': self.chunk_ids}

                # Validate dimension and count
                if loaded_index.d == self.dimension and loaded_index.ntotal == len(self.chunk_ids):
                    self.index = loaded_index
                    return
                else:
                    logger.warning(
                        f"Inconsistent vector store for project {self.project_id}: "
                        f"index.ntotal={loaded_index.ntotal}, chunks={len(self.chunk_ids)}, d={loaded_index.d}."
                    )
            except Exception as e:
                logger.error(f"Error reading vector index for project {self.project_id}: {e}. Creating new index.")

        # Create IndexFlatIP (Inner Product = Cosine Similarity for unit vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunk_ids = []
        self.metadata = {
            'schema_version': '2.0',
            'project_id': self.project_id,
            'embedding_model': self.expected_model,
            'dimension': self.dimension,
            'chunk_ids': [],
            'vector_count': 0,
            'created_at': datetime.now(timezone.utc).isoformat()
        }

    def save(self):
        """
        Persist index and metadata atomically using temporary replacement files.
        Guarantees that index and mapping are published simultaneously without partial states.
        """
        if self.index is None:
            return

        with self.get_lock():
            # Prepare metadata dictionary
            meta_payload = {
                'schema_version': '2.0',
                'project_id': self.project_id,
                'embedding_model': self.expected_model,
                'dimension': self.dimension,
                'vector_count': self.index.ntotal,
                'chunk_ids': self.chunk_ids,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }

            temp_idx_fd, temp_idx_path = tempfile.mkstemp(
                dir=str(self.index_dir),
                prefix=f"p_{self.project_id}_idx_",
                suffix=".tmp"
            )
            os.close(temp_idx_fd)

            temp_map_fd, temp_map_path = tempfile.mkstemp(
                dir=str(self.index_dir),
                prefix=f"p_{self.project_id}_map_",
                suffix=".tmp"
            )
            os.close(temp_map_fd)

            try:
                faiss.write_index(self.index, temp_idx_path)
                with open(temp_map_path, 'w', encoding='utf-8') as f:
                    json.dump(meta_payload, f, indent=2)

                # Atomically replace target files
                os.replace(temp_idx_path, str(self.index_path))
                os.replace(temp_map_path, str(self.mapping_path))
                self.metadata = meta_payload
            except Exception as e:
                logger.error(f"Failed to atomically persist index for project {self.project_id}: {e}")
                # Clean up temp files if replacement failed
                if os.path.exists(temp_idx_path):
                    try:
                        os.remove(temp_idx_path)
                    except Exception:
                        pass
                if os.path.exists(temp_map_path):
                    try:
                        os.remove(temp_map_path)
                    except Exception:
                        pass
                raise e

    def add_chunks(self, chunks: List[Any], embeddings: np.ndarray):
        """
        Add new chunk embeddings to this project's vector store under concurrency lock.
        
        Args:
            chunks: List of DocumentChunk objects or dicts containing 'id'.
            embeddings: 2D numpy array of shape (len(chunks), dimension).
        """
        if len(chunks) == 0:
            return

        if embeddings.shape[0] != len(chunks):
            raise ValueError(f"Mismatch: {len(chunks)} chunks provided but embeddings shape is {embeddings.shape}.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match expected {self.dimension}.")

        with self.get_lock():
            # Reload latest state from disk in case another thread/worker committed
            self._load_or_create()

            # Ensure float32, contiguous, and unit-normalized
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms

            self.index.add(embeddings)

            for chunk in chunks:
                chunk_id = chunk['id'] if isinstance(chunk, dict) else chunk.id
                self.chunk_ids.append(chunk_id)

            self.save()
            logger.info(f"Added {len(chunks)} chunks to project {self.project_id} index (total: {self.index.ntotal}).")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Perform similarity search using query vector.
        Thread-safe read against current in-memory index.
        
        Returns:
            List of (chunk_id, similarity_score) sorted by score descending.
        """
        if self.index is None or self.index.ntotal == 0 or len(self.chunk_ids) == 0:
            return []

        # Ensure float32, contiguous, unit-normalized query vector
        query_vec = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunk_ids):
                chunk_id = self.chunk_ids[idx]
                results.append((chunk_id, float(dist)))

        return results

    def rebuild_from_database(self) -> Dict[str, Any]:
        """
        Reconstruct this project's index exclusively from eligible DocumentChunk records
        belonging to documents with 'indexed' processing_status.
        Completely replaces stale or corrupt index files atomically.
        """
        from documents.models import DocumentChunk
        from .embeddings import embed_texts

        with self.get_lock():
            eligible_chunks = list(
                DocumentChunk.objects.filter(
                    project_id=self.project_id,
                    document__processing_status='indexed'
                ).select_related('document').order_by('id')
            )
            
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunk_ids = []

            if not eligible_chunks:
                self.save()
                return {
                    'status': 'cleared',
                    'indexed_chunks': 0,
                    'project_id': self.project_id
                }

            texts = [c.text for c in eligible_chunks]
            embeddings = embed_texts(texts)
            
            # Format and persist under lock
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms

            self.index.add(embeddings)
            self.chunk_ids = [c.id for c in eligible_chunks]
            self.save()

            logger.info(f"Rebuilt vector index for project {self.project_id} with {len(eligible_chunks)} chunks.")
            return {
                'status': 'rebuilt',
                'indexed_chunks': len(eligible_chunks),
                'project_id': self.project_id
            }

    def validate_integrity(self) -> Dict[str, Any]:
        """
        Audit the vector store against database records.
        Detects missing files, count mismatches, dimension anomalies, and orphaned chunk IDs.
        """
        from documents.models import DocumentChunk

        files_exist = self.index_path.exists() and self.mapping_path.exists()
        if not files_exist:
            db_chunks_count = DocumentChunk.objects.filter(
                project_id=self.project_id,
                document__processing_status='indexed'
            ).count()
            return {
                'is_valid': db_chunks_count == 0,
                'has_files': False,
                'vector_count': 0,
                'mapping_count': 0,
                'db_eligible_count': db_chunks_count,
                'issue': 'Index files missing from disk.' if db_chunks_count > 0 else 'No index needed.'
            }

        ntotal = self.index.ntotal if self.index else 0
        map_count = len(self.chunk_ids)
        dim = self.index.d if self.index else 0

        # Fetch eligible DB chunk IDs
        db_chunk_ids = set(
            DocumentChunk.objects.filter(
                project_id=self.project_id,
                document__processing_status='indexed'
            ).values_list('id', flat=True)
        )

        index_chunk_ids = set(self.chunk_ids)
        orphaned_in_index = list(index_chunk_ids - db_chunk_ids)
        missing_from_index = list(db_chunk_ids - index_chunk_ids)

        is_valid = (
            ntotal == map_count and
            ntotal == len(db_chunk_ids) and
            dim == self.dimension and
            not orphaned_in_index and
            not missing_from_index
        )

        issue = None
        if not is_valid:
            issues = []
            if ntotal != map_count:
                issues.append(f"FAISS count ({ntotal}) != mapping count ({map_count})")
            if dim != self.dimension:
                issues.append(f"Dimension {dim} != expected {self.dimension}")
            if orphaned_in_index:
                issues.append(f"{len(orphaned_in_index)} orphaned chunk IDs in index")
            if missing_from_index:
                issues.append(f"{len(missing_from_index)} eligible DB chunks missing from index")
            issue = "; ".join(issues)

        return {
            'is_valid': is_valid,
            'has_files': True,
            'vector_count': ntotal,
            'mapping_count': map_count,
            'dimension': dim,
            'db_eligible_count': len(db_chunk_ids),
            'orphaned_count': len(orphaned_in_index),
            'missing_count': len(missing_from_index),
            'issue': issue
        }

    def delete_index(self):
        """Remove index files and lock from disk when project is deleted."""
        with self.get_lock():
            try:
                if self.index_path.exists():
                    os.remove(self.index_path)
                if self.mapping_path.exists():
                    os.remove(self.mapping_path)
                self.index = None
                self.chunk_ids = []
                self.metadata = {}
                logger.info(f"Deleted vector index files for project {self.project_id}.")
            except Exception as e:
                logger.error(f"Error deleting index for project {self.project_id}: {e}")
        # Clean up lock file afterwards if possible
        try:
            if self.lock_path.exists():
                os.remove(self.lock_path)
        except Exception:
            pass


def get_project_vector_store(project_id: int) -> ProjectVectorStore:
    """Helper to instantiate or retrieve project vector store."""
    return ProjectVectorStore(project_id)

