import hashlib
import fitz  # PyMuPDF
from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError

class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True

class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput(attrs={
            'class': 'file-input',
            'accept': 'application/pdf,.pdf'
        }))
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = [single_file_clean(data, initial)]
        return result


class DocumentUploadForm(forms.Form):
    """
    Form supporting single or multi-PDF upload with strict server-side validation.
    """
    files = MultipleFileField(
        label="Select Research Paper PDFs",
        help_text=f"Upload up to {getattr(settings, 'MAX_DOCUMENTS_PER_PROJECT', 30)} PDFs (max {getattr(settings, 'MAX_UPLOAD_SIZE_MB', 25)} MB per file)."
    )

    def __init__(self, *args, project=None, **kwargs):
        self.project = project
        super().__init__(*args, **kwargs)

    def clean_files(self):
        uploaded_files = self.cleaned_data.get('files', [])
        if not uploaded_files:
            raise ValidationError("Please select at least one PDF file to upload.")

        max_size_mb = getattr(settings, 'MAX_UPLOAD_SIZE_MB', 25)
        max_size_bytes = max_size_mb * 1024 * 1024
        max_docs_limit = getattr(settings, 'MAX_DOCUMENTS_PER_PROJECT', 30)

        # Check project document quota
        if self.project:
            current_count = self.project.documents.count()
            if current_count + len(uploaded_files) > max_docs_limit:
                remaining_slots = max(0, max_docs_limit - current_count)
                raise ValidationError(
                    f"Upload would exceed the maximum project limit of {max_docs_limit} papers. "
                    f"You have {current_count} papers; only {remaining_slots} more can be uploaded."
                )

        max_pages_limit = getattr(settings, 'MAX_PAGES_PER_DOCUMENT', 500)
        seen_batch_checksums = set()
        seen_batch_names = set()
        validated_files = []

        for uploaded_file in uploaded_files:
            # 1. Extension check
            orig_name = uploaded_file.name or "document.pdf"
            filename_lower = orig_name.lower()
            if not filename_lower.endswith('.pdf'):
                raise ValidationError(f"'{orig_name}' is not a PDF file. Only PDF documents are supported.")

            # Check duplicate filenames in same batch
            if filename_lower in seen_batch_names:
                raise ValidationError(f"Duplicate filename '{orig_name}' detected in the upload batch.")
            seen_batch_names.add(filename_lower)

            # 2. File size check
            if uploaded_file.size > max_size_bytes:
                raise ValidationError(
                    f"'{orig_name}' ({uploaded_file.size / (1024*1024):.1f} MB) exceeds "
                    f"the maximum allowed size of {max_size_mb} MB."
                )
            if uploaded_file.size < 100:
                raise ValidationError(f"'{orig_name}' is empty or too small to be a valid PDF.")

            # 3. Streamingly compute SHA-256 hash in bounded 64KB blocks (memory-safe)
            hasher = hashlib.sha256()
            uploaded_file.seek(0)
            for chunk in uploaded_file.chunks(chunk_size=65536):
                hasher.update(chunk)
            file_hash = hasher.hexdigest()

            # Batch-level duplicate detection
            if file_hash in seen_batch_checksums:
                raise ValidationError(f"Duplicate file contents detected in upload batch for '{orig_name}'.")
            seen_batch_checksums.add(file_hash)

            # 4. Check PDF magic header from initial bytes
            uploaded_file.seek(0)
            header_sample = uploaded_file.read(1024)
            if b'%PDF-' not in header_sample[:128]:
                raise ValidationError(f"'{orig_name}' does not contain a valid %PDF- header signature.")

            # 5. PyMuPDF integrity and page count check
            uploaded_file.seek(0)
            doc = None
            try:
                if hasattr(uploaded_file, 'temporary_file_path'):
                    doc = fitz.open(uploaded_file.temporary_file_path())
                else:
                    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")

                if doc.is_encrypted:
                    raise ValidationError(f"'{orig_name}' is encrypted or password-protected.")
                if doc.page_count < 1:
                    raise ValidationError(f"'{orig_name}' contains no readable pages.")
                if doc.page_count > max_pages_limit:
                    raise ValidationError(
                        f"'{orig_name}' has {doc.page_count} pages, exceeding the limit of {max_pages_limit} pages."
                    )
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(f"'{orig_name}' could not be parsed as a valid PDF: {str(e)}")
            finally:
                if doc:
                    doc.close()
                uploaded_file.seek(0)

            # 6. Checksum duplicate check against existing project documents in DB
            if self.project and self.project.documents.filter(checksum=file_hash).exists():
                raise ValidationError(f"'{orig_name}' has already been uploaded to this project.")

            uploaded_file.checksum = file_hash
            validated_files.append(uploaded_file)

        return validated_files
