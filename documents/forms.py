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

        validated_files = []
        for uploaded_file in uploaded_files:
            # 1. Extension check
            filename = uploaded_file.name.lower()
            if not filename.endswith('.pdf'):
                raise ValidationError(f"'{uploaded_file.name}' is not a PDF file. Only PDF documents are supported.")

            # 2. File size check
            if uploaded_file.size > max_size_bytes:
                raise ValidationError(
                    f"'{uploaded_file.name}' ({uploaded_file.size / (1024*1024):.1f} MB) exceeds "
                    f"the maximum allowed size of {max_size_mb} MB."
                )
            if uploaded_file.size < 100:
                raise ValidationError(f"'{uploaded_file.name}' is empty or too small to be a valid PDF.")

            # 3. Read content to compute checksum and verify PDF magic header
            content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset pointer for subsequent saving

            if not content.startswith(b'%PDF-'):
                raise ValidationError(f"'{uploaded_file.name}' does not have a valid PDF header.")

            # 4. PyMuPDF integrity check
            try:
                doc = fitz.open(stream=content, filetype="pdf")
                if doc.is_encrypted:
                    doc.close()
                    raise ValidationError(f"'{uploaded_file.name}' is encrypted or password-protected.")
                if doc.page_count < 1:
                    doc.close()
                    raise ValidationError(f"'{uploaded_file.name}' contains no readable pages.")
                doc.close()
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(f"'{uploaded_file.name}' could not be parsed: {str(e)}")

            # 5. Checksum duplicate check within this project
            file_hash = hashlib.sha256(content).hexdigest()
            if self.project and self.project.documents.filter(checksum=file_hash).exists():
                raise ValidationError(f"'{uploaded_file.name}' has already been uploaded to this project.")

            # Attach calculated checksum to file object for view efficiency
            uploaded_file.checksum = file_hash
            validated_files.append(uploaded_file)

        return validated_files
