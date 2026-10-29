import datetime
from django import forms

class PaperSearchForm(forms.Form):
    """Search form for academic research paper discovery."""
    SORT_CHOICES = [
        ('relevance', 'Most Relevant'),
        ('citations', 'Most Cited'),
        ('date_desc', 'Newest First'),
        ('date_asc', 'Oldest First'),
    ]

    q = forms.CharField(
        required=True,
        max_length=300,
        widget=forms.TextInput(attrs={
            'class': 'form-input search-input',
            'placeholder': 'Enter research topic or keywords (e.g., "Deep learning for diabetic retinopathy detection")',
            'autofocus': True,
        })
    )
    start_year = forms.IntegerField(
        required=False,
        min_value=1800,
        widget=forms.NumberInput(attrs={
            'class': 'form-input',
            'placeholder': 'From year (e.g. 2019)'
        })
    )
    end_year = forms.IntegerField(
        required=False,
        min_value=1800,
        widget=forms.NumberInput(attrs={
            'class': 'form-input',
            'placeholder': 'To year (e.g. 2025)'
        })
    )
    sort_by = forms.ChoiceField(
        required=False,
        choices=SORT_CHOICES,
        initial='relevance',
        widget=forms.Select(attrs={
            'class': 'form-select'
        })
    )
    page = forms.IntegerField(
        required=False,
        initial=1,
        widget=forms.HiddenInput()
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        current_year = datetime.date.today().year
        self.fields['start_year'].max_value = current_year + 1
        self.fields['end_year'].max_value = current_year + 1

    def clean(self):
        cleaned_data = super().clean()
        start = cleaned_data.get('start_year')
        end = cleaned_data.get('end_year')
        if start and end and start > end:
            self.add_error('start_year', 'Start year cannot be greater than end year.')
        return cleaned_data

