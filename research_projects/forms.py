from django import forms
from .models import ResearchProject

class ResearchProjectForm(forms.ModelForm):
    """Form to create or edit a research project workspace."""
    class Meta:
        model = ResearchProject
        fields = ['name', 'research_topic', 'description']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-input',
                'placeholder': 'e.g. Early Kidney Disease Diagnosis with ML'
            }),
            'research_topic': forms.TextInput(attrs={
                'class': 'form-input',
                'placeholder': 'e.g. Clinical Machine Learning & Biomarkers'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-textarea',
                'rows': 4,
                'placeholder': 'Describe your research goals, specific questions, or literature focus...'
            }),
        }
