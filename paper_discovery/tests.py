from unittest.mock import patch, MagicMock
from django.test import TestCase, Client
from django.urls import reverse
import httpx
from paper_discovery.services.openalex import OpenAlexService
from paper_discovery.services.normalizers import reconstruct_abstract, normalize_openalex_work

class PaperDiscoveryTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.sample_work = {
            'id': 'https://openalex.org/W123456789',
            'title': 'Attention Is All You Need for Kidney Diagnosis',
            'publication_year': 2023,
            'cited_by_count': 142,
            'authorships': [
                {'author': {'display_name': 'Dr. Alice Smith'}},
                {'author': {'display_name': 'Dr. Bob Jones'}},
            ],
            'primary_location': {
                'source': {'display_name': 'Journal of Biomedical Informatics'},
                'landing_page_url': 'https://doi.org/10.1016/j.jbi.2023.104444',
                'pdf_url': 'https://example.org/papers/attention_kidney.pdf',
            },
            'open_access': {
                'is_oa': True,
                'oa_status': 'gold',
                'oa_url': 'https://example.org/papers/attention_kidney.pdf'
            },
            'doi': 'https://doi.org/10.1016/j.jbi.2023.104444',
            'abstract_inverted_index': {
                'Transformer': [0],
                'architectures': [1],
                'improve': [2],
                'early': [3],
                'detection': [4],
                'accuracy': [5],
            }
        }

    def test_reconstruct_abstract(self):
        inverted_index = {
            'Deep': [0],
            'learning': [1],
            'detects': [2],
            'disease': [3],
            'accurately': [4],
        }
        text = reconstruct_abstract(inverted_index)
        self.assertEqual(text, 'Deep learning detects disease accurately')

    def test_reconstruct_abstract_empty(self):
        self.assertEqual(reconstruct_abstract(None), 'Abstract unavailable.')
        self.assertEqual(reconstruct_abstract({}), 'Abstract unavailable.')

    def test_normalize_openalex_work(self):
        normalized = normalize_openalex_work(self.sample_work)
        self.assertEqual(normalized['title'], 'Attention Is All You Need for Kidney Diagnosis')
        self.assertEqual(normalized['authors_display'], 'Dr. Alice Smith, Dr. Bob Jones')
        self.assertEqual(normalized['publication_year'], 2023)
        self.assertEqual(normalized['venue'], 'Journal of Biomedical Informatics')
        self.assertEqual(normalized['abstract'], 'Transformer architectures improve early detection accuracy')
        self.assertEqual(normalized['pdf_url'], 'https://example.org/papers/attention_kidney.pdf')
        self.assertTrue(normalized['is_oa'])
        self.assertEqual(normalized['source_provider'], 'OpenAlex')

    @patch('httpx.Client.get')
    def test_openalex_service_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'meta': {'count': 1},
            'results': [self.sample_work]
        }
        mock_get.return_value = mock_response

        service = OpenAlexService()
        result = service.search_works('kidney disease transformer')

        self.assertIsNone(result['error'])
        self.assertEqual(result['total_count'], 1)
        self.assertEqual(len(result['results']), 1)
        self.assertEqual(result['results'][0]['title'], 'Attention Is All You Need for Kidney Diagnosis')

    def test_openalex_service_empty_query(self):
        service = OpenAlexService()
        result = service.search_works('   ')
        self.assertIsNotNone(result['error'])
        self.assertEqual(result['results'], [])

    @patch('httpx.Client.get')
    def test_openalex_service_timeout(self, mock_get):
        mock_get.side_effect = httpx.TimeoutException("Timed out")
        service = OpenAlexService()
        result = service.search_works('large query')
        self.assertIn('timed out', result['error'].lower())

    @patch('httpx.Client.get')
    def test_openalex_service_rate_limit(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        service = OpenAlexService()
        result = service.search_works('rate limited query')
        self.assertIn('rate limit', result['error'].lower())

    @patch('paper_discovery.services.openalex.OpenAlexService.search_works')
    def test_search_view_integration(self, mock_search):
        mock_search.return_value = {
            'results': [normalize_openalex_work(self.sample_work)],
            'total_count': 1,
            'page': 1,
            'per_page': 10,
            'total_pages': 1,
            'error': None
        }

        response = self.client.get(reverse('paper_search'), {'q': 'kidney disease'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Attention Is All You Need for Kidney Diagnosis')
        self.assertContains(response, 'Journal of Biomedical Informatics')
        self.assertContains(response, 'Open PDF Link')
