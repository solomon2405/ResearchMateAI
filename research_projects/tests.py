from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from research_projects.models import ResearchProject

class ResearchProjectTests(TestCase):
    def setUp(self):
        self.user1 = User.objects.create_user(username='researcher_1', password='Password123!')
        self.user2 = User.objects.create_user(username='researcher_2', password='Password123!')
        
        self.client1 = Client()
        self.client1.login(username='researcher_1', password='Password123!')
        
        self.client2 = Client()
        self.client2.login(username='researcher_2', password='Password123!')

        self.project1 = ResearchProject.objects.create(
            owner=self.user1,
            name='Kidney Disease Prediction',
            research_topic='Clinical Machine Learning',
            description='Study of early biomarker classification'
        )

    def test_project_create_view(self):
        response = self.client1.post(reverse('project_create'), {
            'name': 'Genomic Foundation Models',
            'research_topic': 'Genomics',
            'description': 'Evaluating DNA language models'
        }, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(ResearchProject.objects.filter(name='Genomic Foundation Models', owner=self.user1).exists())

    def test_project_detail_view_owner(self):
        response = self.client1.get(reverse('project_detail', kwargs={'pk': self.project1.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Kidney Disease Prediction')

    def test_multi_user_isolation_cannot_view_other_user_project(self):
        # User 2 tries to view User 1's project -> must be 404
        response = self.client2.get(reverse('project_detail', kwargs={'pk': self.project1.pk}))
        self.assertEqual(response.status_code, 404)

    def test_multi_user_isolation_cannot_edit_other_user_project(self):
        response = self.client2.post(reverse('project_edit', kwargs={'pk': self.project1.pk}), {
            'name': 'Hacked Project Name',
            'research_topic': 'Malicious',
        })
        self.assertEqual(response.status_code, 404)
        self.project1.refresh_from_db()
        self.assertEqual(self.project1.name, 'Kidney Disease Prediction')

    def test_multi_user_isolation_cannot_delete_other_user_project(self):
        response = self.client2.post(reverse('project_delete', kwargs={'pk': self.project1.pk}))
        self.assertEqual(response.status_code, 404)
        self.assertTrue(ResearchProject.objects.filter(pk=self.project1.pk).exists())

    def test_owner_can_delete_own_project(self):
        response = self.client1.post(reverse('project_delete', kwargs={'pk': self.project1.pk}), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(ResearchProject.objects.filter(pk=self.project1.pk).exists())

    def test_dashboard_metrics(self):
        response = self.client1.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['total_projects'], 1)
        self.assertContains(response, 'Kidney Disease Prediction')
