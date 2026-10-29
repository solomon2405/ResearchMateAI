from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User

class AuthenticationTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user_data = {
            'username': 'dr_aloe',
            'email': 'aloe@university.edu',
            'password': 'StrongPassword123!',
            'password_confirm': 'StrongPassword123!',
        }
        self.user = User.objects.create_user(
            username='existing_user',
            email='existing@university.edu',
            password='ValidPassword123!'
        )

    def test_registration_success(self):
        response = self.client.post(reverse('register'), self.user_data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(User.objects.filter(username='dr_aloe').exists())
        # Should be logged in and on dashboard
        self.assertTrue(response.context['user'].is_authenticated)

    def test_registration_password_mismatch(self):
        data = self.user_data.copy()
        data['password_confirm'] = 'DifferentPassword123!'
        response = self.client.post(reverse('register'), data)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username='dr_aloe').exists())
        self.assertFormError(response.context['form'], 'password_confirm', 'Passwords do not match.')

    def test_registration_duplicate_email(self):
        data = self.user_data.copy()
        data['email'] = 'existing@university.edu'
        response = self.client.post(reverse('register'), data)
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response.context['form'], 'email', 'An account with this email already exists.')

    def test_login_success(self):
        response = self.client.post(reverse('login'), {
            'username': 'existing_user',
            'password': 'ValidPassword123!'
        }, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['user'].is_authenticated)

    def test_login_invalid_credentials(self):
        response = self.client.post(reverse('login'), {
            'username': 'existing_user',
            'password': 'WrongPassword!'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Invalid username or password")

    def test_logout_post_success(self):
        self.client.login(username='existing_user', password='ValidPassword123!')
        response = self.client.post(reverse('logout'), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context['user'].is_authenticated)

    def test_logout_get_rejected(self):
        """Ensure GET requests to logout are rejected with 405 Method Not Allowed."""
        self.client.login(username='existing_user', password='ValidPassword123!')
        response = self.client.get(reverse('logout'))
        self.assertEqual(response.status_code, 405)

    def test_login_safe_redirect(self):
        """Ensure valid relative next redirects are honored."""
        response = self.client.post(
            f"{reverse('login')}?next=/projects/create/",
            {
                'username': 'existing_user',
                'password': 'ValidPassword123!'
            }
        )
        self.assertRedirects(response, '/projects/create/')

    def test_login_unsafe_redirect_rejected(self):
        """Ensure external/malicious next redirects are blocked and fall back to dashboard."""
        unsafe_urls = [
            'https://malicious-external-site.com',
            '//attacker.com/evil',
            'javascript:alert(1)',
        ]
        for bad_url in unsafe_urls:
            response = self.client.post(
                f"{reverse('login')}?next={bad_url}",
                {
                    'username': 'existing_user',
                    'password': 'ValidPassword123!'
                }
            )
            self.assertRedirects(response, reverse('dashboard'))

    def test_registration_weak_password_rejected(self):
        """Ensure weak/common password is rejected by Django's password validators."""
        data = self.user_data.copy()
        data['username'] = 'dr_weak'
        data['email'] = 'weak@university.edu'
        data['password'] = '12345678'
        data['password_confirm'] = '12345678'
        response = self.client.post(reverse('register'), data)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username='dr_weak').exists())

    def test_protected_pages_redirect_to_login(self):
        protected_urls = [
            reverse('dashboard'),
            reverse('project_list'),
            reverse('project_create'),
            reverse('chat_main'),
            reverse('profile'),
        ]
        for url in protected_urls:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 302)
            self.assertTrue(response.url.startswith(reverse('login')))
