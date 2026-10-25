from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_view, name='chat_main'),
    path('<int:conversation_id>/', views.chat_view, name='chat_conversation'),
    path('new/<int:project_id>/', views.create_conversation_view, name='chat_create_conversation'),
    path('<int:conversation_id>/send/', views.send_message_api, name='chat_send_message'),
    path('<int:conversation_id>/rename/', views.rename_conversation_view, name='chat_rename'),
    path('<int:conversation_id>/delete/', views.delete_conversation_view, name='chat_delete'),
]
