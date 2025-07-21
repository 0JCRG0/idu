from django.urls import path

from . import views

urlpatterns = [
    path("", views.extract_entities_ui, name="extract_entities_ui"),
    path("extract-entities/", views.extract_entities, name="extract_entities"),
    path("healthcheck/", views.health_check, name="health_check"),
]
