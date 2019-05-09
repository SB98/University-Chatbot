from django.db import models

# Create your models here.


class entities(models.Model):
    EnId=models.IntegerField(primary_key=True)
    Question=models.TextField()
    Entity=models.TextField()
