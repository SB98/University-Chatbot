from django.db import models

# Create your models here.


class intents(models.Model):
    InId=models.IntegerField(primary_key=True)
    Question=models.TextField()
    Intent=models.TextField()
