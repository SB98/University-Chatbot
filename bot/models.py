from django.db import models

# Create your models here.

class UniqueEntity(models.Model):
    EId = models.IntegerField(primary_key=True)
    Entity=models.TextField()

class UniqueIntent(models.Model):
    IId = models.IntegerField(primary_key=True)
    Intent = models.TextField()


class Responses(models.Model):
    RId=models.IntegerField(primary_key=True)
    Entity_id=models.ForeignKey(UniqueEntity,on_delete='true',default=0,db_column='Entity_id')
    Intent_id = models.ForeignKey(UniqueIntent, on_delete='true',default=0,db_column='Intent_id')
    Response=models.TextField()



