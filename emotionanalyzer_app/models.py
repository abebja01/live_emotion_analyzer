from django.db import models

# Create your models here.

class UploadedFile(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.CharField(max_length=255)
    url = models.CharField(max_length=255)
    date = models.DateField()
    def __str__(self):
        return self.url

