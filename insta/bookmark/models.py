from django.db import models

class Bookmark(models.Model):
    site_name = models.CharField(max_length=30)
    url = models.URLField()
    contents = models.TextField(blank=True)
    created = models.DateField(auto_now_add=True)

    def __str__(self):
        return "Site name : "+self.site_name+". URL : "+self.url
        
    class Meta:
        ordering = ["-created"]
