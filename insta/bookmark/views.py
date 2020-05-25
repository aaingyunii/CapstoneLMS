from django.shortcuts import render

from .models import Bookmark

from django.views.generic.list import ListView
from django.views.generic.edit import CreateView, DeleteView, UpdateView
from django.views.generic.detail import DetailView

class BookmarkList(ListView):
    model = Bookmark

class BookmarkCreate(CreateView):
    model = Bookmark
    fields = ['site_name', 'url', 'contents']
    template_name_suffix = '_create'
    # tempate_name = 'bookmark/temlates/bookmark/'해도됨
    success_url = '/bookmark/' #메인화면으로 돌아간다.
class BookmarkUpdate(UpdateView):
    model = Bookmark
    fields = ['site_name', 'url', 'contents']
    template_name_suffix = '_update'
    success_url = '/bookmark/'

class BookmarkDelete(DeleteView):
    model = Bookmark
    fields = ['site_name', 'url', 'contents']
    template_name_suffix = '_delete'
    success_url = '/bookmark/'

class BookmarkDetail(DetailView):
    model = Bookmark
    template_name_suffix = '_detail'
    success_url = '/'