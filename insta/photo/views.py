from django.shortcuts import render, redirect
from django.views.generic.list import ListView
from django.views.generic.edit import UpdateView, CreateView, DeleteView
from django.views.generic.detail import DetailView
from .models import Photo
from django.http import HttpResponseRedirect
from django.contrib import messages
from django.contrib.auth.models import User

class PhotoList(ListView):
    model = Photo
    template_name_suffix = '_list'

class PhotoCreate(CreateView):
    model = Photo
    fields = ['text', 'image']
    template_name_suffix = '_create'
    success_url = '/'

    def form_valid(self, form):
        form.instance.author_id=self.request.user.id
        if form.is_valid():
            #올바르다면
            #form: 모델폼
            form.instance.save()
            return redirect('/')
        else:
            #올바르지 않다면
            return self.render_to_response({'form': form})

class PhotoUpdate(UpdateView):
    model = Photo
    fields = ['text', 'image']
    template_name_suffix = '_update'
    success_url = '/'

    def dispatch(self, request, *args, **kwargs):
        object = self.get_object()
        if object.author != request.user:
            messages.warning(request, '수정할 권한이 없습니다.')
            return HttpResponseRedirect('/')
            #삭제 페이지에서 권한이 없다고 띄우거나 detail페이지로 들어가서 삭제에 실패했다 라고 띄우거나
        else:
            return super(PhotoUpdate, self).dispatch(request, *args, **kwargs)

class PhotoDelete(DeleteView):
    model = Photo
    template_name_suffix = '_delete'
    success_url = '/'

    def dispatch(self, request, *args, **kwargs):
        object = self.get_object()
        if object.author != request.user:
            messages.warning(request, '수정할 권한이 없습니다.')
            return HttpResponseRedirect('/')
            #삭제 페이지에서 권한이 없다고 띄우거나 detail페이지로 들어가서 삭제에 실패했다 라고 띄우거나
        else:
            return super(PhotoDelete, self).dispatch(request, *args, **kwargs)

class PhotoDetail(DetailView):
    model = Photo
    template_name_suffix = '_detail'

def test(request):
    return render(request, 'photo/test.html')

def home(request):
    return render(request, 'photo/home.html')