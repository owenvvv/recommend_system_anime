from django.contrib import admin
from movie_recommend.models import  UserProfile

class MoviesAdmin(admin.ModelAdmin):
	list_display = ['title', 'description']

admin.site.register(UserProfile)

