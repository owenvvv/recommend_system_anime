from django.db import models
from django.contrib.auth.models import User
import jsonfield
import json
import numpy as np


class UserProfile(models.Model):
	user = models.ForeignKey(User, unique=True,on_delete=models.CASCADE)
	array = jsonfield.JSONField()
	arrayratedmoviesindxs = jsonfield.JSONField()
	name = models.CharField(max_length=100)
	lastrecs = jsonfield.JSONField()

	def __str__(self):
		return self.name

	def save(self, *args, **kwargs):
		create = kwargs.pop('create', None)
		recsvec = kwargs.pop('recsvec', [])
		if create:
			super(UserProfile, self).save(*args, **kwargs)
		elif len(recsvec) != 0:
			self.lastrecs = json.dumps(recsvec.tolist())
			super(UserProfile, self).save(*args, **kwargs)
		else:

			super(UserProfile, self).save(*args, **kwargs)


class AnimeRated(models.Model):

	userid = models.IntegerField()#
	AnimeID = models.IntegerField(default=-1) #
	value = models.IntegerField() #

	def __str__(self):
		return self.AnimeID


class AnimeData(models.Model):
	title = models.TextField(max_length=2000) # Anime Name
	animeID = models.IntegerField() # AnimeID
	genre=models.CharField(max_length=200)
	type=models.CharField(max_length=200)
	status=models.CharField(max_length=200)
	duration=models.CharField(max_length=200)
	studio=models.CharField(max_length=200)
	description = models.TextField(max_length=4000) # Anime Description
	rating=models.FloatField()
	imagepath=models.CharField(max_length=200)
	category=models.CharField(max_length=200)
	def __str__(self):
		return self.title






