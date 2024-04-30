from instagrapi import Client
from tiktok_uploader.upload import upload_video, upload_videos
from youtube_uploader_selenium import YoutubeUploader
import requests
import os
import random
import sys
import time
import httplib2

class InstagramUploader:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.client = Client()
        
    def login(self):
        self.client.login(self.username, self.password)
        
    def upload_post(self, file_path, caption='', type='photo'):
        if type == 'photo':
            self.client.photo_upload(file_path, caption)
        elif type == 'video':
            self.client.video_upload(file_path, caption)
        else:
            print('Type must be "photo" or "video".')

def upload_tiktok(video_path, description, cookies="cookies.txt", comment=False, stitch=False, duet=False, proxy=None, schedule=None):
    upload_video(filename=video_path, description=description, cookies=cookies, comment=comment, stitch=stitch, duet=duet, proxy=proxy, schedule=schedule)
    print(f"Uploaded {video_path} to tiktok")

class YoutubeUploader:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
    def upload(self, video_args=None, file_path=None):
        uploader = YoutubeUploader(self.client_id, self.client_secret)
        uploader.authenticate()
        uploader.upload(file_path, options=video_args)
        uploader.close()
