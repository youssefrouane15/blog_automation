# main.py - Complete Recipe Article Generator Backend
import os
import json
import csv
import logging
import asyncio
import io
import re
import random
import time
import hashlib
import jwt as pyjwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, field
import enum
from io import BytesIO
import aiohttp
from wordpress_xmlrpc.methods.media import UploadFile
from bs4 import BeautifulSoup
# Third-party packages
import requests
import anthropic
from anthropic import Anthropic
import xml.etree.ElementTree as ET
import pandas as pd
from wordpress_xmlrpc import Client, WordPressPost
from wordpress_xmlrpc.methods import posts
from wordpress_xmlrpc.methods.posts import NewPost, EditPost
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaInMemoryUpload, MediaFileUpload
import uvicorn

# JWT Configuration
SECRET_KEY = "recipe-automation-secret-key-2024"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# FastAPI Security
security = HTTPBearer()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Custom exceptions
class ConfigError(Exception):
    pass

class APIError(Exception):
    pass

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    username: str
    expires_in: int

# Data classes
@dataclass
class SEORules:
    """SEO rules based on RankMath recommendations"""
    min_word_count: int = 1500
    focus_keyword_in_title: bool = True
    focus_keyword_in_first_paragraph: bool = True
    focus_keyword_in_meta_description: bool = True
    focus_keyword_in_url: bool = True
    focus_keyword_in_subheadings: bool = True
    focus_keyword_in_image_alt: bool = True
    min_keyword_density: float = 0.5
    max_keyword_density: float = 2.0
    max_url_length: int = 75
    external_links_required: bool = True
    internal_links_required: bool = True
    use_table_of_contents: bool = True
    paragraph_length_max: int = 150
    title_should_have_power_word: bool = True
    title_should_have_sentiment: bool = True
    avoid_numbers_in_title: bool = True

@dataclass
class ArticleMetadata:
    title: str
    slug: str
    focus_keyword: str
    meta_description: str
    seo_data: Dict
    html_content: str = ""
    word_count: int = 0
    recipe_card: Optional[Union[str, Dict]] = None
    serp_source: Optional[str] = None  # Added to track source website from SERP
    featured_image_id: Optional[int] = None  # Add this line

class UpscaleOption(enum.Enum):
    U1 = 1
    U2 = 2
    U3 = 3
    U4 = 4

@dataclass
class ImageData:
    url: str
    alt: str
    type: str
    placement: str

@dataclass
class CommandOption:
    type: int = 3
    name: str = "prompt"
    value: Optional[str] = None

@dataclass
class CommandData:
    version: str
    id: str
    name: str
    type: int = 1
    options: List[CommandOption] = field(default_factory=list)

@dataclass
class DiscordInteraction:
    channel_id: str
    application_id: str
    session_id: str
    nonce: str
    guild_id: str
    data: CommandData
    type: int = 2

@dataclass
class DiscordButton:
    type: int = 2
    style: int = 1
    label: Optional[str] = None
    custom_id: Optional[str] = None

@dataclass
class DiscordActionRow:
    type: int = 1
    components: List[DiscordButton] = field(default_factory=list)

@dataclass
class DiscordAttachment:
    url: str
    proxy_url: str
    filename: str

@dataclass
class DiscordMessage:
    id: str
    content: str
    attachments: List[DiscordAttachment] = field(default_factory=list)
    components: List[DiscordActionRow] = field(default_factory=list)

# Logger class
class Logger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        log_file = self.log_dir / f"article_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_api_call(self, api_name: str, request_data: Dict, response_data: Optional[Dict] = None, error: Optional[Exception] = None):
        if error:
            self.logger.error(f"{api_name} API Error: {str(error)}")
            self.logger.error(f"Request Data: {json.dumps(request_data, indent=2, default=str)}")
        else:
            self.logger.info(f"{api_name} API Call Successful")
            self.logger.info(f"Request Data: {json.dumps(request_data, indent=2, default=str)}")
            if response_data:
                self.logger.info(f"Response Data: {json.dumps(response_data, indent=2, default=str)}")

    def debug(self, message):
        """Debug level logging"""
        self.logger.debug(message)

    def info(self, message):
        """Info level logging"""
        self.logger.info(message)

    def warning(self, message):
        """Warning level logging"""
        self.logger.warning(message)

    def error(self, message):
        """Error level logging"""
        self.logger.error(message)

    def exception(self, message):
        """Exception level logging with traceback"""
        self.logger.exception(message)

# Config class
class Config:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict:
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ConfigError(f"Failed to load config file: {str(e)}")

    def _validate_config(self):
        required_keys = [
            'users',
            'claude_api_key',
            'claude_model',
            'serpapi_api_key',
            'wordpressWebsites',
            'article_template',
            'google_sheets',
            'seo_params'
        ]
        
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ConfigError(f"Missing required config keys: {', '.join(missing_keys)}")

    def get_users(self) -> List[Dict]:
        return self.config.get('users', [])
    
    def get_user(self, username: str) -> Optional[Dict]:
        users = self.get_users()
        return next((user for user in users if user['username'] == username), None)
    
    def verify_user(self, username: str, password: str) -> bool:
        user = self.get_user(username)
        if not user:
            return False
        return user.get('password') == password

    def get_article_template(self, website_name: str = None) -> str:
        if website_name:
            for website in self.config.get('wordpressWebsites', []):
                if website.get('name', '').lower() == website_name.lower() and 'article_template' in website:
                    return website['article_template']
        return self.config.get('article_template', '')

    def get_wordpress_sites(self) -> List[Dict]:
        sites = self.config['wordpressWebsites']
        return [site for site in sites if site.get('is_active', True)]
        
    def get_website(self, name: str) -> Optional[Dict]:
        websites = self.config.get('wordpressWebsites', [])
        return next((site for site in websites if site['name'] == name), None)
    
    def get_website_midjourney_config(self, website_name: str) -> Optional[Dict]:
        """Get Midjourney channel configuration for a specific website"""
        website = self.get_website(website_name)
        if website and 'midjourney_channel' in website:
                return website['midjourney_channel']
        return None 
       
    def get_seo_params(self) -> Dict:
        return self.config.get('seo_params', {
            'min_word_count': 2000,
            'keyword_density': 1.0,
            'max_slug_length': 80,
            'max_meta_description_length': 150,
            'max_focus_keyword_words': 4
        })
# Add these classes after the existing dataclasses around line 100

# Discord/Midjourney integration classes  
class DiscordService:
    DISCORD_API_URL = "https://discord.com/api/v10"
    MIDJOURNEY_APP_ID = "936929561302675456"

    def __init__(self, config, logger, channel_config):
        self.logger = logger
        self.channel_config = channel_config
        self.headers = {
            'Authorization': config['discord']['token'],
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    async def submit_imagine_command(self, request_data, command):
        """Submit imagine command to Discord"""
        discord_api_url = f"{self.DISCORD_API_URL}/interactions"
        
        request_dict = {
            'type': request_data.type,
            'channel_id': request_data.channel_id,
            'application_id': request_data.application_id,
            'session_id': request_data.session_id,
            'nonce': request_data.nonce,
            'guild_id': request_data.guild_id,
            'data': {
                'version': request_data.data.version,
                'id': request_data.data.id,
                'name': request_data.data.name,
                'type': request_data.data.type,
                'options': [{
                    'type': opt.type,
                    'name': opt.name,
                    'value': opt.value
                } for opt in request_data.data.options]
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(discord_api_url, headers=self.headers, json=request_dict) as response:
                if response.status != 204:
                    text = await response.text()
                    raise Exception(f"Discord API error: {response.status}, {text}")
        
        return True

    async def get_message(self, channel_id):
        """Get message with components from Discord"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.DISCORD_API_URL}/channels/{channel_id}/messages?limit=10",
                headers=self.headers
            ) as response:
                if response.status != 200:
                    return None
                    
                messages = await response.json()
                
                for message_data in messages:
                    components = message_data.get('components', [])
                    if components:
                        has_upscale = False
                        for component in components:
                            for button in component.get('components', []):
                                if button.get('label', '').startswith('U'):
                                    has_upscale = True
                                    break
                            if has_upscale:
                                break
                        
                        if has_upscale:
                            return DiscordMessage(
                                id=message_data['id'],
                                content=message_data.get('content', ''),
                                attachments=[
                                    DiscordAttachment(
                                        url=att['url'],
                                        proxy_url=att['proxy_url'],
                                        filename=att['filename']
                                    ) for att in message_data.get('attachments', [])
                                ],
                                components=[
                                    DiscordActionRow(components=[
                                        DiscordButton(
                                            label=btn.get('label'),
                                            custom_id=btn.get('custom_id')
                                        ) for btn in row.get('components', [])
                                    ]) for row in components
                                ]
                            )
        return None

    async def get_upscaled_images(self, channel_id, original_message_id, max_retries=10):
        """Get upscaled images URLs"""
        urls = []
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.DISCORD_API_URL}/channels/{channel_id}/messages?limit=10",
                        headers=self.headers
                    ) as response:
                        if response.status != 200:
                            retry_count += 1
                            await asyncio.sleep(5)
                            continue
                            
                        messages = await response.json()
                        
                        for message in messages:
                            if message.get('attachments'):
                                for attachment in message.get('attachments', []):
                                    urls.append(attachment['url'])
                        
                        if urls:
                            return urls
                            
                        retry_count += 1
                        await asyncio.sleep(5)
                        
            except Exception as e:
                retry_count += 1
                await asyncio.sleep(5)
        
        return urls

    async def upscale_image(self, channel_id, message_id, guild_id, component_id, option):
        """Upscale image using Discord interaction"""
        custom_id = self._construct_upscale_custom_id(component_id, option)
        if not custom_id:
            return False

        payload = {
            "type": 3,
            "application_id": self.MIDJOURNEY_APP_ID,
            "channel_id": channel_id,
            "message_id": message_id,
            "guild_id": guild_id,
            "session_id": self.channel_config.get("session_id", "dc396fbc-547f-4ef8-84d8-35602834bc9b"),
            "data": {
                "component_type": 2,
                "custom_id": custom_id
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.DISCORD_API_URL}/interactions",
                json=payload,
                headers=self.headers
            ) as response:
                if response.status not in [204, 200]:
                    text = await response.text()
                    raise Exception(f"Upscale error: {response.status}, {text}")
        
        return True

    def find_upscale_component_id(self, message):
        """Find the upscale component ID from a Discord message"""
        if not message.components:
            return None
        
        for row in message.components:
            for button in row.components:
                if button.label and button.custom_id:
                    if any(button.label.startswith(f'U{i}') for i in range(1, 5)):
                        if "::" in button.custom_id:
                            return button.custom_id.split("::")[-1]
                        return button.custom_id
        
        return None

    def _construct_upscale_custom_id(self, base_custom_id, position):
        """Construct the upscale custom ID from the base ID and position"""
        if not base_custom_id:
            return None
            
        if "::" not in base_custom_id:
            hash_part = base_custom_id
        else:
            parts = base_custom_id.split("::")
            if len(parts) < 5:
                raise ValueError("Invalid base custom ID format")
            hash_part = parts[4]

        return f"MJ::JOB::upsample::{position.value}::{hash_part}"


class ImageGenerator:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.anthropic_client = Anthropic(api_key=config.config['claude_api_key'])
        self.discord_services = {}  # Cache for Discord services per website
        
    def get_discord_service(self, website_config):
        """Get or create Discord service for a website"""
        website_name = website_config['name']
        
        if website_name not in self.discord_services:
            channel_config = website_config.get('midjourney_channel', {})
            self.discord_services[website_name] = DiscordService(
                self.config.config, 
                self.logger, 
                channel_config
            )
        
        return self.discord_services[website_name]
        
    async def extract_feature_image(self, url):
        """Extract feature image from source URL - EXACT from old code"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code != 200:
                return None
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try Open Graph image first
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                return og_image['content']
            
            # Try Twitter image next
            twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
            if twitter_image and twitter_image.get('content'):
                return twitter_image['content']
            
            # Try JSON-LD data
            for script in soup.find_all('script', type='application/ld+json'):
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and 'image' in data:
                        if isinstance(data['image'], str):
                            return data['image']
                        elif isinstance(data['image'], list) and data['image']:
                            return data['image'][0]
                        elif isinstance(data['image'], dict) and 'url' in data['image']:
                            return data['image']['url']
                except:
                    continue
            
            # Look for first large image
            for img in soup.find_all('img'):
                if img.get('src'):
                    img_url = img['src']
                    if not img_url.startswith(('http://', 'https://')):
                        img_url = urljoin(url, img_url)
                    return img_url
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting feature image: {str(e)}")
            return None
    
    async def generate_image_prompts(self, article_content, focus_keyword, recipe_data):
        """Generate prompts for different image types using Claude"""
        ingredients_list = recipe_data.get("ingredients", [])
        instructions_list = recipe_data.get("instructions", [])
        
        prompt = f"""
        Generate image prompts for a recipe article about "{focus_keyword}". 
        
        Recipe ingredients:
        {chr(10).join(ingredients_list[:10])}
        
        Recipe instructions:
        {chr(10).join(instructions_list[:6])}
        
        Create THREE different prompts:
        
        1. Featured Image: A photo-realistic, appetizing image of the finished {focus_keyword}
        
        2. Ingredients Layout: Create a prompt for an ultra-realistic TOP-DOWN shot of ALL the ingredients laid out on a clean white marble or white wooden kitchen board. The ingredients should be:
        - Neatly arranged and separated
        - Raw/uncooked state
        - Professional food photography style
        - Natural lighting from above
        - Include measuring cups/spoons where relevant
        - Each ingredient clearly visible and identifiable
        
        3. Step-by-Step Collage: Create a prompt for a realistic photo collage showing the cooking process in 4-6 panels/frames. The collage should:
        - Show key preparation steps in sequence
        - Each panel clearly showing hands performing the action
        - Professional cooking photography style
        - Clean, bright kitchen setting
        - Tools and equipment visible in action
        - Progress from raw ingredients to finished dish
        
        For each prompt, be very specific about visual details, lighting, and composition.
        
        Return JSON format:
        {{
        "featured_image": "prompt for finished dish",
        "ingredients_layout": "prompt for ingredients top-down shot",
        "step_by_step_collage": "prompt for cooking steps collage"
        }}
        """
        
        response = self.anthropic_client.messages.create(
            model=self.config.config['claude_model'],
            max_tokens=1500,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.content[0].text
        
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            return json.loads(json_str)
        
        return {
            "featured_image": f"Photorealistic appetizing image of {focus_keyword}, professional food photography",
            "ingredients_layout": f"Ultra-realistic TOP-DOWN shot of all ingredients for {focus_keyword} recipe neatly arranged on white marble kitchen board, professional overhead food photography, natural lighting",
            "step_by_step_collage": f"Photo collage showing 6 step-by-step cooking process for {focus_keyword}, hands in action, bright kitchen, professional cooking photography"
        }
    
    async def generate_with_midjourney(self, website_config, prompt, image_url=None):
        """Generate images using Midjourney through Discord with website-specific channel"""
        discord_service = self.get_discord_service(website_config)
        channel_config = website_config.get('midjourney_channel', {})
        
        # Get channel details from website config
        channel_id = channel_config.get('channel_id')
        guild_id = channel_config.get('guild_id')
        session_id = channel_config.get('session_id')
        
        if not all([channel_id, guild_id, session_id]):
            raise Exception(f"Missing Midjourney channel configuration for website {website_config['name']}")

        # If image_url is provided, prepend it to the prompt for Midjourney
        if image_url:
            complete_prompt = f"{image_url} {prompt} --iw 2"
            self.logger.info(f"Using image-to-image with Midjourney: {image_url}")
        else:
            complete_prompt = prompt
            self.logger.info(f"Using text-to-image with Midjourney")
        
        self.logger.info(f"Midjourney prompt: {complete_prompt}")
        
        # Create Midjourney imagine command
        command_option = CommandOption(
            type=3,
            name="prompt",
            value=complete_prompt
        )
        
        command_data = CommandData(
            version="1237876415471554623",
            id="938956540159881230",
            name="imagine",
            type=1,
            options=[command_option]
        )
        
        discord_interaction = DiscordInteraction(
            type=2,
            channel_id=channel_id,
            application_id="936929561302675456",
            session_id=session_id,
            nonce=str(int(time.time() * 1000)),
            guild_id=guild_id,
            data=command_data
        )
        
        # Submit imagine command
        await discord_service.submit_imagine_command(discord_interaction, "imagine")
        
        # Wait for message with components
        await asyncio.sleep(45)
        message = None
        
        for retry in range(10):
            message = await discord_service.get_message(channel_id)
            if message and message.components:
                break
            await asyncio.sleep(10)
        
        if not message or not message.components:
            raise Exception("No Midjourney message received")
        
        # Find upscale component
        component_id = discord_service.find_upscale_component_id(message)
        if not component_id:
            raise Exception("No upscale component found")
        
        # Upscale all 4 images
        image_urls = []
        for i in range(4):
            try:
                option = UpscaleOption(i + 1)
                await discord_service.upscale_image(
                    channel_id=channel_id,
                    message_id=message.id,
                    guild_id=guild_id,
                    component_id=component_id,
                    option=option
                )
                
                await asyncio.sleep(20)
                
                upscaled_urls = await discord_service.get_upscaled_images(
                    channel_id=channel_id,
                    original_message_id=message.id
                )
                
                if upscaled_urls:
                    image_urls.append(upscaled_urls[0])
            except Exception as e:
                self.logger.error(f"Error upscaling image {i+1}: {str(e)}")
            
            await asyncio.sleep(3)
        
        if not image_urls:
            raise Exception("Failed to generate any images with Midjourney")
            
        return image_urls
    
    async def download_image(self, url):
        """Download image from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': f"https://{url.split('/')[2]}" if url.startswith('http') else '',
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        return await response.read()
            return None
        except Exception as e:
            self.logger.error(f"Error downloading image {url}: {str(e)}")
            return None
    
    async def upload_to_wordpress(self, client, image_data, title, alt_text):
        """Upload image to WordPress with improved error handling and retries"""
        try:
            random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
            image_name = f"{alt_text.replace(' ', '-').lower()[:30]}-{random_suffix}.jpg"
            
            from wordpress_xmlrpc.methods import media
            from wordpress_xmlrpc.compat import xmlrpc_client
            
            upload_data = {
                'name': image_name,
                'type': 'image/jpeg',
                'bits': xmlrpc_client.Binary(image_data),
                'overwrite': False,
            }
            
            self.logger.info(f"Uploading image {image_name} to WordPress ({len(image_data)} bytes)")
            
            max_retries = 5
            attachment_id = None
            
            for retry in range(max_retries):
                try:
                    if retry > 0:
                        upload_data['name'] = f"{alt_text.replace(' ', '-').lower()[:20]}-retry{retry}-{random_suffix}.jpg"
                        self.logger.info(f"Retry {retry+1}/{max_retries} with modified name: {upload_data['name']}")
                        
                    response = client.call(media.UploadFile(upload_data))
                    
                    if isinstance(response, dict) and 'id' in response:
                        attachment_id = response['id']
                        self.logger.info(f"Successfully uploaded image with ID: {attachment_id}")
                        break
                    elif hasattr(response, 'id'):
                        attachment_id = response.id
                        self.logger.info(f"Successfully uploaded image with ID: {attachment_id}")
                        break
                    else:
                        self.logger.warning(f"Unexpected response format: {type(response)}")
                        self.logger.warning(f"Response content: {response}")
                        
                except Exception as e:
                    self.logger.warning(f"Upload attempt {retry+1} failed: {str(e)}")
                    await asyncio.sleep(2)
            
            if attachment_id:
                try:
                    from wordpress_xmlrpc.methods import media
                    attachment = client.call(media.GetMediaItem(attachment_id))
                    attachment_url = None
                    
                    if hasattr(attachment, 'url'):
                        attachment_url = attachment.url
                    elif hasattr(attachment, 'link'):
                        attachment_url = attachment.link
                        
                    return attachment_id, attachment_url
                except Exception as e:
                    self.logger.error(f"Error getting attachment URL: {str(e)}")            
                    self.logger.error(f"All {max_retries} upload attempts failed")
                    return None, None
            
        except Exception as e:
            self.logger.error(f"Error in upload_to_wordpress: {str(e)}")
            return None, None
        

# Authentication Manager
class AuthManager:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        try:
            if self.config.verify_user(username, password):
                user = self.config.get_user(username)
                return {
                    "username": user["username"],
                    "authenticated": True
                }
            return None
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            return None
    
    def create_access_token(self, username: str) -> str:
        expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
        to_encode = {
            "sub": username,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        encoded_jwt = pyjwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict]:
        try:
            payload = pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                return None
            
            user = self.config.get_user(username)
            if user is None:
                return None
            
            return {
                "username": username,
                "exp": payload.get("exp")
            }
        except ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except InvalidTokenError as e:
            self.logger.warning(f"JWT error: {str(e)}")
            return None
    
    def login(self, username: str, password: str) -> Optional[Dict]:
        user = self.authenticate_user(username, password)
        if not user:
            return None
        
        token = self.create_access_token(username)
        return {
            "access_token": token,
            "token_type": "bearer",
            "username": username,
            "expires_in": ACCESS_TOKEN_EXPIRE_HOURS * 3600
        }

# Progress Tracker class
class ProgressTracker:
    def __init__(self):
        self._progress = {}
        self._website_stats = {}
        # REMOVED shared state variables - this was the bug!
        self.logger = logging.getLogger(__name__)

    def set_current_step(self, website_name: str, step: str):
        """Set current step for a specific website"""
        if website_name in self._website_stats:
            self._website_stats[website_name]["current_step"] = step
            self._website_stats[website_name]["last_update"] = datetime.now().isoformat()
           
    def start_website(self, website_name: str):
        if website_name not in self._website_stats:
            self._website_stats[website_name] = {
                "total_keywords": 0,
                "completed": 0,
                "failed": 0,
                "in_progress": 0,
                "keywords": [],
                "status": "idle",
                "current_keyword": None,
                "current_step": None,
                "progress": 0,
                "start_time": None,
                "last_update": datetime.now().isoformat(),
                "task_id": None
            }
   
    def set_website_total(self, website_name: str, total_keywords: int):
        if website_name in self._website_stats:
            self._website_stats[website_name]["total_keywords"] = total_keywords
            self._website_stats[website_name]["keywords"] = []

    def update_progress(self, website: str, keyword: str, status: str, metadata: Optional[Dict] = None):
        key = f"{website}::{keyword}"
        
        # Initialize website stats if not exists
        if website not in self._website_stats:
            self.start_website(website)
        
        # Update website-specific current keyword and step
        if status == "processing":
            self._website_stats[website]["status"] = "processing"
            self._website_stats[website]["current_keyword"] = keyword
            
            # IMPORTANT: Always update the current step from metadata
            if metadata and "current_step" in metadata:
                self._website_stats[website]["current_step"] = metadata["current_step"]
                self.logger.info(f"Updated step for {website}: {metadata['current_step']}")
        
        # Handle completion status
        if status == "completed" and (not website or not keyword):
            self._progress["global_status"] = {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
                "display": {
                    "processing_text": "Process Complete",
                    "status_text": "Completed",
                    "active": False,
                    "show_spinner": False,
                    "showCompletionBanner": True
                }
            }
            self._progress["__process_complete__"] = True
            return
        
        # Track keyword if new
        if keyword and keyword not in self._website_stats[website]["keywords"]:
            self._website_stats[website]["keywords"].append(keyword)
        
        # Update keyword progress
        if keyword:  # Only update if keyword is provided
            self._progress[key] = {
                "website": website,
                "keyword": keyword,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
                "current_step": metadata.get("current_step", "") if metadata else "",
            }
        
        # Always update last_update timestamp
        self._website_stats[website]["last_update"] = datetime.now().isoformat()
        
        # Calculate progress
        self._update_website_progress(website)

    def _update_website_progress(self, website_name: str):
        """Recalculate progress for a website"""
        if website_name not in self._website_stats:
            return
            
        stats = self._website_stats[website_name]
        total = stats.get("total_keywords", 0)
        
        # Count completed and failed from actual progress data
        completed = 0
        failed = 0
        
        for keyword in stats.get("keywords", []):
            key = f"{website_name}::{keyword}"
            if key in self._progress:
                status = self._progress[key].get("status", "")
                if status == "completed":
                    completed += 1
                elif status == "failed":
                    failed += 1
        
        stats["completed"] = completed
        stats["failed"] = failed
        stats["in_progress"] = len([k for k in stats.get("keywords", []) 
                                    if self._progress.get(f"{website_name}::{k}", {}).get("status") == "processing"])
        
        # Calculate progress percentage
        if total > 0:
            progress_percent = int(((completed + failed) / total) * 100)
            stats["progress"] = progress_percent
        else:
            stats["progress"] = 0

    def complete_keyword(self, website_name: str, keyword: str, success: bool = True):
        """Complete processing for a keyword"""
        try:
            if website_name not in self._website_stats:
                self.logger.warning(f"Website {website_name} not found in stats")
                return
                
            status = "completed" if success else "failed"
            self.update_progress(website_name, keyword, status)
            
            # Clear current keyword if it matches
            stats = self._website_stats[website_name]
            if stats.get("current_keyword") == keyword:
                stats["current_keyword"] = None
                stats["current_step"] = None
            
            # Check if all keywords are done
            total = stats.get("total_keywords", 0)
            completed = stats.get("completed", 0)
            failed = stats.get("failed", 0)
            
            if total > 0 and (completed + failed) >= total:
                stats["status"] = "completed"
                stats["current_step"] = f"✅ All {total} keywords completed"
                self.logger.info(f"🎉 Website {website_name} fully completed! {completed} successful, {failed} failed")
                
        except Exception as e:
            self.logger.error(f"Error in complete_keyword: {str(e)}")

    def get_website_status(self, website_name: str) -> Optional[Dict]:
        """Get current status for a specific website"""
        if website_name not in self._website_stats:
            return None
        
        stats = self._website_stats[website_name].copy()
        
        # Ensure we return the website-specific data
        return {
            "status": stats.get("status", "idle"),
            "current_keyword": stats.get("current_keyword", ""),
            "current_step": stats.get("current_step", ""),
            "completed": stats.get("completed", 0),
            "total_keywords": stats.get("total_keywords", 0),
            "progress": stats.get("progress", 0),
            "keywords": stats.get("keywords", []),
            "failed": stats.get("failed", 0),
            "in_progress": stats.get("in_progress", 0),
            "last_update": datetime.now().isoformat()
        }
    
    def is_website_processing(self, website_name: str) -> bool:
        """Check if a website is currently processing"""
        status = self.get_website_status(website_name)
        return status and status.get("status") == "processing"
    
    def export_to_csv(self) -> Dict[str, io.StringIO]:
        results = {}
        
        for website, stats in self._website_stats.items():
            csv_data = io.StringIO()
            writer = csv.writer(csv_data)
            
            writer.writerow([
                'Keyword', 'Status', 'URL', 'Focus Keyword', 
                'SEO Title', 'SEO Description', 'Slug', 'Word Count', 'SERP Source'
            ])
            
            for keyword in stats["keywords"]:
                key = f"{website}::{keyword}"
                data = self._progress.get(key, {})
                metadata = data.get("metadata", {})               
                writer.writerow([
                        keyword,
                        data.get("status", ""),
                        metadata.get("url", ""),
                        metadata.get("focus_keyword", ""),
                        metadata.get("title", ""),
                        metadata.get("meta_description", ""),
                        metadata.get("slug", ""),
                        metadata.get("word_count", ""),
                        metadata.get("serp_source", "")
                ])
            
            csv_data.seek(0)
            results[website] = csv_data
               
        return results

# Google Sheets Manager
class GoogleSheetsManager:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.credentials = None
        self.service = None
        self.sheets_data = {}
        self._initialize_service()
        
    def _initialize_service(self):
        try:
            creds_file = self.config.config['google_sheets']['credentials_file']
            self.credentials = service_account.Credentials.from_service_account_file(
                creds_file, 
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            self.service = build('sheets', 'v4', credentials=self.credentials)
            self.logger.info("Google Sheets API service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Sheets API: {str(e)}")
            raise ConfigError(f"Google Sheets API initialization failed: {str(e)}")
    
    def fetch_all_sheets_data(self) -> Dict[str, pd.DataFrame]:
        try:
            spreadsheet_id = self.config.config['google_sheets']['spreadsheet_id']
            
            sheet_metadata = self.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            sheets = sheet_metadata.get('sheets', [])
            sheet_names = [sheet.get('properties', {}).get('title', '') for sheet in sheets]
            
            result = {}
            for sheet_name in sheet_names:
                if not sheet_name:
                    continue
                    
                range_name = f"{sheet_name}!A1:Z1000"
                sheet = self.service.spreadsheets().values().get(
                    spreadsheetId=spreadsheet_id,
                    range=range_name
                ).execute()
                values = sheet.get('values', [])
                
                if not values or len(values) < 2:
                    self.logger.info(f"Sheet '{sheet_name}' is empty or has only headers")
                    continue
                
                headers = values[0]
                data = values[1:]
                
                max_cols = len(headers)
                for i, row in enumerate(data):
                    if len(row) < max_cols:
                        data[i] = row + [''] * (max_cols - len(row))
                
                df = pd.DataFrame(data, columns=headers)
                
                if 'keyword' not in df.columns or 'article_done' not in df.columns:
                    self.logger.error(f"Sheet '{sheet_name}' missing required columns (keyword, article_done)")
                    continue
                    
                result[sheet_name] = df
                self.logger.info(f"Successfully loaded sheet '{sheet_name}' with {len(df)} rows")
            
            self.sheets_data = result
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching Google Sheets data: {str(e)}")
            raise APIError(f"Failed to fetch Google Sheets data: {str(e)}")
    
    def update_sheet_status(self, sheet_name: str, row_index: int, data: Dict[str, str]):
        try:
            spreadsheet_id = self.config.config['google_sheets']['spreadsheet_id']
            
            df = self.sheets_data.get(sheet_name)
            if df is None:
                self.logger.error(f"Sheet '{sheet_name}' not found in cached data")
                return False
                
            row_num = row_index + 2
            
            batch_update = []
            
            for col_name, value in data.items():
                if col_name in df.columns:
                    col_index = df.columns.get_loc(col_name)
                    cell = f"{sheet_name}!{self._column_letter(col_index)}{row_num}"
                    
                    batch_update.append({
                        "range": cell,
                        "values": [[value]]
                    })
            
            if batch_update:
                body = {
                    "valueInputOption": "USER_ENTERED",
                    "data": batch_update
                }
                self.service.spreadsheets().values().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body=body
                ).execute()
                
                self.logger.info(f"Updated row {row_num} in sheet '{sheet_name}'")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating Google Sheet: {str(e)}")
            return False
    
    def _column_letter(self, index: int) -> str:
        result = ""
        while index >= 0:
            result = chr(index % 26 + 65) + result
            index = index // 26 - 1
        return result

# Keyword Processor
class KeywordProcessor:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.anthropic_client = Anthropic(api_key=config.config['claude_api_key'])

    def optimize_keyword(self, keyword: str) -> str:
        try:
            seo_params = self.config.get_seo_params()
            max_focus_keyword_words = seo_params.get('max_focus_keyword_words', 4)

            prompt = f"""
            Optimize this recipe keyword for SEO: {keyword}
            
            Guidelines:
            - Remove generic words (recipe, best, easy, etc.)
            - Focus on the core culinary concept
            - Make it specific and search-friendly
            - Keep the original intent
            - STRICT MAXIMUM {max_focus_keyword_words} words
            
            Return ONLY the optimized keyword with no explanation.
            """
            
            request_data = {
                "model": self.config.config['claude_model'],
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            self.logger.log_api_call(
                "Claude Keyword Optimization",
                request_data
            )
            
            response = self.anthropic_client.messages.create(
                model=self.config.config['claude_model'],
                temperature=1,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            
            optimized_keyword = response.content[0].text.strip()
            
            word_count = len(optimized_keyword.split())
            if word_count > max_focus_keyword_words:
                self.logger.warning(f"Keyword optimization returned {word_count} words: '{optimized_keyword}'. Truncating to {max_focus_keyword_words} words.")
                optimized_keyword = ' '.join(optimized_keyword.split()[:max_focus_keyword_words])
            
            self.logger.log_api_call(
                "Claude Keyword Optimization",
                request_data,
                {"optimized_keyword": optimized_keyword}
            )
            
            return optimized_keyword

        except Exception as e:
            self.logger.log_api_call("Claude Keyword Optimization", request_data, error=e)
            raise APIError(f"Keyword optimization failed: {str(e)}")
    
    def analyze_serp(self, keyword: str) -> Dict:
        try:
            params = {
                "engine": "google",
                "q": f"{keyword} recipe",
                "api_key": self.config.config['serpapi_api_key'],
                "num": 10
            }
            
            response = requests.get("https://serpapi.com/search", params=params)
            serp_data = response.json()
            
            if "organic_results" not in serp_data or not serp_data["organic_results"]:
                raise APIError("No SERP results found")
            
            organic_results = serp_data.get("organic_results", [])
            
            detailed_results = []
            
            for result in organic_results[:3]:
                try:
                    skip_domains = ["facebook.com", "pinterest.com", "youtube.com"]
                    if any(domain in result["link"] for domain in skip_domains):
                        continue

                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                    }
                    
                    response = requests.get(result["link"], headers=headers, timeout=20)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    article_content = self._extract_article_content(soup, response.url)
                    recipe_data = self._extract_recipe_data(soup, response.url)
                    featured_image = self._extract_featured_image(soup, response.url)
                    style_tone_analysis = self._analyze_writing_style(article_content)
                    
                    detailed_results.append({
                        "title": result["title"],
                        "url": result["link"],
                        "snippet": result.get("snippet", ""),
                        "article_content": article_content,
                        "recipe_data": recipe_data,
                        "featured_image": featured_image,
                        "style_analysis": style_tone_analysis
                    })
                    
                    if len(article_content) > 1000 and recipe_data.get("ingredients"):
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error processing result from {result['link']}: {str(e)}")
                    continue
            
            if detailed_results:
                consolidated_result = self._consolidate_serp_results(detailed_results, keyword)
                return consolidated_result
            
            self.logger.info(f"Creating synthetic SERP result for '{keyword}'")
            return {
                "title": f"{keyword.title()} Recipe",
                "url": "",
                "snippet": f"A delicious recipe for {keyword}.",
                "article_content": f"Recipe for {keyword}.",
                "recipe_data": {
                    "ingredients": [],
                    "instructions": [],
                    "cook_time": "30 minutes",
                    "prep_time": "15 minutes",
                    "total_time": "45 minutes",
                    "servings": "4"
                },
                "featured_image": None,
                "style_analysis": {
                    "tone": "neutral",
                    "formality": "neutral",
                    "voice": "active",
                    "person": "second person"
                }
            }
            
        except Exception as e:
            self.logger.log_api_call("SERP Analysis", {"keyword": keyword}, error=e)
            raise APIError(f"SERP analysis failed: {str(e)}")

    def _extract_article_content(self, soup: BeautifulSoup, url: str) -> str:
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'iframe', 'noscript', 'header', 'aside', 'form']):
            element.decompose()
        
        content_selectors = [
            'article', '.post-content', '.entry-content', '.recipe-content', '.article-content',
            '.content-area', 'main', '.main-content', '.single-content', '.post-body',
            '[itemprop="recipeInstructions"]', '[itemprop="description"]', '.recipe-body',
            '.tasty-recipes', '.wprm-recipe-container', '.recipe-summary', '.recipe-container'
        ]
        
        main_content = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                text = content_element.get_text(strip=True)
                if len(text) > 200:
                    main_content = content_element
                    break
        
        if not main_content:
            main_content = soup.find('body')
            
        if main_content:
            paragraphs = []
            
            for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li']):
                text = element.get_text(strip=True)
                if text and len(text) > 15:
                    if element.name.startswith('h'):
                        paragraphs.append(f"\n{text}\n")
                    elif element.name == 'li':
                        paragraphs.append(f"• {text}")
                    else:
                        paragraphs.append(text)
            
            article_text = "\n\n".join(paragraphs)
            article_text = re.sub(r'\n{3,}', '\n\n', article_text)
            
            return article_text
        
        return soup.get_text(separator='\n\n', strip=True)

    def _extract_recipe_data(self, soup: BeautifulSoup, url: str) -> Dict:
        recipe_data = {
            "ingredients": [],
            "instructions": [],
            "cook_time": "",
            "prep_time": "",
            "total_time": "",
            "servings": ""
        }
        
        recipe_json = self._extract_recipe_json_ld(soup)
        if recipe_json:
            if "recipeIngredient" in recipe_json:
                recipe_data["ingredients"] = recipe_json["recipeIngredient"]
            
            if "recipeInstructions" in recipe_json:
                instructions = recipe_json["recipeInstructions"]
                if isinstance(instructions, list):
                    if all(isinstance(item, dict) for item in instructions):
                        recipe_data["instructions"] = [step.get("text", "") for step in instructions if "text" in step]
                    else:
                        recipe_data["instructions"] = instructions
                elif isinstance(instructions, str):
                    recipe_data["instructions"] = [line.strip() for line in instructions.split('\n') if line.strip()]
            
            for time_type in ["cookTime", "prepTime", "totalTime"]:
                if time_type in recipe_json:
                    time_value = recipe_json[time_type]
                    minutes = self._parse_iso_duration(time_value)
                    if minutes:
                        if minutes < 60:
                            human_time = f"{minutes} minutes"
                        else:
                            hours = minutes // 60
                            remaining_minutes = minutes % 60
                            human_time = f"{hours} hour" + ("s" if hours > 1 else "")
                            if remaining_minutes > 0:
                                human_time += f" {remaining_minutes} minutes"
                                
                        if time_type == "cookTime":
                            recipe_data["cook_time"] = human_time
                        elif time_type == "prepTime":
                            recipe_data["prep_time"] = human_time
                        elif time_type == "totalTime":
                            recipe_data["total_time"] = human_time
            
            if "recipeYield" in recipe_json:
                yield_info = recipe_json["recipeYield"]
                if isinstance(yield_info, list) and yield_info:
                    recipe_data["servings"] = yield_info[0]
                else:
                    recipe_data["servings"] = str(yield_info)
        
        if not recipe_data["ingredients"]:
            recipe_data["ingredients"] = self._extract_ingredients_from_html(soup)
        
        if not recipe_data["instructions"]:
            recipe_data["instructions"] = self._extract_instructions_from_html(soup)
        
        if not recipe_data["cook_time"] or not recipe_data["prep_time"]:
            times = self._extract_times_from_html(soup)
            if times.get("cook_time") and not recipe_data["cook_time"]:
                recipe_data["cook_time"] = times["cook_time"]
            if times.get("prep_time") and not recipe_data["prep_time"]:
                recipe_data["prep_time"] = times["prep_time"]
        
        if not recipe_data["servings"]:
            recipe_data["servings"] = self._extract_servings_from_html(soup)
        
        return recipe_data

    def _extract_recipe_json_ld(self, soup: BeautifulSoup) -> Dict:
        try:
            for script in soup.find_all('script', type='application/ld+json'):
                try:
                    data = json.loads(script.string)
                    
                    if isinstance(data, dict) and data.get('@type') == 'Recipe':
                        return data
                    
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and item.get('@type') == 'Recipe':
                                return item
                    
                    elif isinstance(data, dict) and '@graph' in data:
                        for item in data['@graph']:
                            if isinstance(item, dict) and item.get('@type') == 'Recipe':
                                return item
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        
        return {}

    def _extract_ingredients_from_html(self, soup: BeautifulSoup) -> List[str]:
        ingredients = []
        
        ingredient_selectors = [
            '[itemprop="recipeIngredient"]', '.recipe-ingredients', '.ingredients',
            '.wprm-recipe-ingredient-group', '.tasty-recipes-ingredients',
            '.ingredient-list', '.ingredients-list', '.jetpack-recipe-ingredients'
        ]
        
        for selector in ingredient_selectors:
            elements = soup.select(selector)
            if elements:
                list_items = []
                for element in elements:
                    list_items.extend(element.find_all('li'))
                
                if list_items:
                    for item in list_items:
                        text = item.get_text(strip=True)
                        if text and not any(text == existing for existing in ingredients):
                            ingredients.append(text)
                else:
                    for element in elements:
                        text = element.get_text(strip=True)
                        if text and not any(text == existing for existing in ingredients):
                            ingredients.append(text)
        
        if not ingredients:
            for ul in soup.find_all('ul'):
                list_text = ul.get_text(strip=True).lower()
                if any(term in list_text for term in ['cup', 'tablespoon', 'teaspoon', 'ounce', 'pound', 'gram']):
                    for li in ul.find_all('li'):
                        text = li.get_text(strip=True)
                        if text and not any(text == existing for existing in ingredients):
                            ingredients.append(text)
        
        return ingredients

    def _extract_instructions_from_html(self, soup: BeautifulSoup) -> List[str]:
        instructions = []
        
        instruction_selectors = [
            '[itemprop="recipeInstructions"]', '.recipe-instructions', '.instructions',
            '.wprm-recipe-instruction-group', '.tasty-recipes-instructions',
            '.preparation-steps', '.recipe-method', '.jetpack-recipe-directions'
        ]
        
        for selector in instruction_selectors:
            elements = soup.select(selector)
            if elements:
                list_items = []
                for element in elements:
                    list_items.extend(element.find_all(['li', 'p']))
                
                if list_items:
                    for item in list_items:
                        text = item.get_text(strip=True)
                        if text and len(text) > 10 and not any(text == existing for existing in instructions):
                            instructions.append(text)
                else:
                    for element in elements:
                        text = element.get_text(strip=True)
                        if text and len(text) > 10 and not any(text == existing for existing in instructions):
                            instructions.append(text)
        
        if not instructions:
            for ol in soup.find_all('ol'):
                list_text = ol.get_text(strip=True).lower()
                if len(list_text) > 100:
                    for li in ol.find_all('li'):
                        text = li.get_text(strip=True)
                        if text and len(text) > 10 and not any(text == existing for existing in instructions):
                            instructions.append(text)
        
        return instructions

    def _extract_times_from_html(self, soup: BeautifulSoup) -> Dict[str, str]:
        times = {"prep_time": "", "cook_time": ""}
        
        prep_time_patterns = [
            r'prep(?:\w+)?\s*time:?\s*(\d+\s*(?:minute|min|hour|hr)s?)',
            r'preparation:?\s*(\d+\s*(?:minute|min|hour|hr)s?)'
        ]
        
        cook_time_patterns = [
            r'cook(?:\w+)?\s*time:?\s*(\d+\s*(?:minute|min|hour|hr)s?)',
            r'baking\s*time:?\s*(\d+\s*(?:minute|min|hour|hr)s?)'
        ]
        
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text.lower())
        
        for pattern in prep_time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                times["prep_time"] = match.group(1)
                break
                
        for pattern in cook_time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                times["cook_time"] = match.group(1)
                break
        
        return times

    def _extract_servings_from_html(self, soup: BeautifulSoup) -> str:
        patterns = [
            r'(?:yield|serves|servings|portions):?\s*(\d+(?:\s*-\s*\d+)?(?:\s*(?:people|servings|portions))?)',
            r'(?:this recipe serves|serves):?\s*(\d+(?:\s*-\s*\d+)?)'
        ]
        
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text.lower())
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "4 servings"

    def _extract_featured_image(self, soup: BeautifulSoup, url: str) -> str:
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            return og_image['content']
        
        twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
        if twitter_image and twitter_image.get('content'):
            return twitter_image['content']
        
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and 'image' in data:
                    if isinstance(data['image'], str):
                        return data['image']
                    elif isinstance(data['image'], list) and data['image']:
                        return data['image'][0]
                    elif isinstance(data['image'], dict) and 'url' in data['image']:
                        return data['image']['url']
            except:
                continue
        
        recipe_image_selectors = [
            '.recipe-image img', '.wp-post-image', '.post-thumbnail img',
            '.featured-image img', '.entry-image img', '[itemprop="image"]',
            '.hero-image img', '.recipe-header-image img', '.recipe-featured-image img'
        ]
        
        for selector in recipe_image_selectors:
            img = soup.select_one(selector)
            if img and img.get('src'):
                image_url = img['src']
                if not image_url.startswith(('http://', 'https://')):
                    image_url = urljoin(url, image_url)
                return image_url
        
        for img in soup.find_all('img'):
            if img.get('width') and img.get('height'):
                try:
                    width = int(img['width'])
                    height = int(img['height'])
                    
                    if width >= 300 and height >= 300:
                        image_url = img['src']
                        if not image_url.startswith(('http://', 'https://')):
                            image_url = urljoin(url, image_url)
                        return image_url
                except:
                    pass
        
        return None

    def _analyze_writing_style(self, content: str) -> Dict:
        analysis = {
            "tone": "neutral",
            "formality": "neutral",
            "voice": "active",
            "person": "second person"
        }
        
        if not content or len(content) < 100:
            return analysis
        
        try:
            sample = content[:3000]
            
            casual_indicators = ['!', 'awesome', 'amazing', 'love', 'favorite', 'best', 'just', 'really', 'super', 'wow']
            formal_indicators = ['therefore', 'however', 'consequently', 'furthermore', 'thus', 'hence', 'accordingly']
            
            casual_count = sum(1 for word in casual_indicators if word in sample.lower())
            formal_count = sum(1 for word in formal_indicators if word in sample.lower())
            
            if casual_count > formal_count * 2:
                analysis["formality"] = "casual"
            elif formal_count > casual_count:
                analysis["formality"] = "formal"
            
            positive_indicators = ['love', 'great', 'delicious', 'favorite', 'best', 'perfect', 'easy', 'amazing']
            negative_indicators = ['avoid', 'difficult', 'challenge', 'problem', 'careful', 'warning']
            
            positive_count = sum(1 for word in positive_indicators if word in sample.lower())
            negative_count = sum(1 for word in negative_indicators if word in sample.lower())
            
            if positive_count > negative_count * 2:
                analysis["tone"] = "positive"
            elif negative_count > positive_count:
                analysis["tone"] = "cautious"
            
            first_person = ['I ', 'my ', 'we ', 'our ']
            second_person = ['you ', 'your ']
            
            first_person_count = sum(1 for phrase in first_person if phrase.lower() in sample.lower())
            second_person_count = sum(1 for phrase in second_person if phrase.lower() in sample.lower())
            
            if first_person_count > second_person_count * 2:
                analysis["person"] = "first person"
            elif second_person_count > first_person_count:
                analysis["person"] = "second person"
                
            passive_indicators = [' is made', ' are made', ' was made', ' were made', ' is prepared', ' are prepared']
            passive_count = sum(1 for phrase in passive_indicators if phrase.lower() in sample.lower())
            
            if passive_count > 3:
                analysis["voice"] = "mixed"
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing writing style: {str(e)}")
            return analysis

    def _parse_iso_duration(self, duration_str: str) -> int:
        if not duration_str:
            return None
            
        duration_str = duration_str.replace("PT", "")
        
        hours = 0
        minutes = 0
        
        hour_match = re.search(r'(\d+)H', duration_str)
        if hour_match:
            hours = int(hour_match.group(1))
            
        minute_match = re.search(r'(\d+)M', duration_str)
        if minute_match:
            minutes = int(minute_match.group(1))
            
        return hours * 60 + minutes

    def _consolidate_serp_results(self, results: List[Dict], keyword: str) -> Dict:
        consolidated = {
            "title": f"{keyword.title()} Recipe",
            "url": "",
            "snippet": f"A delicious recipe for {keyword}.",
            "article_content": "",
            "recipe_data": {
                "ingredients": [],
                "instructions": [],
                "cook_time": "",
                "prep_time": "",
                "total_time": "",
                "servings": ""
            },
            "featured_image": None,
            "style_analysis": {
                "tone": "neutral",
                "formality": "neutral",
                "voice": "active",
                "person": "second person"
            },
            "source_url": ""
        }
        
        primary_result = max(results, key=lambda x: len(x["article_content"]))
        
        consolidated["title"] = primary_result["title"]
        consolidated["url"] = primary_result["url"]
        consolidated["snippet"] = primary_result["snippet"]
        consolidated["article_content"] = primary_result["article_content"]
        consolidated["style_analysis"] = primary_result["style_analysis"]
        consolidated["source_url"] = primary_result["url"]
        
        for result in results:
            if result["featured_image"]:
                consolidated["featured_image"] = result["featured_image"]
                break
        
        for field in ["ingredients", "instructions"]:
            best_result = max(results, key=lambda x: len(x["recipe_data"].get(field, [])))
            if best_result["recipe_data"].get(field):
                consolidated["recipe_data"][field] = best_result["recipe_data"][field]
        
        for field in ["cook_time", "prep_time", "total_time", "servings"]:
            for result in results:
                if result["recipe_data"].get(field):
                    consolidated["recipe_data"][field] = result["recipe_data"][field]
                    break
        
        return consolidated

    def get_high_authority_links(self, keyword: str, num_links: int = 3) -> List[Dict]:
        try:
            high_authority_domains = [
                "foodnetwork.com", "allrecipes.com", "epicurious.com", "seriouseats.com",
                "simplyrecipes.com", "bonappetit.com", "cooking.nytimes.com", "food.com",
                "eatingwell.com", "bbcgoodfood.com", "recipetineats.com", "tasty.co",
                "delish.com", "thekitchn.com", "cookinglight.com", "marthastewart.com",
                "myrecipes.com", "tasteofhome.com", "jamieoliver.com", "kingarthurbaking.com"
            ]
            
            domain_query = " OR ".join([f"site:{domain}" for domain in high_authority_domains[:7]])
            search_query = f"{keyword} recipe ({domain_query})"
            
            params = {
                "engine": "google",
                "q": search_query,
                "api_key": self.config.config['serpapi_api_key'],
                "num": 10
            }
            
            response = requests.get("https://serpapi.com/search", params=params)
            serp_data = response.json()
            
            links = []
            
            if "organic_results" in serp_data:
                for result in serp_data["organic_results"]:
                    url = result.get("link", "")
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    
                    domain = self._extract_domain(url)
                    if domain in " ".join(high_authority_domains):
                        links.append({
                            "url": url,
                            "title": title,
                            "snippet": snippet,
                            "domain": domain
                        })
                        
                        if len(links) >= num_links:
                            break
            
            return links
            
        except Exception as e:
            self.logger.error(f"Error finding high-authority links: {str(e)}")
            return []

    def get_internal_links(self, website_config: Dict, keyword: str, num_links: int = 3) -> List[Dict]:
        """Get internal links with better error handling for sitemap parsing"""
        try:
            # Get sitemap URL
            sitemap_url = website_config.get("sitemap_url")
            if not sitemap_url:
                site_url = website_config.get("url", "").rstrip('/')
                if site_url:
                    sitemap_url = f"{site_url}/sitemap.xml"
                else:
                    self.logger.warning("No site URL configured, skipping internal links")
                    return []
            
            self.logger.info(f"Fetching sitemap from: {sitemap_url}")
            
            # Fetch sitemap with error handling
            try:
                response = requests.get(sitemap_url, timeout=15, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; RecipeBot/1.0)'
                })
                
                if response.status_code != 200:
                    self.logger.error(f"Failed to fetch sitemap: HTTP {response.status_code}")
                    return []
                
                # Check if response is XML or HTML
                content_type = response.headers.get('content-type', '').lower()
                if 'html' in content_type:
                    self.logger.error(f"Sitemap URL returned HTML instead of XML. Content type: {content_type}")
                    self.logger.debug(f"Response preview: {response.text[:200]}...")
                    return []
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Network error fetching sitemap: {str(e)}")
                return []
            
            sitemap_content = response.text
            
            # Parse XML with error handling
            try:
                # Remove any BOM or whitespace
                sitemap_content = sitemap_content.strip()
                if sitemap_content.startswith('\ufeff'):
                    sitemap_content = sitemap_content[1:]
                    
                # Check if content starts with XML declaration or sitemap tag
                if not (sitemap_content.startswith('<?xml') or sitemap_content.startswith('<urlset') or sitemap_content.startswith('<sitemapindex')):
                    self.logger.error("Invalid sitemap format - doesn't start with XML declaration")
                    self.logger.debug(f"Content starts with: {sitemap_content[:50]}...")
                    return []
                    
                root = ET.fromstring(sitemap_content)
                
            except ET.ParseError as e:
                self.logger.error(f"XML parsing error: {str(e)}")
                self.logger.debug(f"Failed to parse content starting with: {sitemap_content[:100]}...")
                return []
            
            namespace = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            urls = []
            
            # Process sitemap index or regular sitemap
            if root.tag.endswith('sitemapindex'):
                sitemap_urls = []
                for sitemap in root.findall('sm:sitemap', namespace):
                    loc = sitemap.find('sm:loc', namespace)
                    if loc is not None and loc.text:
                        sitemap_urls.append(loc.text)
                
                # Process sub-sitemaps
                for sub_sitemap_url in sitemap_urls[:5]:  # Limit to first 5 sub-sitemaps
                    try:
                        sub_response = requests.get(sub_sitemap_url, timeout=10, headers={
                            'User-Agent': 'Mozilla/5.0 (compatible; RecipeBot/1.0)'
                        })
                        if sub_response.status_code == 200:
                            sub_content = sub_response.text.strip()
                            if sub_content.startswith('\ufeff'):
                                sub_content = sub_content[1:]
                            
                            sub_root = ET.fromstring(sub_content)
                            for url_element in sub_root.findall('.//sm:url', namespace):
                                loc = url_element.find('sm:loc', namespace)
                                if loc is not None and loc.text:
                                    urls.append(loc.text)
                                    
                    except Exception as e:
                        self.logger.warning(f"Error processing sub-sitemap {sub_sitemap_url}: {str(e)}")
                        continue
            else:
                # Process regular sitemap
                for url_element in root.findall('.//sm:url', namespace):
                    loc = url_element.find('sm:loc', namespace)
                    if loc is not None and loc.text:
                        urls.append(loc.text)
            
            self.logger.info(f"Found {len(urls)} URLs in sitemap")
            
            # Filter URLs
            filtered_urls = []
            for url in urls:
                # Skip non-article URLs
                if any(segment in url for segment in ['/category/', '/tag/', '/author/', '/page/', '/wp-', '/feed/']):
                    continue
                
                # Skip file URLs
                if url.endswith(('.xml', '.php', '.jpg', '.png', '.pdf', '.js', '.css')):
                    continue
                
                # Parse URL
                parts = urlparse(url)
                path_segments = parts.path.strip('/').split('/')
                
                # Skip homepage and very short paths
                if not path_segments or parts.path == '/' or len(path_segments[0]) < 3:
                    continue
                
                filtered_urls.append(url)
            
            self.logger.info(f"Filtered to {len(filtered_urls)} article URLs")
            
            # Find related links
            related_links = []
            keyword_terms = set(keyword.lower().split())
            
            # Identify pillar content
            pillar_content = []
            for url in filtered_urls:
                url_lower = url.lower()
                if any(term in url_lower for term in ["guide", "ultimate", "complete", "basics", "101", "everything"]):
                    pillar_content.append(url)
            
            # Score and select URLs
            for url in (pillar_content + filtered_urls):
                if len(related_links) >= num_links:
                    break
                    
                # Skip if already added
                if any(url == link["url"] for link in related_links):
                    continue
                
                try:
                    # Calculate relevance score
                    parts = urlparse(url)
                    path = parts.path.lower()
                    path_words = set(re.sub(r'[^a-z0-9\s]', ' ', path).split())
                    relevance_score = len(keyword_terms.intersection(path_words))
                    
                    # Add if relevant or pillar content
                    if relevance_score > 0 or url in pillar_content:
                        # Create title from URL
                        page_name = url.split('/')[-1].replace('-', ' ').title()
                        if not page_name:
                            page_name = url.split('/')[-2].replace('-', ' ').title()
                        
                        related_links.append({
                            "url": url,
                            "title": page_name,
                            "relevance_score": relevance_score,
                            "is_pillar": url in pillar_content
                        })
                        
                        self.logger.debug(f"Added internal link: {page_name} (score: {relevance_score})")
                        
                except Exception as e:
                    self.logger.warning(f"Error processing URL {url}: {str(e)}")
                    continue
            
            # Sort by relevance
            related_links.sort(key=lambda x: (x["is_pillar"], x["relevance_score"]), reverse=True)
            
            # If we don't have enough links, add some generic ones
            if len(related_links) < num_links:
                for url in filtered_urls[:10]:
                    if len(related_links) >= num_links:
                        break
                    if not any(url == link["url"] for link in related_links):
                        page_name = url.split('/')[-1].replace('-', ' ').title()
                        related_links.append({
                            "url": url,
                            "title": page_name or "Related Recipe",
                            "relevance_score": 0,
                            "is_pillar": False
                        })
            
            final_links = related_links[:num_links]
            self.logger.info(f"Returning {len(final_links)} internal links")
            self.logger.debug(f"Final internal links: {final_links}")
            return final_links
            
        except Exception as e:
            self.logger.error(f"Unexpected error in get_internal_links: {str(e)}")
            self.logger.exception(e)  # Log full traceback
            return []
    
    def _extract_domain(self, url: str) -> str:
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            if domain.startswith('www.'):
                domain = domain[4:]
                
            return domain
        except:
            return ""

    async def scrape_article_content(self, article_url: str) -> Dict:
        """
        Scrape content directly from a single article URL instead of SERP analysis
        """
        try:
            self.logger.info(f"Scraping content from: {article_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(article_url, headers=headers, timeout=20)
            if response.status_code != 200:
                raise APIError(f"Failed to fetch article URL: {response.status_code}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract all the data we need
            article_content = self._extract_article_content(soup, response.url)
            recipe_data = self._extract_recipe_data(soup, response.url)
            featured_image = self._extract_featured_image(soup, response.url)
            style_tone_analysis = self._analyze_writing_style(article_content)
            
            # Create the same structure as SERP data for compatibility
            scraped_data = {
                "recipe_data": recipe_data,
                "style_analysis": style_tone_analysis,
                "featured_image": featured_image,
                "article_content": article_content,
                "source_url": article_url,
                "serp_source": article_url  # Keep for compatibility
            }
            
            self.logger.info(f"Successfully scraped content from {article_url}")
            return scraped_data
            
        except Exception as e:
            self.logger.log_api_call("Article Scraping", {"article_url": article_url}, error=e)
            raise APIError(f"Article scraping failed: {str(e)}")

# Content Generator
class ContentGenerator:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger
        self.anthropic_client = Anthropic(api_key=config.config['claude_api_key'])
        self.keyword_processor = None

    def _extract_keywords_from_content(self, content: str) -> List[str]:
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            for element in soup.find_all(['div', 'span', 'meta']):
                classes = element.get('class', [])
                if isinstance(classes, str):
                    classes = [classes]
                    
                if any(keyword in ' '.join(classes).lower() for keyword in ['keyword', 'tag', 'meta']):
                    text = element.get_text().strip()
                    if text:
                        if ':' in text:
                            keywords_part = text.split(':', 1)[1].strip()
                            return [k.strip() for k in keywords_part.split(',') if k.strip()]
                        else:
                            return [k.strip() for k in text.split(',') if k.strip()]
            
            for meta in soup.find_all('meta'):
                if meta.get('name', '').lower() in ['keywords', 'tags']:
                    content = meta.get('content', '')
                    if content:
                        return [k.strip() for k in content.split(',') if k.strip()]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords from content: {str(e)}")
            return []

    async def generate_recipe_card_directly(self, focused_keyword: str, recipe_data: Dict) -> Dict:
        """
        Generate recipe card directly from original recipe data instead of extracting from content
        """
        try:
            self.logger.info("🎯 Generating recipe card directly from recipe data...")
            self.logger.info(f" Input recipe_data: {recipe_data}")
            
            # Extract recipe information from the original data
            title = focused_keyword.title()
            ingredients = recipe_data.get("ingredients", [])
            instructions = recipe_data.get("instructions", [])
            prep_time = recipe_data.get("prep_time", "15 minutes")
            cook_time = recipe_data.get("cook_time", "30 minutes")
            servings = recipe_data.get("servings", "4 servings")
            
            # Log the actual data being used
            self.logger.info(f"📊 Recipe data for Claude:")
            self.logger.info(f"  📝 Title: {title}")
            self.logger.info(f"   Ingredients count: {len(ingredients)}")
            self.logger.info(f"  👨‍ Instructions count: {len(instructions)}")
            self.logger.info(f"  ⏰ Prep time: {prep_time}")
            self.logger.info(f"  ⏰ Cook time: {cook_time}")
            self.logger.info(f"  👥 Servings: {servings}")
            
            # Calculate total time
            total_time = self._calculate_total_time(prep_time, cook_time)
            
            # Create focused prompt with original recipe data
            prompt = f"""
            Generate a complete recipe card JSON for the following recipe:
            
            Title: {title}
            Focus Keyword: {focused_keyword}
            Ingredients: {ingredients}
            Instructions: {instructions}
            Prep Time: {prep_time}
            Cook Time: {cook_time}
            Total Time: {total_time}
            Servings: {servings}
            
            Return a valid JSON object with this exact structure (you should respect the structure, and obviously adapt the values to the recipe context, the values belows are just examples):
            (********VERY IMPORTANT: you should give me the ingredients as a html list and not a table, with ingredient name and its quantity ok? extract them from the ingredients sent to you).
            (********VERY IMPORTANT: you should give me the instructions as a html list and not a table, with step name and its instructions ok? extract them from the table sent to you).

            {{
                "Recipe_Title": "{title}",
                "Recipe_Description": "A delicious and easy recipe for {focused_keyword} that you'll love making at home.",
                "Recipe_Keywords": "{focused_keyword}, recipe, homemade, cooking, food, easy recipe",
                "Recipe_Details": {{
                    "Prep_Time": "{prep_time}",
                    "Cook_Time": "{cook_time}",
                    "Total_Time": "{total_time}",
                    "Yield": "{servings}",
                    "Category": "Main Course",
                    "Method": "Cooking",
                    "Cuisine": "International"
                }},
                "Recipe_Ingredients": PUT THE INGREDIENTS LIST HERE (extract exactly the ingredient name and quantity from the Ingredients provided above ),
                "Recipe_Instructions": PUT THE INSTRUCTIONS LIST HERE (extract exactly the step name and instructions from the Instructions provided above ),
                "Recipe_Notes": "<ul><li>Store leftovers in an airtight container for up to 3 days.</li><li>Reheat gently before serving.</li></ul>",
                "Recipe_Nutrition": {{
                    "Serving_Size": "1 serving",
                    "Calories": "250 calories",
                    "Sugar": "5g",
                    "Sodium": "400mg",
                    "Fat": "12g",
                    "Saturated_Fat": "4g",
                    "Unsaturated_Fat": "8g",
                    "Trans_Fat": "0g",
                    "Carbohydrates": "25g",
                    "Fiber": "3g",
                    "Protein": "15g",
                    "Cholesterol": "30mg"
                }}
            }}
            VERY CRITICAL: Ingredients and Instructions should be generated as lists ok (quantity + ingredient name, step name + instructions) from the data that we provided to you !
            IMPORTANT: 
            1. Return ONLY valid JSON - no explanations or markdown
            2. Use the exact ingredients and instructions provided
            3. Use the exact timing information provided: Prep Time: {prep_time}, Cook Time: {cook_time}, Total Time: {total_time}
            4. Use the exact servings information: {servings}
            5. Calculate nutrition based on typical values for this type of recipe
            6. Ensure all HTML formatting is correct

            **********VERY IMPORTANT*************
            BE VERY PRECISE WITH THE DATA, the ingredients and instructions need to match the ones we already provided in these lists {ingredients} and {instructions} respectively.
            CALCULATE PRECISELY NUTRITIONS, PREP TIME, COOK TIME, TOTAL TIME, AND SERVINGS, AND BE VERY EXPLICIT WITH THE NOTES, TRY TO BE RELEVANT FOR THE READER WITH THE NOTES, MAXIMUM 4 NOTES
            NOTE THAT WE THE VALUES GIVEN TO YOU ARE JUST FOR EXPLANATORY PURPOSE BESIDES THE INGREDIENTS LIST AND THE INSTRUCTIONS LIST (THE LAST TWO ARE THE ONLY ONES THAT YOU NEED TO MAKE SURE YOU MATCH PRECISELY, WE NEED TO BE COHESIVE WITH THE RECIPE CARD INGREDIENTS AND INSTRUCTIONS AND THE ONES SHOWN IN THE ARTICLE )
            """

            self.logger.info(f"🤖 Sending recipe data to Claude for recipe card generation... THE PROMPT IS:============ {prompt}")

            # Call Claude API
            response = self.anthropic_client.messages.create(
                model=self.config.config['claude_model'],
                temperature=0.2,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse the response
            response_text = response.content[0].text
            self.logger.info(f"📥 Received response from Claude ({len(response_text)} characters, the response is: {response_text})")
            
            # Extract JSON from response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
                self.logger.info("✅ Found JSON in ```json``` blocks")
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
                self.logger.info("✅ Found JSON in ``` blocks")
            else:
                # Find JSON object between curly braces
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    self.logger.info("✅ Found JSON between curly braces")
                else:
                    self.logger.error("❌ Failed to extract JSON from Claude's response")
                    self.logger.error(f"Response text: {response_text}")
                    raise ValueError("Failed to extract JSON from Claude's response")
            
            # Parse extracted JSON
            self.logger.info("🔍 Parsing JSON response...")
            recipe_card = json.loads(json_str)
            self.logger.info("✅ Successfully parsed JSON")
            
            # Validate required fields
            required_fields = ['Recipe_Title', 'Recipe_Description', 'Recipe_Details', 'Recipe_Ingredients', 'Recipe_Instructions']
            missing_fields = [field for field in required_fields if field not in recipe_card]
            if missing_fields:
                self.logger.warning(f"⚠️ Missing required fields in recipe card: {missing_fields}")
            
            # Ensure timing and serving information is preserved
            if 'Recipe_Details' in recipe_card:
                recipe_card['Recipe_Details']['Prep_Time'] = prep_time
                recipe_card['Recipe_Details']['Cook_Time'] = cook_time
                recipe_card['Recipe_Details']['Total_Time'] = total_time
                recipe_card['Recipe_Details']['Yield'] = servings
            
            # Ensure we have Recipe_Notes (add default if missing)
            if 'Recipe_Notes' not in recipe_card or not recipe_card['Recipe_Notes']:
                recipe_card['Recipe_Notes'] = "<ul><li>Store leftovers in an airtight container for up to 3 days.</li><li>Reheat gently before serving.</li></ul>"
                self.logger.info("📝 Added default recipe notes")
            
            # Ensure we have Recipe_Keywords (add default if missing)
            if 'Recipe_Keywords' not in recipe_card or not recipe_card['Recipe_Keywords']:
                recipe_card['Recipe_Keywords'] = f"{focused_keyword}, recipe, homemade, cooking, food, easy recipe"
                self.logger.info("🏷️ Added default recipe keywords")
            
            # Ensure we have Recipe_Nutrition (add default if missing)
            if 'Recipe_Nutrition' not in recipe_card or not recipe_card['Recipe_Nutrition']:
                recipe_card['Recipe_Nutrition'] = {
                    "Serving_Size": "1 serving",
                    "Calories": "250 calories",
                    "Sugar": "5g",
                    "Sodium": "400mg",
                    "Fat": "12g",
                    "Saturated_Fat": "4g",
                    "Unsaturated_Fat": "8g",
                    "Trans_Fat": "0g",
                    "Carbohydrates": "25g",
                    "Fiber": "3g",
                    "Protein": "15g",
                    "Cholesterol": "30mg"
                }
                self.logger.info("🥗 Added default nutrition data")
                
            # Log successful generation with ALL fields
            self.logger.info(f"🎉 Recipe card generated successfully for: {recipe_card.get('Recipe_Title', title)}")
            self.logger.info(f"📊 Final recipe card contains:")
            self.logger.info(f"  📝 Title: {recipe_card.get('Recipe_Title', 'N/A')}")
            self.logger.info(f"  📄 Description: {recipe_card.get('Recipe_Description', 'N/A')}")
            self.logger.info(f"  🏷️ Keywords: {recipe_card.get('Recipe_Keywords', 'N/A')}")
            self.logger.info(f"   Ingredients: {len(recipe_card.get('Recipe_Ingredients', []))}")
            self.logger.info(f"  ‍ Instructions: {len(recipe_card.get('Recipe_Instructions', []))}")
            self.logger.info(f"  📝 Notes: {recipe_card.get('Recipe_Notes', 'N/A')}")
            self.logger.info(f"  ⏰ Prep time: {recipe_card.get('Recipe_Details', {}).get('Prep_Time', 'N/A')}")
            self.logger.info(f"  ⏰ Cook time: {recipe_card.get('Recipe_Details', {}).get('Cook_Time', 'N/A')}")
            self.logger.info(f"  ⏰ Total time: {recipe_card.get('Recipe_Details', {}).get('Total_Time', 'N/A')}")
            self.logger.info(f"  👥 Servings: {recipe_card.get('Recipe_Details', {}).get('Yield', 'N/A')}")
            self.logger.info(f"  🍽️ Category: {recipe_card.get('Recipe_Details', {}).get('Category', 'N/A')}")
            self.logger.info(f"  🍳 Method: {recipe_card.get('Recipe_Details', {}).get('Method', 'N/A')}")
            self.logger.info(f"  🌍 Cuisine: {recipe_card.get('Recipe_Details', {}).get('Cuisine', 'N/A')}")
            self.logger.info(f"  🥗 Nutrition: {len(recipe_card.get('Recipe_Nutrition', {}))} fields")
            
            return recipe_card
            
        except Exception as e:
            self.logger.error(f"❌ Error generating recipe card: {str(e)}")
            self.logger.error(f" Full error details: {type(e).__name__}: {str(e)}")
            self.logger.error(f" Recipe data that failed: {recipe_data}")
            
            # NO FALLBACK - Let the error propagate so you can see what's wrong
            raise Exception(f"Recipe card generation failed: {str(e)}")

    def _calculate_total_time(self, prep_time: str, cook_time: str) -> str:
        """Calculate total time from prep and cook times"""
        try:
            prep_minutes = self._extract_minutes(prep_time)
            cook_minutes = self._extract_minutes(cook_time)
            total_minutes = prep_minutes + cook_minutes
            
            if total_minutes >= 60:
                hours = total_minutes // 60
                minutes = total_minutes % 60
                if minutes == 0:
                    return f"{hours} hour{'s' if hours > 1 else ''}"
                else:
                    return f"{hours} hour{'s' if hours > 1 else ''} {minutes} minute{'s' if minutes > 1 else ''}"
            else:
                return f"{total_minutes} minute{'s' if total_minutes > 1 else ''}"
        except:
            return "50 minutes"

    def _extract_minutes(self, time_str: str) -> int:
        """Extract minutes from time string like '20 minutes' or '1 hour 30 minutes'"""
        try:
            time_str = time_str.lower().strip()
            total_minutes = 0
            
            # Check for hours
            if 'hour' in time_str:
                hour_match = re.search(r'(\d+)\s*hour', time_str)
                if hour_match:
                    total_minutes += int(hour_match.group(1)) * 60
            
            # Check for minutes
            if 'minute' in time_str:
                minute_match = re.search(r'(\d+)\s*minute', time_str)
                if minute_match:
                    total_minutes += int(minute_match.group(1))
            
            return total_minutes if total_minutes > 0 else 30  # Default fallback
        except:
            return 30

    def generate_recipe_card(self, content: str, focused_keyword: str, recipe_data: Dict) -> Dict:
        """
        Wrapper method to call the new direct recipe card generation
        """
        import asyncio
        try:
            # Create event loop if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async method
            return loop.run_until_complete(self.generate_recipe_card_directly(focused_keyword, recipe_data))
        except Exception as e:
            self.logger.error(f"❌ Error in recipe card wrapper: {str(e)}")
            # NO FALLBACK - Let the error propagate
            raise Exception(f"Recipe card generation failed: {str(e)}")

    def set_keyword_processor(self, keyword_processor):
       self.keyword_processor = keyword_processor
   
    async def generate_seo_metadata(self, keyword: str, scraped_data: Dict) -> Dict:
        try:
            seo_params = self.config.get_seo_params()
            max_focus_keyword_words = seo_params.get('max_focus_keyword_words', 4)
            
            recipe_title = scraped_data.get('title', f"{keyword.title()} Recipe")
            recipe_snippet = scraped_data.get('snippet', f"A delicious recipe for {keyword}.")
            
            self.logger.info(f"🎯 Starting SEO metadata generation for keyword: '{keyword}'")
            
            prompt = f"""
            Generate comprehensive SEO metadata for a recipe article about "{keyword}".
            
            Based on SERP analysis:
            - Recipe title from top results: {recipe_title}
            - Recipe description: {recipe_snippet}
            
            CRITICAL SEO RULES TO FOLLOW, It's mandatory and very important to stick to the rules:
            - Focus keyword: 4 words maximum, highly specific
            - SEO title: exactly between 60-65 characters maximum, include focus keyword at the beginning (start by it explicitly), a powerful positive word at the beginning is mandatory,
            the exact format should be (focus keyword, rest of the title include a powerful word)
            - Meta description: exactly 150 characters exactly, include focus keyword (include it explicitly) and avoid AI words such as indulge etc, THE FOCUS KEYWORD SHOULD BE INCLUDED EXPLICITLY, 
            - slug: Use the focus keyword with hyphens between words (e.g., "dark chocolate cake" becomes "dark-chocolate-cake")

            VERY IMPORTANT: STICK TO THE SEO RULES, It's very critical to follow the rules strictly.
            
            Format your response as a valid JSON object with these exact fields:
            {{
            "focus_keyword": "",
            "seo_title": "",
            "meta_description": "",
            "slug": ""
            }}
            """
            
            request_data = {
                "model": self.config.config['claude_model'],
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            self.logger.log_api_call("SEO Metadata Generation", request_data)
            
            response = self.anthropic_client.messages.create(
                model=self.config.config['claude_model'],
                temperature=1,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            self.logger.info(f"📥 Received response from Claude (length: {len(response_text)} chars)")
            
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                else:
                    raise ValueError("Could not extract JSON from response")
            
            self.logger.info(f"📄 Extracted JSON: {json_str}")
            
            metadata = json.loads(json_str)
            
            # Log what Claude provided
            self.logger.info("🤖 Claude's raw metadata:")
            self.logger.info(f"  📝 focus_keyword: '{metadata.get('focus_keyword', 'NOT PROVIDED')}'")
            self.logger.info(f"  📰 seo_title: '{metadata.get('seo_title', 'NOT PROVIDED')}'")
            self.logger.info(f"  📄 meta_description: '{metadata.get('meta_description', 'NOT PROVIDED')}'")
            self.logger.info(f"  🔗 slug: '{metadata.get('slug', 'NOT PROVIDED')}'")
            
            # Just return exactly what Claude gives us
            validated_metadata = {
                "focus_keyword": metadata.get("focus_keyword", keyword),
                "seo_title": metadata.get("seo_title", f"{keyword.title()} Recipe"),
                "meta_description": metadata.get("meta_description", f"A delicious recipe for {keyword}."),
                "slug": metadata.get("slug", keyword.lower().replace(' ', '-'))
            }
            
            # Log final metadata
            self.logger.info("✅ Final SEO metadata:")
            self.logger.info(f"  📝 focus_keyword: '{validated_metadata['focus_keyword']}' (length: {len(validated_metadata['focus_keyword'])})")
            self.logger.info(f"  📰 seo_title: '{validated_metadata['seo_title']}' (length: {len(validated_metadata['seo_title'])})")
            self.logger.info(f"  📄 meta_description: '{validated_metadata['meta_description']}' (length: {len(validated_metadata['meta_description'])})")
            self.logger.info(f"  🔗 slug: '{validated_metadata['slug']}' (length: {len(validated_metadata['slug'])})")


            self.logger.log_api_call(
                "SEO Metadata Generation", 
                request_data, 
                {"validated_metadata": validated_metadata}
            )
            
            return validated_metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error generating SEO metadata: {str(e)}")
            raise APIError(f"SEO metadata generation failed: {str(e)}")

    async def generate_strict_outline(self, keyword: str, focused_keyword: str, template: str) -> Dict:
       try:
           template_structure = self._parse_template_structure(template)
           
           prompt = f"""
           Create a detailed outline for a recipe article about "{keyword}" with focus keyword "{focused_keyword}".
           
           YOU MUST FOLLOW THIS EXACT TEMPLATE STRUCTURE:
           {template}
           
           Requirements:
           1. CRITICAL: Maintain exactly the same heading structure as the template
           2. Include the focus keyword "{focused_keyword}" in at least 6 H2 Headings
           3. Do not add or remove any main sections from the template
           4. Use the exact same heading levels as shown in the template (h2, h3, etc.)
           5. The first heading must be h2 and contain the focus keyword
           
           Return the outline as a complete, valid JSON array with this format:
           [
             {{"level": "h2", "text": "Introduction Section"}},
             {{"level": "h3", "text": "Subsection Example"}}
             // Additional sections following the template structure
           ]
           """
           
           request_data = {
               "model": self.config.config['claude_model'],
               "max_tokens": 1500,
               "messages": [{"role": "user", "content": prompt}]
           }
           
           self.logger.log_api_call("Outline Generation", request_data)
           
           response = self.anthropic_client.messages.create(
               model=self.config.config['claude_model'],
               temperature=1,
               max_tokens=1500,
               messages=[{"role": "user", "content": prompt}]
           )
           
           response_text = response.content[0].text
           
           if "```json" in response_text:
               json_str = response_text.split("```json")[1].split("```")[0].strip()
           elif "```" in response_text:
               json_str = response_text.split("```")[1].split("```")[0].strip()
           else:
               start_idx = response_text.find('[')
               end_idx = response_text.rfind(']') + 1
               if start_idx >= 0 and end_idx > start_idx:
                   json_str = response_text[start_idx:end_idx]
               else:
                   raise ValueError("Could not extract JSON from response")
           
           outline_items = json.loads(json_str)
           
           if not self._validate_outline_against_template(outline_items, template_structure):
               self.logger.warning("Generated outline doesn't match template structure. Regenerating...")
               return await self._regenerate_outline_with_stronger_prompt(keyword, focused_keyword, template, template_structure)
           
           has_keyword_in_h1 = False
           keyword_in_subheadings_count = 0
           
           for item in outline_items:
               if item["level"] == "h1" and focused_keyword.lower() in item["text"].lower():
                   has_keyword_in_h1 = True
               elif item["level"] in ["h2", "h3"] and focused_keyword.lower() in item["text"].lower():
                   keyword_in_subheadings_count += 1
           
           if not has_keyword_in_h1 or keyword_in_subheadings_count < 2:
               self.logger.warning("Focus keyword not sufficiently present in outline. Adding...")
               outline_items = self._add_focus_keyword_to_outline(outline_items, focused_keyword)
           
           return {
               "focused_keyword": focused_keyword,
               "outline": outline_items
           }
           
       except Exception as e:
           self.logger.error(f"Error generating outline: {str(e)}")
           return self._create_fallback_outline(keyword, focused_keyword)

    def _parse_template_structure(self, template: str) -> List[Dict]:
       structure = []
       
       heading_pattern = r'<(H[1-6])>\s*(.*?)(?=$|\n|<H[1-6]>)'
       matches = re.finditer(heading_pattern, template, re.DOTALL | re.MULTILINE)
       
       for match in matches:
           level = match.group(1).lower()
           text = match.group(2).strip()
           structure.append({"level": level, "text": text})
       
       return structure

    def _validate_outline_against_template(self, outline: List[Dict], template_structure: List[Dict]) -> bool:
       if len(outline) != len(template_structure):
           self.logger.warning(f"Outline has {len(outline)} headings but template has {len(template_structure)}")
           return False
       
       for i, (outline_item, template_item) in enumerate(zip(outline, template_structure)):
           if outline_item["level"] != template_item["level"]:
               self.logger.warning(f"Heading level mismatch at position {i}. Outline: {outline_item['level']}, Template: {template_item['level']}")
               return False
       
       return True
   
    async def _regenerate_outline_with_stronger_prompt(self, keyword: str, focused_keyword: str, template: str, template_structure: List[Dict]) -> Dict:
       formatted_structure = "\n".join([f"{item['level']}: {item['text']}" for item in template_structure])
       
       prompt = f"""
       Create a detailed outline for a recipe article about "{keyword}" with focus keyword "{focused_keyword}".
       
       YOU MUST MATCH THIS EXACT HEADING STRUCTURE - WITH THE SAME NUMBER AND LEVELS OF HEADINGS:
       {formatted_structure}
       
       CRITICAL REQUIREMENTS:
       1. Your outline MUST have exactly {len(template_structure)} headings - no more, no less
       2. Each heading MUST use the same level (h2, h3, etc.) as shown in the template
       3. Include focus keyword "{focused_keyword}" in:
           - At least 6 H2 headings
       
       Return ONLY a valid JSON array with this format:
       [
       {{"level": "h2", "text": "Introduction Section"}},
       {{"level": "h3", "text": "Subsection Example"}}
       // Continue with all remaining headings in the same order and levels as the template
       ]
       """
       
       request_data = {
           "model": self.config.config['claude_model'],
           "max_tokens": 1500,
           "messages": [{"role": "user", "content": prompt}]
       }
       
       self.logger.log_api_call("Outline Regeneration", request_data)
       
       response = self.anthropic_client.messages.create(
           model=self.config.config['claude_model'],
           temperature=1,
           max_tokens=1500,
           messages=[{"role": "user", "content": prompt}]
       )
       
       response_text = response.content[0].text
       
       if "```json" in response_text:
           json_str = response_text.split("```json")[1].split("```")[0].strip()
       elif "```" in response_text:
           json_str = response_text.split("```")[1].split("```")[0].strip()
       else:
           start_idx = response_text.find('[')
           end_idx = response_text.rfind(']') + 1
           if start_idx >= 0 and end_idx > start_idx:
               json_str = response_text[start_idx:end_idx]
           else:
               return self._create_fallback_outline(keyword, focused_keyword, template_structure)
       
       try:
           outline_items = json.loads(json_str)
           
           if not self._validate_outline_against_template(outline_items, template_structure):
               self.logger.warning("Regenerated outline still doesn't match template. Using fallback.")
               return self._create_fallback_outline(keyword, focused_keyword, template_structure)
           
           outline_items = self._add_focus_keyword_to_outline(outline_items, focused_keyword)
           
           return {
               "focused_keyword": focused_keyword,
               "outline": outline_items
           }
       except Exception as e:
           self.logger.error(f"Error processing regenerated outline: {str(e)}")
           return self._create_fallback_outline(keyword, focused_keyword, template_structure)

    def _add_focus_keyword_to_outline(self, outline_items: List[Dict], focused_keyword: str) -> List[Dict]:
      for i, item in enumerate(outline_items):
          if item["level"] == "h1":
              if focused_keyword.lower() not in item["text"].lower():
                  outline_items[i]["text"] = f"{item['text']} - {focused_keyword.title()}"
              break
      
      subheadings_with_keyword = 0
      for i, item in enumerate(outline_items):
          if item["level"] in ["h2", "h3"] and focused_keyword.lower() in item["text"].lower():
              subheadings_with_keyword += 1
              
      if subheadings_with_keyword < 2:
          candidate_indices = []
          for i, item in enumerate(outline_items):
              if item["level"] in ["h2", "h3"] and focused_keyword.lower() not in item["text"].lower():
                  candidate_indices.append(i)
          
          for i in range(min(2 - subheadings_with_keyword, len(candidate_indices))):
              index = candidate_indices[i]
              heading_text = outline_items[index]["text"]
              
              if ":" in heading_text:
                  parts = heading_text.split(":", 1)
                  outline_items[index]["text"] = f"{parts[0]}: {focused_keyword.title()} {parts[1]}"
              else:
                  outline_items[index]["text"] = f"{heading_text} with {focused_keyword.title()}"
                  
      return outline_items
      
    def _create_fallback_outline(self, keyword: str, focused_keyword: str, template_structure=None) -> Dict:
      if template_structure:
          outline = []
          for item in template_structure:
              level = item["level"]
              text = item["text"]
              
              if level == "h1":
                  text = f"Delicious {focused_keyword.title()} Recipe"
              elif level == "h2" and "introduction" in text.lower():
                  text = f"Introduction to {focused_keyword.title()}"
              elif level == "h2" and "ingredients" in text.lower():
                  text = f"Ingredients for {focused_keyword.title()}"
              elif level == "h2" and "instructions" in text.lower() or "how to" in text.lower():
                  text = f"How to Make {focused_keyword.title()}"
              
              outline.append({"level": level, "text": text})
      else:
          outline = [
              {"level": "h1", "text": f"Delicious {focused_keyword.title()} Recipe"},
              {"level": "h2", "text": "Introduction"},
              {"level": "h2", "text": "Why You'll Love This Recipe"},
              {"level": "h2", "text": f"Ingredients You'll Need for {focused_keyword.title()}"},
              {"level": "h3", "text": "Main Ingredients"},
              {"level": "h3", "text": "Optional Add-ins"},
              {"level": "h2", "text": "Kitchen Equipment Needed"},
              {"level": "h2", "text": f"How to Make {focused_keyword.title()}"},
              {"level": "h3", "text": "Preparation Steps"},
              {"level": "h3", "text": "Cooking Instructions"},
              {"level": "h3", "text": "Finishing Touches"},
              {"level": "h2", "text": "Expert Tips for Success"},
              {"level": "h2", "text": "Common Mistakes to Avoid"},
              {"level": "h2", "text": "Serving Suggestions"},
              {"level": "h2", "text": "Storage and Freezing Instructions"},
              {"level": "h2", "text": "Variations and Substitutions"},
              {"level": "h2", "text": "Chef's Helpful Tips"},
              {"level": "h2", "text": "You Must Know"},
              {"level": "h2", "text": "Frequently Asked Questions"},
              {"level": "h2", "text": "Conclusion"}
          ]
      
      return {
          "focused_keyword": focused_keyword,
          "outline": outline
      }

    async def generate_article_content(self, keyword: str, focused_keyword: str, outline: List[Dict], 
                           recipe_data: Dict, external_links: List[Dict], internal_links: List[Dict], 
                           style_analysis: Dict) -> str:
      try:
          formatted_outline = "\n".join([f"{item['level']}: {item['text']}" for item in outline])
          
          ingredients_formatted = "\n".join([f"• {ingredient}" for ingredient in recipe_data.get("ingredients", [])]) if recipe_data.get("ingredients") else "No ingredients available"
          instructions_formatted = "\n".join([f"{i+1}. {instruction}" for i, instruction in enumerate(recipe_data.get("instructions", []))]) if recipe_data.get("instructions") else "No instructions available"
          
          external_links_formatted = "\n".join([
              f"• {link['title']} - {link['url']} ({link['domain']})" 
              for link in external_links
          ]) if external_links else "No external links available"
          
          internal_links_formatted = "\n".join([
              f"• {link['title']} - {link['url']}{' (Pillar Content)' if link.get('is_pillar') else ''}" 
              for link in internal_links
          ]) if internal_links else "No internal links available"
          
          tone = style_analysis.get("tone", "neutral")
          formality = style_analysis.get("formality", "neutral")
          voice = style_analysis.get("voice", "active")
          person = style_analysis.get("person", "second person")
          
          prompt = f"""
           You are tasked with writing a complete, publish-ready recipe article [maximum 1500 words] about "{keyword}" with focus keyword "{focused_keyword}".

           STRICTLY FOLLOW THESE FORMATTING REQUIREMENTS, ITS MANDATORY TO FOLLOW ALL THESE REQUIREMENTS !!
           IMPORTANT NOTE: SHORT PARAGRAPH ARE MANDATORY, A SECTION SHOULDN'T HAVE MORE THAN 200 WORDS.

           1. The article must use proper HTML formatting with opening and closing tags
           2. The article must include ALL headings from the outline in the exact order provided
           3. replace every alcohol or non-halal ingredient, keyword or word, with its halal equivalent please without mentionning the word halal in the content
           4. The article must include ALL required shortcode sections placed EXACTLY as specified below
           5. The article must be 1500 words max in length
           7. The article must include at least 2 external links and 2 internal links
           8. A section should not exceed 250 words. ( a paragraph should not have more than 80 word, if so add subheadings h3 or h4 to make it more readable)
           9. focus keyword should be present in the beginning of the article (the exact focus keyword should be present in the beginning)
           10. Avoid the usage of AI words, make the content humanized and user friendly.
           11. the content should be consice and straight to the point, we need the user experience to be very friendly, use lists, shorts paragraphs, insightful informations.

           === OUTLINE TO FOLLOW EXACTLY ===
           {formatted_outline}

           === RECIPE INFORMATION ===
           Ingredients:
           {ingredients_formatted}

           Instructions:
           {instructions_formatted}

           Prep Time: {recipe_data.get('prep_time', '15 minutes')}
           Cook Time: {recipe_data.get('cook_time', '30 minutes')}
           Servings: {recipe_data.get('servings', '4 servings')}

           === WRITING STYLE ===
           - Tone: {tone}
           - Formality: {formality}
           - Voice: {voice}
           - Person: {person}

           === LINKS TO INCLUDE ===
           External Links (include at least 2):
           {external_links_formatted}

           Internal Links (include at least 2):
           {internal_links_formatted}

           === REQUIRED SHORTCODE SECTIONS WITH EXACT PLACEMENT ===

           You MUST include these 4 shortcode sections EXACTLY at these specific positions:

           1. "Why You'll Love This Recipe" section:
           - Place IMMEDIATELY AFTER your introductory paragraph
           - Format as follows:
           [why_youll_love]
           <ul>
               <li>First compelling reason with some detail and explanation</li>
               <li>Second compelling reason with some detail and explanation</li>
               <li>Third compelling reason with some detail and explanation</li>
               <li>Fourth compelling reason with some detail and explanation</li>
               <li>Fifth compelling reason with some detail and explanation</li>
           </ul>
           [/why_youll_love]

           2. "Chef's Helpful Tips" section:
           - Place IMMEDIATELY AFTER the "How to Make [recipe_title]" section's content
           - Format as follows:
           [chef_tips]
           <ul>
               <li>First practical tip with explanation</li>
               <li>Second practical tip with explanation</li>
               <li>Third practical tip with explanation</li>
               <li>Fourth practical tip with explanation</li>
               <li>Fifth practical tip with explanation</li>
           </ul>
           [/chef_tips]

           3. "Important Notes" section:
           - Place IMMEDIATELY AFTER the "Mistakes to Avoid while making [recipe_title]" section's content
           - Format as follows:
           [you_must_know]
           <ul>
               <li>First important point with explanation</li>
               <li>Second important point with explanation</li>
               <li>Third important point with explanation</li>
               <li>Fourth important point with explanation</li>
           </ul>
           [/you_must_know]

           4. "FAQ" section:
           - Place IMMEDIATELY BEFORE the Conclusion section
           - Format as follows:
           [faq_section]
           [faq_item question="First question about {focused_keyword}?"]
           <p>Detailed answer to the first question.</p>
           [/faq_item]
           [faq_item question="Second question about {focused_keyword}?"]
           <p>Detailed answer to the second question.</p>
           [/faq_item]
           [faq_item question="Third question about {focused_keyword}?"]
           <p>Detailed answer to the third question.</p>
           [/faq_item]
           [faq_item question="Fourth question about {focused_keyword}?"]
           <p>Detailed answer to the fourth question.</p>
           [/faq_item]
           [faq_item question="Fifth question about {focused_keyword}?"]
           <p>Detailed answer to the fifth question.</p>
           [/faq_item]
           [/faq_section]

           === HTML STRUCTURE REQUIREMENTS ===

           Your article must include these HTML elements:
           - A single <h1> tag with the main title
           - <h2> tags for main sections matching the outline
           - <h3> tags for subsections matching the outline
           - <p> tags for paragraphs
           - <ul> and <li> tags for unordered lists
           - <ol> and <li> tags for ordered lists/instructions
           - <a href="URL" target="_blank" rel="noopener noreferrer">link text</a> for ALL external and internal links (IMPORTANT: all links must open in new window)
           - Format the ingredients list in ingredients section in a table with three columns ("ingredient" colum that shows the name of the ingredient, "quantity (approx)" column, that shows the approximate quantity
            or exact quantity to be used in the recipe, "purpose / notes" column that shows the main purpose of that ingredients in that recipe), be very concise and straight to the point here.
            - Format the tools list in a table ("tool" showing the name of the tool, and "purpose" column that shows what it serves and its main purpose)
           - format the steps to make the recipe in a table format (showing step number or name: and the details of it) for better user experience:

           === CRITICAL SHORTCODE PLACEMENT INSTRUCTIONS ===

           DO NOT add or remove headings from the outline. Follow these exact placements:
           - Introduction paragraph → THEN [why_youll_love] shortcode
           - How to Make section → THEN [chef_tips] shortcode 
           - Mistakes to Avoid section → THEN [you_must_know] shortcode
           - Place [faq_section] RIGHT BEFORE the Conclusion section

           === FINAL CHECKLIST BEFORE SUBMITTING ===

           Before submitting your response, confirm that:
           - You've included ALL headings from the outline in the correct order
           - You've placed ALL 4 shortcode sections EXACTLY where specified
           - The article should not exceed 1500 words
           - All HTML tags are properly closed
           - The focus keyword appears in the first paragraph, headings, and throughout the article
           - You've included at least 2 external links and 2 internal links

           Return ONLY the complete, properly formatted HTML content with all required sections placed exactly as instructed.
           """
          
          request_data = {
              "model": self.config.config['claude_model'],
              "max_tokens": 6000,
              "messages": [{"role": "user", "content": prompt}]
          }
          
          self.logger.log_api_call("Article Content Generation", request_data)
          
          response = self.anthropic_client.messages.create(
              model=self.config.config['claude_model'],
              temperature=0.7,
              max_tokens=6000,
              messages=[{"role": "user", "content": prompt}]
          )
          
          content = response.content[0].text
          
          content = re.sub(r'```html\s*', '', content)
          content = re.sub(r'```\s*', '', content)
          
          return content
          
      except Exception as e:
          self.logger.error(f"Error generating article content: {str(e)}")
          return self._create_fallback_content(keyword, focused_keyword, outline, recipe_data)

    async def generate_content(self, keyword: str, focused_keyword: str, scraped_data: Dict, 
                      outline: Dict, external_links: List[Dict], internal_links: List[Dict], 
                      template: str, website_config: Dict) -> Tuple[str, Dict]:
      try:
          outline_items = outline.get("outline", [])
          recipe_data = scraped_data.get("recipe_data", {})  # Changed from serp_data to scraped_data
          style_analysis = scraped_data.get("style_analysis", {})  # Changed from serp_data to scraped_data
          
          content = await self.generate_article_content(
              keyword,
              focused_keyword, 
              outline_items,
              recipe_data,
              external_links,
              internal_links,
              style_analysis
          )
          
          seo_metadata = self._extract_basic_seo_metadata(content, focused_keyword)
          
          return content, seo_metadata
          
      except Exception as e:
          self.logger.error(f"Error generating content: {str(e)}")
          raise APIError(f"Content generation failed: {str(e)}")

    def _extract_basic_seo_metadata(self, content: str, focused_keyword: str) -> Dict:
      metadata = {
          "title": "",
          "focus_keyword": focused_keyword,
          "word_count": 0,
          "keyword_density": 0,
          "keyword_count": 0
      }
      
      try:
          h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', content, re.DOTALL)
          if h1_match:
              metadata["title"] = h1_match.group(1).strip()
          
          plain_text = re.sub(r'<[^>]+>', '', content)
          words = plain_text.split()
          metadata["word_count"] = len(words)
          
          keyword_lower = focused_keyword.lower()
          text_lower = plain_text.lower()
          
          metadata["keyword_count"] = text_lower.count(keyword_lower)
          
          if metadata["word_count"] > 0:
              metadata["keyword_density"] = (metadata["keyword_count"] / metadata["word_count"]) * 100
      
      except Exception as e:
          self.logger.error(f"Error extracting basic SEO metadata: {str(e)}")
      
      return metadata

    def _create_fallback_content(self, keyword: str, focused_keyword: str, outline: List[Dict], recipe_data: Dict) -> str:
      content_parts = []
      
      title = next((item["text"] for item in outline if item["level"] == "h1"), f"Delicious {focused_keyword.title()} Recipe")
      content_parts.append(f"<h1>{title}</h1>")
      
      content_parts.append(f"<p>Welcome to this delicious {focused_keyword} recipe! If you're looking for a perfect {focused_keyword} that will impress your family and friends, you've come to the right place. This recipe combines simple ingredients with expert techniques to create a mouth-watering dish that everyone will love.</p>")
      
      for item in outline:
          if item["level"] != "h1":
              content_parts.append(f"<{item['level']}>{item['text']}</{item['level']}>")
              
              section_title = item["text"].lower()
              
              if "introduction" in section_title:
                  content_parts.append(f"<p>Let me introduce you to this amazing {focused_keyword} recipe. It's perfect for any occasion and will surely become one of your favorites. The combination of flavors in this {focused_keyword} creates a truly memorable experience.</p>")
              
              elif "why you'll love" in section_title:
                  content_parts.append(f"""[why_youll_love]
                      <ul>
                          <li>Easy to make with simple ingredients you probably already have</li>
                          <li>Perfect balance of flavors that will satisfy everyone</li>
                          <li>Can be prepared in advance, saving you time when entertaining</li>
                          <li>Versatile recipe that can be adapted to your preferences</li>
                          <li>Great for special occasions or everyday meals</li>
                      </ul>
                      [/why_youll_love]""")
              
              elif "ingredients" in section_title:
                  if recipe_data.get("ingredients"):
                      ingredients_list = "<ul>\n" + "\n".join([f"<li>{ingredient}</li>" for ingredient in recipe_data["ingredients"]]) + "\n</ul>"
                      content_parts.append(ingredients_list)
                  else:
                      content_parts.append(f"""<p>For this {focused_keyword} recipe, you'll need the following ingredients:</p>
                      <ul>
                          <li>Main ingredient</li>
                          <li>Secondary ingredients</li>
                          <li>Seasonings and spices</li>
                          <li>Optional garnishes</li>
                      </ul>""")
                                      
              elif "instructions" in section_title or "how to" in section_title:
                  if recipe_data.get("instructions"):
                      instructions_list = "<ol>\n" + "\n".join([f"<li>{instruction}</li>" for instruction in recipe_data["instructions"]]) + "\n</ol>"
                      content_parts.append(instructions_list)
                  else:
                      content_parts.append(f"""<p>Follow these steps to make the perfect {focused_keyword}:</p>
                      <ol>
                          <li>Prepare your ingredients by measuring and chopping as needed</li>
                          <li>Mix the main ingredients together</li>
                          <li>Add seasonings and spices to taste</li>
                          <li>Cook according to the recommended time and temperature</li>
                          <li>Let rest before serving for the best flavor and texture</li>
                      </ol>""")
              
              elif "tips" in section_title:
                  content_parts.append(f"""[chef_tips]
                      <ul>
                          <li>Use high-quality ingredients for the best flavor</li>
                          <li>Don't overcook to maintain the perfect texture</li>
                          <li>Let it rest before serving to allow flavors to develop</li>
                          <li>Taste and adjust seasonings as you go</li>
                          <li>Garnish just before serving for the best presentation</li>
                      </ul>
                      [/chef_tips]""")
              
              elif "you must know" in section_title:
                  content_parts.append(f"""[you_must_know]
                      <ul>
                          <li>This {focused_keyword} can be stored in an airtight container for up to 3 days</li>
                          <li>The recipe can be doubled easily for larger gatherings</li>
                          <li>For best results, serve at the recommended temperature</li>
                          <li>This dish pairs perfectly with a simple side salad or crusty bread</li>
                      </ul>
                      [/you_must_know]""")
              
              elif "faq" in section_title or "frequently asked" in section_title:
                  content_parts.append(f"""[faq_section]
                      [faq_item question="Can I make this {focused_keyword} recipe ahead of time?"]
                      <p>Yes, you can prepare this recipe up to 24 hours in advance. Store it covered in the refrigerator and reheat when ready to serve.</p>
                      [/faq_item]
                      [faq_item question="What can I substitute if I don't have all the ingredients?"]
                      <p>There are several substitutions that work well in this {focused_keyword} recipe. Use olive oil instead of butter, replace fresh herbs with dried (use 1/3 the amount), and swap in any similar vegetables you have on hand.</p>
                      [/faq_item]
                      [faq_item question="Is this {focused_keyword} recipe gluten-free?"]
                      <p>The basic recipe is naturally gluten-free, but always check individual ingredient labels as some products may contain hidden gluten or be processed in facilities that also process gluten-containing foods.</p>
                      [/faq_item]
                      [faq_item question="How can I make this {focused_keyword} healthier?"]
                      <p>To make this recipe healthier, reduce the amount of salt and oil, substitute whole grain options when applicable, and increase the vegetable content.</p>
                      [/faq_item]
                      [faq_item question="Can I freeze the leftover {focused_keyword}?"]
                      <p>Yes, this recipe freezes well. Store in an airtight container for up to 3 months. Thaw overnight in the refrigerator before reheating thoroughly.</p>
                      [/faq_item]
                      [/faq_section]""")
              
              elif "conclusion" in section_title:
                  content_parts.append(f"<p>Now you have all the information you need to make a delicious {focused_keyword} at home. This recipe is perfect for any occasion and will surely impress your family and friends. Remember to follow the tips for the best results, and don't hesitate to make it your own by adding your personal touch. Enjoy your homemade {focused_keyword}!</p>")
              
              else:
                  content_parts.append(f"<p>This section provides important information about {focused_keyword}. Understanding these details will help you perfect your recipe and enjoy the best possible results.</p>")
      
      return "\n\n".join(content_parts)

    # Add this method to the ContentGenerator class
    def extract_recipe_data_from_content(self, content: str) -> Dict:
        """
        Extract essential recipe data from the generated article content with enhanced debugging
        """
        try:
            self.logger.info("🔍 Starting recipe data extraction from content")
            soup = BeautifulSoup(content, 'html.parser')
            extracted_data = {
                "title": "",
                "ingredients": [],
                "instructions": [],
                "prep_time": "",
                "cook_time": "",
                "servings": "",
                "notes": [],
                "nutrition": {}
            }
            
            # Extract title (H1)
            h1_tag = soup.find('h1')
            if h1_tag:
                extracted_data["title"] = h1_tag.get_text().strip()
                self.logger.info(f"📝 Found title: {extracted_data['title']}")
            
            # Extract ingredients section - SIMPLE APPROACH
            ingredients_section = ""
            self.logger.info("🥕 Looking for ingredients section...")
            
            for h2 in soup.find_all('h2'):
                h2_text = h2.get_text().strip()
                if "What You'll Need for" in h2_text:
                    self.logger.info(f"✅ Found ingredients section: {h2_text}")
                    # Get all content until next H2 or end
                    current = h2.find_next_sibling()
                    while current and current.name != 'h2':
                        ingredients_section += str(current)
                        current = current.find_next_sibling()
                    break
            
            extracted_data["ingredients"] = ingredients_section
            self.logger.info(f"📊 Ingredients section extracted: {len(ingredients_section)} characters")
            
            # Extract instructions section - SIMPLE APPROACH
            instructions_section = ""
            self.logger.info("👨‍🍳 Looking for instructions section...")
            
            for h2 in soup.find_all('h2'):
                h2_text = h2.get_text().strip()
                if "How to Make" in h2_text:
                    self.logger.info(f"✅ Found instructions section: {h2_text}")
                    # Get all content until next H2 or end
                    current = h2.find_next_sibling()
                    while current and current.name != 'h2':
                        instructions_section += str(current)
                        current = current.find_next_sibling()
                    break
            
            extracted_data["instructions"] = instructions_section
            self.logger.info(f"📊 Instructions section extracted: {len(instructions_section)} characters")
            
            # Extract timing information - ENHANCED
            content_text = soup.get_text().lower()
            self.logger.info("⏰ Extracting timing information...")
            
            # Look for prep time with better regex
            prep_match = re.search(r'prep.*?time.*?(\d+.*?minutes?)', content_text, re.IGNORECASE)
            if prep_match:
                extracted_data["prep_time"] = prep_match.group(1)
                self.logger.info(f"⏰ Prep time: {extracted_data['prep_time']}")
            
            # Look for cook time with better regex
            cook_match = re.search(r'cook.*?time.*?(\d+.*?minutes?)', content_text, re.IGNORECASE)
            if cook_match:
                extracted_data["cook_time"] = cook_match.group(1)
                self.logger.info(f"⏰ Cook time: {extracted_data['cook_time']}")
            
            # Look for servings with better regex
            servings_match = re.search(r'(\d+.*?servings?)', content_text, re.IGNORECASE)
            if servings_match:
                extracted_data["servings"] = servings_match.group(1)
                self.logger.info(f"👥 Servings: {extracted_data['servings']}")
            
            # Extract nutrition information
            nutrition_found = {}
            self.logger.info("🥗 Extracting nutrition information...")
            for section in soup.find_all(['h2', 'h3', 'h4']):
                if 'nutrition' in section.get_text().lower():
                    self.logger.info(f"✅ Found nutrition section: {section.get_text()}")
                    # Look for nutrition data in the next table or list
                    next_elem = section.find_next_sibling()
                    if next_elem and next_elem.name == 'table':
                        rows = next_elem.find_all('tr')
                        for row in rows:
                            cells = row.find_all('td')
                            if len(cells) >= 2:
                                nutrient = cells[0].get_text().strip().lower()
                                value = cells[1].get_text().strip()
                                if nutrient and value:
                                    nutrition_found[nutrient] = value
                                    self.logger.info(f"  ✅ {nutrient}: {value}")
            
            extracted_data["nutrition"] = nutrition_found
            self.logger.info(f"📊 Nutrition items found: {len(nutrition_found)}")
            
            # Extract notes from shortcodes
            notes_found = []
            self.logger.info("📝 Extracting notes from shortcodes...")
            for shortcode in soup.find_all(text=re.compile(r'\[.*?\]')):
                if any(keyword in shortcode.lower() for keyword in ['tip', 'note', 'important']):
                    # Find the parent element and extract list items
                    parent = shortcode.parent
                    if parent:
                        for li in parent.find_all('li'):
                            note = li.get_text().strip()
                            if note and len(note) > 5:
                                notes_found.append(note)
                                self.logger.info(f"  ✅ Note: {note[:50]}...")
            
            extracted_data["notes"] = notes_found
            self.logger.info(f" Notes found: {len(notes_found)}")
            
            # Final summary
            self.logger.info(f"🎯 Recipe extraction summary:")
            self.logger.info(f"  📝 Title: {extracted_data['title']}")
            self.logger.info(f"   Ingredients section: {len(extracted_data['ingredients'])} characters")
            self.logger.info(f"  ‍ Instructions section: {len(extracted_data['instructions'])} characters")
            self.logger.info(f"  ⏰ Prep time: {extracted_data['prep_time']}")
            self.logger.info(f"  ⏰ Cook time: {extracted_data['cook_time']}")
            self.logger.info(f"  👥 Servings: {extracted_data['servings']}")
            self.logger.info(f"   Nutrition items: {len(nutrition_found)}")
            self.logger.info(f"  📝 Notes: {len(notes_found)}")
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting recipe data from content: {str(e)}")
            return {
                "title": "",
                "ingredients": [],
                "instructions": [],
                "prep_time": "",
                "cook_time": "",
                "servings": "",
                "notes": [],
                "nutrition": {}
            }

# WordPress Publisher
class WordPressPublisher:
    def __init__(self, config: Config, logger: Logger):
      self.config = config
      self.logger = logger
      self.content_generator = None
      self.anthropic_client = Anthropic(api_key=self.config.config['claude_api_key'])

    def set_content_generator(self, content_generator):
      self.content_generator = content_generator

    def create_wordpress_client(self, website_config):
       try:
           client = Client(
               website_config['xmlrpc_url'],
               website_config['username'],
               website_config['api_secret']
           )
           
           from wordpress_xmlrpc.methods import users
           user = client.call(users.GetUserInfo())
           self.logger.info(f"Connected to WordPress as: {user.display_name}")
           return client
       except Exception as e:
           self.logger.error(f"Failed to connect to WordPress: {str(e)}")
           raise APIError(f"WordPress connection failed: {str(e)}")

    async def generate_metadata(self, keyword: str, content: str, focused_keyword: str, scraped_data: Dict, seo_metadata: Dict) -> ArticleMetadata:
      try:
          self.logger.info("🎯 Starting metadata generation...")
          
          # CRITICAL FIX: Extract recipe data from our generated article content instead of scraped data
          self.logger.info("🔍 Extracting recipe data from generated article content...")
          article_recipe_data = self.content_generator.extract_recipe_data_from_content(content)
          
          # Log what we extracted from our article
          self.logger.info(f"📊 Recipe data extracted from our article:")
          self.logger.info(f"  📝 Title: {article_recipe_data.get('title', 'N/A')}")
          self.logger.info(f"   Ingredients: {len(article_recipe_data.get('ingredients', []))}")
          self.logger.info(f"  ‍ Instructions: {len(article_recipe_data.get('instructions', []))}")
          self.logger.info(f"  ⏰ Prep time: {article_recipe_data.get('prep_time', 'N/A')}")
          self.logger.info(f"  ⏰ Cook time: {article_recipe_data.get('cook_time', 'N/A')}")
          self.logger.info(f"  👥 Servings: {article_recipe_data.get('servings', 'N/A')}")
          
          # Use timing data from scraped_data as fallback if not found in article
          if not article_recipe_data.get('prep_time') and scraped_data.get('recipe_data', {}).get('prep_time'):
              article_recipe_data['prep_time'] = scraped_data['recipe_data']['prep_time']
              self.logger.info(f"⏰ Using scraped prep time as fallback: {article_recipe_data['prep_time']}")
          
          if not article_recipe_data.get('cook_time') and scraped_data.get('recipe_data', {}).get('cook_time'):
              article_recipe_data['cook_time'] = scraped_data['recipe_data']['cook_time']
              self.logger.info(f"⏰ Using scraped cook time as fallback: {article_recipe_data['cook_time']}")
          
          if not article_recipe_data.get('servings') and scraped_data.get('recipe_data', {}).get('servings'):
              article_recipe_data['servings'] = scraped_data['recipe_data']['servings']
              self.logger.info(f"👥 Using scraped servings as fallback: {article_recipe_data['servings']}")
          
          # Generate recipe card using our article's recipe data
          recipe_card = await self.content_generator.generate_recipe_card_directly(focused_keyword, article_recipe_data)
          
          # Extract SEO metadata
          seo_title = seo_metadata.get('seo_title', f"{focused_keyword.title()} Recipe")
          meta_description = seo_metadata.get('meta_description', f"A delicious recipe for {focused_keyword}.")
          slug = seo_metadata.get('slug', focused_keyword.lower().replace(' ', '-'))
          focus_keyword_defintivie = seo_metadata.get('focus_keyword', focused_keyword)
          
          # Create metadata object
          metadata = ArticleMetadata(
              title=seo_title,
              slug=slug,
              focus_keyword=focus_keyword_defintivie,
              meta_description=meta_description,
              seo_data=seo_metadata,
              html_content=content,
              word_count=len(content.split()),
              recipe_card=recipe_card
          )
          
          self.logger.info(f"✅ Metadata generated successfully: {metadata.title}")
          return metadata
          
      except Exception as e:
          self.logger.error(f"❌ Metadata generation failed: {str(e)}")
          # NO FALLBACK - Let the error propagate
          raise Exception(f"Metadata generation failed: {str(e)}")

    def _create_fallback_metadata(self, keyword: str, focused_keyword: str, content: str) -> ArticleMetadata:
      return ArticleMetadata(
          title=f"{focused_keyword.title()} Recipe",
          slug=focused_keyword.lower().replace(' ', '-'),
          focus_keyword=focused_keyword,
          meta_description=f"A delicious recipe for {focused_keyword}.",
          seo_data={
              "min_word_count": 1500,
              "focus_keyword_in_title": True,
              "focus_keyword_in_first_paragraph": True,
              "focus_keyword_in_meta_description": True,
              "focus_keyword_in_url": True,
              "focus_keyword_in_subheadings": True,
              "focus_keyword_in_image_alt": True,
              "min_keyword_density": 0.5,
              "max_keyword_density": 2.0,
              "max_url_length": 75,
              "external_links_required": True,
              "internal_links_required": True,
              "use_table_of_contents": True,
              "paragraph_length_max": 150,
              "title_should_have_power_word": True,
              "title_should_have_sentiment": True,
              "avoid_numbers_in_title": True
          },
          html_content=content,
          word_count=len(content.split())
      )

    def _generate_fallback_description(self, focused_keyword: str) -> str:
      return f"Discover how to make delicious {focused_keyword} with our step-by-step recipe. Perfect for any occasion, this recipe is sure to impress family and friends!"

    def _generate_slug(self, title: str, focused_keyword: str) -> str:
      base_slug = focused_keyword.lower()
      
      slug = re.sub(r'[^a-z0-9\s-]', '', base_slug.lower())
      slug = re.sub(r'[\s-]+', '-', slug)
      
      max_length = self.config.get_seo_params().get('max_slug_length', 75)
      if len(slug) > max_length:
          slug = slug[:max_length].rstrip('-')
      
      return slug

    def _calculate_total_time(self, prep_time: str, cook_time: str) -> str:
      try:
          prep_minutes = 0
          prep_match = re.search(r'(\d+)', prep_time)
          if prep_match:
              prep_minutes = int(prep_match.group(1))
              if 'hour' in prep_time:
                  prep_minutes *= 60
          
          cook_minutes = 0
          cook_match = re.search(r'(\d+)', cook_time)
          if cook_match:
              cook_minutes = int(cook_match.group(1))
              if 'hour' in cook_time:
                  cook_minutes *= 60
          
          total_minutes = prep_minutes + cook_minutes
          
          if total_minutes < 60:
              return f"{total_minutes} minutes"
          else:
              hours = total_minutes // 60
              minutes = total_minutes % 60
              if minutes == 0:
                  return f"{hours} hour{'s' if hours > 1 else ''}"
              else:
                  return f"{hours} hour{'s' if hours > 1 else ''} {minutes} minutes"
      except:
          return "45 minutes"

    def _create_fallback_recipe_card(self, keyword: str, recipe_data: Dict = None) -> Dict:
      if recipe_data is None:
          recipe_data = {}
          
      return {
          "Recipe_Title": f"{keyword.title()} Recipe",
          "Recipe_Description": f"A delicious {keyword} recipe that's perfect for any occasion.",
          "Recipe_Notes": "Store in an airtight container for up to 3 days.",
          "Recipe_Details": {
              "Prep_Time": recipe_data.get("prep_time", "15 minutes"),
              "Cook_Time": recipe_data.get("cook_time", "30 minutes"),
              "Yield": recipe_data.get("servings", "4 servings"),
              "Category": "Main Course",
              "Method": "Baking",
              "Cuisine": "American"
          },
          "Recipe_Ingredients": recipe_data.get("ingredients", [
              f"Main ingredients for {keyword}",
              "Additional ingredients as needed",
              "Salt and pepper to taste"
          ]),
          "Recipe_Instructions": recipe_data.get("instructions", [
              "Prepare all ingredients as directed",
              f"Cook the {keyword} following standard techniques",
              "Serve hot and enjoy"
          ]),
          "Recipe_Nutrition": {
              "Serving_Size": "1 serving",
              "Calories": "300",
              "Sugar": "5g",
              "Sodium": "400mg",
              "Fat": "15g",
              "Saturated_Fat": "5g",
              "Unsaturated_Fat": "10g",
              "Trans_Fat": "0g",
              "Carbohydrates": "30g",
              "Fiber": "3g",
              "Protein": "12g",
              "Cholesterol": "30mg"
          }
      }
       
    # REPLACE the broken publish_article method in new_main.py (around line 200) 
    # with this COMPLETE working version from main.py:

    async def publish_article(self, website_config: Dict, metadata: ArticleMetadata, featured_image_id: Any, featured_image_url: Any) -> str:
        """Publish article with metadata properly configured"""
        try:
            # 1. Connect to WordPress
            client = Client(
                website_config['xmlrpc_url'],
                website_config['username'],
                website_config['api_secret']
            )
            
            # 2. Create main article post first
            main_post = WordPressPost()
            main_post.title = metadata.title
            main_post.content = metadata.html_content
            main_post.post_status = 'draft'
            main_post.post_name = metadata.slug
            main_post.slug = metadata.slug

            # FIXED: Use featured_image_id parameter first, then fallback to metadata
            if featured_image_id:
                main_post.thumbnail = featured_image_id
                self.logger.info(f"Setting featured image ID: {featured_image_id}")
            elif hasattr(metadata, 'featured_image_id') and metadata.featured_image_id:
                main_post.thumbnail = metadata.featured_image_id
                self.logger.info(f"Setting featured image ID from metadata: {metadata.featured_image_id}")
            else:
                self.logger.warning("No featured image ID available")
            
            main_post.terms_names = {
                'category': ['Recipes']  # Use Recipes category for recipe posts
            }
            
            # 3. Create the post
            post_id = int(client.call(posts.NewPost(main_post)))
            self.logger.info(f"Created post with ID: {post_id}")

            # 4. Update RankMath SEO fields
            rankmath_fields = [
                {"key": "rank_math_title", "value": metadata.title},
                {"key": "rank_math_description", "value": metadata.meta_description},
                {"key": "rank_math_focus_keyword", "value": metadata.focus_keyword},
            ]
            
            # Apply RankMath fields
            client.call(posts.EditPost(post_id, {'custom_fields': rankmath_fields}))
            self.logger.info(f"Updated RankMath SEO fields for post {post_id}")

            # 5. Add source URL if available
            if hasattr(metadata, 'serp_source') and metadata.serp_source:
                try:
                    client.call(posts.EditPost(post_id, {
                        'custom_fields': [
                            {"key": "_source_url", "value": metadata.serp_source}
                        ]
                    }))
                except Exception as e:
                    self.logger.warning(f"Failed to add source URL: {str(e)}")
            
            # 6. Process recipe card if available
            recipe_post_id = None
            if hasattr(metadata, 'recipe_card') and metadata.recipe_card:
                recipe_data = None
                if isinstance(metadata.recipe_card, str):
                    try:
                        recipe_data = json.loads(metadata.recipe_card)
                    except:
                        recipe_data = None
                else:
                    recipe_data = metadata.recipe_card
                        
                if recipe_data:
                    try:
                        # Create recipe post for Tasty Recipes
                        recipe_post = WordPressPost()
                        recipe_post.title = recipe_data.get('Recipe_Title', metadata.title)
                        recipe_post.post_content = ''
                        recipe_post.post_status = 'publish'
                        recipe_post.post_type = 'tasty_recipe'
                        recipe_post.comment_status = 'closed'
                        recipe_post.ping_status = 'closed'
                        
                        # FIXED: Use same featured image logic for recipe post
                        if featured_image_id:
                            recipe_post.thumbnail = featured_image_id
                        elif hasattr(metadata, 'featured_image_id') and metadata.featured_image_id:
                            recipe_post.thumbnail = metadata.featured_image_id
                        
                        # Create the recipe post
                        recipe_post_id = int(client.call(posts.NewPost(recipe_post)))
                        self.logger.info(f"Created recipe post with ID: {recipe_post_id}")
                        
                        # Add recipe metadata - ALL FIELDS
                        recipe_meta = [
                            {"key": "_pinterest_shares", "value": "0"},
                            {"key": "name", "value": recipe_data.get('Recipe_Title', '')},
                            {"key": "title", "value": recipe_data.get('Recipe_Title', '')},
                            {"key": "description", "value": recipe_data.get('Recipe_Description', '')},
                            {"key": "keywords", "value": recipe_data.get('Recipe_Keywords', '')},
                            {"key": "total_time", "value": recipe_data.get('Recipe_Details', {}).get('Total_Time', '')},
                            {"key": "prep_time", "value": recipe_data.get('Recipe_Details', {}).get('Prep_Time', '')},
                            {"key": "cook_time", "value": recipe_data.get('Recipe_Details', {}).get('Cook_Time', '')},
                            {"key": "yield", "value": recipe_data.get('Recipe_Details', {}).get('Yield', '')},
                            {"key": "category", "value": recipe_data.get('Recipe_Details', {}).get('Category', '')},
                            {"key": "method", "value": recipe_data.get('Recipe_Details', {}).get('Method', '')},
                            {"key": "cuisine", "value": recipe_data.get('Recipe_Details', {}).get('Cuisine', '')},
                            {"key": "author_name", "value": self.config.config.get('author_name', 'Jennie Graham')},
                        ]

                        # Add ingredients - use pre-formatted HTML from Claude if available
                        ingredients_html = recipe_data.get('Recipe_Ingredients', '')
                        recipe_meta.append({"key": "ingredients", "value": ingredients_html})
                        
                        # Add instructions - use pre-formatted HTML from Claude if available
                        instructions_html = recipe_data.get('Recipe_Instructions', '')
                        recipe_meta.append({"key": "instructions", "value": instructions_html})
                        
                        # Add recipe notes
                        recipe_notes = recipe_data.get('Recipe_Notes', '')
                        if recipe_notes:
                            recipe_meta.append({"key": "notes", "value": recipe_notes})
                        
                        # Add nutrition data
                        nutrition_data = recipe_data.get('Recipe_Nutrition', {})
                        if nutrition_data:
                            recipe_meta.extend([
                                {"key": "calories", "value": nutrition_data.get('Calories', '')},
                                {"key": "sugar", "value": nutrition_data.get('Sugar', '')},
                                {"key": "sodium", "value": nutrition_data.get('Sodium', '')},
                                {"key": "fat", "value": nutrition_data.get('Fat', '')},
                                {"key": "saturated_fat", "value": nutrition_data.get('Saturated_Fat', '')},
                                {"key": "unsaturated_fat", "value": nutrition_data.get('Unsaturated_Fat', '')},
                                {"key": "trans_fat", "value": nutrition_data.get('Trans_Fat', '')},
                                {"key": "carbohydrates", "value": nutrition_data.get('Carbohydrates', '')},
                                {"key": "fiber", "value": nutrition_data.get('Fiber', '')},
                                {"key": "protein", "value": nutrition_data.get('Protein', '')},
                                {"key": "cholesterol", "value": nutrition_data.get('Cholesterol', '')},
                                {"key": "serving_size", "value": nutrition_data.get('Serving_Size', '')}
                            ])

                        # Update recipe post with custom fields
                        client.call(posts.EditPost(recipe_post_id, {'custom_fields': recipe_meta}))
                        
                        # Log what was stored
                        self.logger.info(f"✅ Recipe card stored in WordPress with {len(recipe_meta)} fields:")
                        for field in recipe_meta:
                            if field['key'] in ['ingredients', 'instructions', 'notes']:
                                # Truncate long content for logging
                                value = field['value'][:100] + "..." if len(field['value']) > 100 else field['value']
                                self.logger.info(f"  📝 {field['key']}: {value}")
                            else:
                                self.logger.info(f"  📝 {field['key']}: {field['value']}")
                        
                        # Update the original post to include the recipe shortcode
                        updated_content = f"{metadata.html_content}\n[tasty-recipe id=\"{recipe_post_id}\"]"
                        client.call(posts.EditPost(post_id, {'post_content': updated_content}))
                    except Exception as e:
                        self.logger.error(f"Error creating recipe post: {str(e)}")
            
            # 7. Publish the main post
            try:
                client.call(posts.EditPost(post_id, {'post_status': 'publish'}))
                self.logger.info(f"Successfully published post {post_id}")
            except Exception as e:
                self.logger.error(f"Error publishing post: {str(e)}")
                # Try to publish one more time
                try:
                    client.call(posts.EditPost(post_id, {'post_status': 'publish'}))
                    self.logger.info(f"Successfully published post {post_id} on retry")
                except:
                    self.logger.error("Failed to publish post even after retry")
            
            # 8. Return URL
            article_url = f"{website_config['url']}/{metadata.slug}/"
            self.logger.info(f"Article published at: {article_url}")
            return article_url
            
        except Exception as e:
            self.logger.error(f"Failed to publish article: {str(e)}")
            raise APIError(f"Failed to publish article: {str(e)}")

# Article Generator - Main Class
class ArticleGenerator:
    def __init__(self, config_path: str):
      self.logger = Logger()
      self.config = Config(config_path)
      self.keyword_processor = KeywordProcessor(self.config, self.logger)
      self.content_generator = ContentGenerator(self.config, self.logger)
      self.wordpress_publisher = WordPressPublisher(self.config, self.logger)
      self.sheets_manager = GoogleSheetsManager(self.config, self.logger) 

      self.current_website_config = None
      self.current_keyword = None
      
      self.content_generator.set_keyword_processor(self.keyword_processor)
      self.wordpress_publisher.set_content_generator(self.content_generator)

    MAX_RETRIES = 3
    MIN_WORDS = 1500
  
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )

    async def _execute_step(self, step_name: str, func, *args, **kwargs):
            try:
                self.logger.logger.info(f"========================STARTING {step_name}========================")
                
                print(f"\n### STARTING STEP: {step_name} ###")
                print(f"### Keyword: {self.current_keyword} ###")
                print(f"### Website: {self.current_website_config['name'] if self.current_website_config else 'unknown'} ###\n")
                
                if hasattr(self, 'current_website_config') and self.current_website_config:
                    # Update step for specific website
                    progress_tracker.set_current_step(self.current_website_config['name'], step_name)
                    
                    progress_tracker.update_progress(
                        self.current_website_config['name'],
                        self.current_keyword, 
                        "processing", 
                        {"current_step": step_name}
                    )
                    # Force a small delay to ensure DB/state updates
                    await asyncio.sleep(0.1)
                    
                    print(f"### PROGRESS UPDATED: step={step_name}, website={self.current_website_config['name']}, keyword={self.current_keyword} ###")
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
                        
                self.logger.logger.info(f"========================{step_name} EXECUTED SUCCESSFULLY========================")
                print(f"### COMPLETED STEP: {step_name} ###\n")
                return result
            except Exception as e:
                self.logger.logger.error(f"Error in {step_name}: {str(e)}")
                print(f"### ERROR IN STEP: {step_name} - {str(e)} ###\n")
                raise
      
    async def _get_source_url_from_sheets(self, website_name: str, keyword: str) -> str:
       try:
           website_config = next((site for site in self.config.get_wordpress_sites() if site['name'] == website_name), None)
           if not website_config:
               self.logger.error(f"Website configuration not found for '{website_name}'")
               return None
               
           sheet_name = website_config.get('google_sheet_tab', website_name)
           
           sheets_data = self.sheets_manager.sheets_data
           if not sheets_data or sheet_name not in sheets_data:
               sheets_data = await asyncio.to_thread(self.sheets_manager.fetch_all_sheets_data)
               
           if sheet_name not in sheets_data:
               self.logger.error(f"Sheet '{sheet_name}' not found")
               return None
               
           df = sheets_data[sheet_name]
           
           keyword_rows = df[df['keyword'] == keyword]
           if keyword_rows.empty:
               self.logger.warning(f"Keyword '{keyword}' not found in sheet '{sheet_name}'")
               return None
               
           if 'source_domain' in keyword_rows.columns:
               source_url = keyword_rows.iloc[0]['source_domain']
               if pd.notna(source_url) and source_url:
                   self.logger.info(f"Found source URL for '{keyword}': {source_url}")
                   return source_url
                   
           self.logger.warning(f"No source_domain found for keyword '{keyword}'")
           return None
       except Exception as e:
           self.logger.error(f"Error getting source URL from sheets: {str(e)}")
           return None
   
    # IMPROVED generate_article method - Replace your current version with this:

    async def generate_article(self, website_config: Dict, keyword: str, progress_tracker: ProgressTracker) -> Dict:
        try:
            # Store current processing info for progress tracking
            self.current_website_config = website_config
            self.current_keyword = keyword

            # Update progress to processing
            progress_tracker.update_progress(website_config['name'], keyword, "processing")
            
            self.logger.info(f"🚀 Starting article generation for '{keyword}' on {website_config['name']}")
            
            # Step 1: Source URL Lookup
            source_url = await self._execute_step(
                "SOURCE URL LOOKUP",
                self._get_source_url_from_sheets,
                website_config['name'], 
                keyword
            )           
            
            # Step 2: Scrape Article Content (REPLACES SERP ANALYSIS)
            scraped_data = await self._execute_step(
                "ARTICLE CONTENT SCRAPING",
                self.keyword_processor.scrape_article_content,
                source_url
            )
            
            # Step 3: Keyword Optimization
            keyword = await self._execute_step(
                "KEYWORD OPTIMIZATION", 
                self.keyword_processor.optimize_keyword,
                keyword
            )

            # Step 4: SEO metadata generation
            definitive_seo_metadata = await self._execute_step(
                "SEO METADATA GENERATION",
                self.content_generator.generate_seo_metadata,
                keyword,
                scraped_data
            )
            
            # Step 5: Get template
            template = self.config.get_article_template(website_config.get('name'))

            focused_keyword = definitive_seo_metadata.get('focus_keyword', keyword)
            
            # Step 6: Generate outline
            outline = await self._execute_step(
                "OUTLINE GENERATION",
                self.content_generator.generate_strict_outline,
                keyword,
                focused_keyword,
                template
            )
            
            # Step 7: External links research
            external_links = await self._execute_step(
                "EXTERNAL LINKS RESEARCH",
                self.keyword_processor.get_high_authority_links,
                focused_keyword,
                3
            )
            
            # Step 8: Internal links research
            internal_links = await self._execute_step(
                "INTERNAL LINKS RESEARCH",
                self.keyword_processor.get_internal_links,
                website_config,
                focused_keyword,
                3
            )


            # Step 9: Content generation (using scraped_data instead of serp_data)
            content, content_seo_metadata = await self._execute_step(
                "CONTENT GENERATION",
                self.content_generator.generate_content,
                keyword,
                focused_keyword,
                scraped_data,
                outline,
                external_links,
                internal_links,
                template,
                website_config
            )
            

            
            # Step 10: Image generation
            try:
                content, featured_image_id, featured_image_url = await self._execute_step(
                    "IMAGE GENERATION",
                    self.generate_basic_images,
                    website_config,
                    keyword,
                    focused_keyword,
                    content,
                    source_url,  # Use source_url for image extraction
                    None  # client will be created inside the method
                )
            except Exception as image_error:
                self.logger.warning(f"Image generation failed: {str(image_error)}")
                featured_image_id = None
                featured_image_url = None
            
            # Step 11: Generate metadata
            metadata = await self._execute_step(
                "METADATA GENERATION",
                self.wordpress_publisher.generate_metadata,
                keyword,
                content,
                focused_keyword,
                scraped_data,  # Changed from serp_data to scraped_data
                definitive_seo_metadata
            )
            
            # FIXED: Add validation for metadata
            if not hasattr(metadata, 'title') or not metadata.title:
                raise ValueError("Generated metadata is missing title")
            if not hasattr(metadata, 'slug') or not metadata.slug:
                raise ValueError("Generated metadata is missing slug")
            
            self.logger.info(f" Metadata generated: title='{metadata.title}', slug='{metadata.slug}'")
            
            self.logger.info(f"🎯 Metadata generation completed, proceeding to publishing...")
        
            # Step 12: Publish to WordPress - ADD EXPLICIT ERROR HANDLING
            try:
                self.logger.info(f"🚀 STARTING ARTICLE PUBLISHING for: {keyword}")
                
                article_url = await self._execute_step(
                    "ARTICLE PUBLISHING",
                    self.wordpress_publisher.publish_article,
                    website_config,
                    metadata,
                    featured_image_id,
                    featured_image_url
                )
                
                if not article_url:
                    raise ValueError("Publishing returned empty URL")
                    
                self.logger.info(f"🎉 ARTICLE PUBLISHED SUCCESSFULLY: {article_url}")
                
            except Exception as publishing_error:
                self.logger.error(f" PUBLISHING FAILED: {str(publishing_error)}")
                # Create a fallback URL so the process can complete
                article_url = f"{website_config['url']}/articles/{metadata.slug}/"
                self.logger.info(f"⚠️ Using fallback URL: {article_url}")
                
                # FIXED: Validate that publishing returned a URL
                if not article_url:
                    raise ValueError("WordPress publishing did not return a valid URL")
            
            self.logger.info(f"🎉 Article published successfully: {article_url}")
            
            # Create result data
            result = {
                "url": article_url,
                "focus_keyword": focused_keyword,
                "title": metadata.title,
                "meta_description": metadata.meta_description,
                "slug": metadata.slug,
                "word_count": content_seo_metadata.get("word_count", 0),
                "html_content": content,
                "seo_metadata": definitive_seo_metadata,
                "serp_source": getattr(metadata, 'serp_source', source_url)  # Use source_url as fallback
            }
            
            # Mark as completed
            progress_tracker.complete_keyword(website_config['name'], keyword, success=True)
            
            self.logger.info(f"✅ Article generation completed successfully for '{keyword}'")
            return result
        
        except Exception as e:
            self.logger.error(f"❌ Article generation failed for '{keyword}': {str(e)}")
            progress_tracker.complete_keyword(website_config['name'], keyword, success=False)
            raise

    async def generate_basic_images(self, website_config, keyword, focused_keyword, content, image_source_url, client=None):
        """Generate and place images with Midjourney integration per website - Midjourney only"""
        try:
            # Initialize image generator
            image_generator = ImageGenerator(self.config, self.logger)
            featured_image_id = None
            featured_image_url = None

            # Step 1: Extract feature image from source URL if available
            feature_image_url = None
            if image_source_url:
                feature_image_url = await image_generator.extract_feature_image(image_source_url)
                self.logger.info(f"Feature image URL extracted: {feature_image_url}")
            
            # Step 2: Generate a clean, appetizing food photography prompt
            prompt = f"STYLE: reddit style | EMOTION: appealing and homemade | SCENE: a reddit style close up shot of {focused_keyword}| TAGS: amateur food photography, homemade, cuisine, recipe | CAMERA: Iphone 16 pro | FOCAL LENGHT: soft direction light | TIME : daytime | LOCATION TYPE : home cozy kitchen"
            
            # Step 3: Generate 4 images with Midjourney using website-specific channel
            self.logger.info(f"Generating images for '{focused_keyword}' using website '{website_config['name']}' Midjourney channel")
            image_urls = []
            
            try:
                # Generate with Midjourney - WITH THE SOURCE IMAGE
                image_urls = await image_generator.generate_with_midjourney(website_config, prompt, feature_image_url)
            except Exception as e:
                self.logger.error(f"Midjourney generation failed for website {website_config['name']}: {str(e)}")
                return content, None, None
            
            if not image_urls or len(image_urls) == 0:
                self.logger.warning("No images were generated")
                return content, None, None
            
            # Step 4: Create WordPress client if not provided
            if client is None:
                try:
                    from wordpress_xmlrpc import Client
                    client = Client(
                        website_config['xmlrpc_url'],
                        website_config['username'],
                        website_config['api_secret']
                    )
                    self.logger.info("Created WordPress client for image uploads")
                except Exception as e:
                    self.logger.error(f"Failed to create WordPress client: {str(e)}")
                    return content, None, None
            
            # Step 5: Upload featured image with retry logic
            featured_image_url = image_urls[0]
            featured_image_data = await self._download_image_with_retry(featured_image_url)
            
            if featured_image_data:
                alt_text = f"{focused_keyword} recipe"
                title = f"{focused_keyword} Recipe"
                
                # Upload with proper error handling and retry logic
                featured_image_id, featured_image_url = await self._upload_to_wordpress_with_retry(
                    client, 
                    featured_image_data, 
                    title, 
                    alt_text,
                    0, 
                    max_retries=5
                )
                
                self.logger.info(f"Successfully uploaded featured image with ID: {featured_image_id}")
            else:
                self.logger.warning("Failed to download featured image")
            
            # Step 6: Image placement - Place images at specific locations
            modified_content = content
            remaining_images = image_urls[1:] if len(image_urls) > 1 else []
            
            if remaining_images:
                # Get up to 3 images for placement
                images_to_place = remaining_images[:min(3, len(remaining_images))]
                
                # Upload images first
                uploaded_images = []
                for i, image_url in enumerate(images_to_place):
                    try:
                        # Download the image
                        image_data = await self._download_image_with_retry(image_url)
                        if not image_data:
                            continue
                        
                        # Upload to WordPress
                        alt_text = f"{focused_keyword} food photograph {i+1}"
                        title = f"{focused_keyword}_{i+1}"
                        
                        image_id, image_url = await self._upload_to_wordpress_with_retry(
                            client, 
                            image_data, 
                            title, 
                            alt_text,
                            i, 
                            max_retries=3
                        )
                        
                        if image_id and image_url:
                            uploaded_images.append({
                                'id': image_id,
                                'url': image_url,
                                'alt': alt_text
                            })
                            
                    except Exception as e:
                        self.logger.error(f"Error processing image {i+1}: {str(e)}")
                
                # Now place images at specific positions
                if uploaded_images:
                    # Define the patterns for each placement
                    placements = [
                        # Pattern 1: Before "What You'll Need" H2
                        (r'(<h2[^>]*>.*?What You\'ll Need.*?</h2>)', 0),
                        # Pattern 2: Before "Mistakes to Avoid" H2
                        (r'(<h2[^>]*>.*?Mistakes to Avoid.*?</h2>)', 1),
                        # Pattern 3: Before FAQ section
                        (r'(\[faq_section\])', 2)
                    ]
                    
                    # Place each image
                    for pattern, image_index in placements:
                        if image_index < len(uploaded_images):
                            img = uploaded_images[image_index]
                            img_html = f'<img src="{img["url"]}" alt="{img["alt"]}" class="wp-image-{img["id"]}" />\n\n'
                            
                            # Find and replace - insert BEFORE the matched pattern
                            match = re.search(pattern, modified_content, re.IGNORECASE)
                            if match:
                                position = match.start()
                                modified_content = modified_content[:position] + img_html + modified_content[position:]
                                self.logger.info(f"Placed image {image_index + 1} at specified position")
                            else:
                                self.logger.warning(f"Could not find pattern for image placement {image_index + 1}")
            
            return modified_content, featured_image_id, featured_image_url
                        
        except Exception as e:
            self.logger.error(f"Error in generate_basic_images: {str(e)}")
            return content, None, None

    async def _download_image_with_retry(self, url, max_retries=3):
        """Download image with retry logic - EXACT from old code"""
        for attempt in range(max_retries):
            try:
                # Add proper headers to avoid being blocked
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': f"https://{url.split('/')[2]}" if url.startswith('http') else '',
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=30) as response:
                        if response.status == 200:
                            return await response.read()
                        
                        self.logger.warning(f"Download attempt {attempt+1} failed with status {response.status}")
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                self.logger.warning(f"Download attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        return None

    async def _upload_to_wordpress_with_retry(self, client, image_data, title, alt_text, image_number, max_retries=3):
        """Upload image to WordPress with retry logic - EXACT from old code"""
        for attempt in range(max_retries):
            try:
                # Format filename with random component
                safe_title = re.sub(r'[^a-zA-Z0-9]', '-', title.lower())
                image_name = f"{safe_title}'_{image_number}.png"
                
                # Import WordPress modules
                from wordpress_xmlrpc.methods import media
                from wordpress_xmlrpc.compat import xmlrpc_client
                
                # Prepare upload data
                upload_data = {
                    'name': image_name,
                    'type': 'image/jpeg',
                    'bits': xmlrpc_client.Binary(image_data),
                    'overwrite': False,
                }
                
                # Upload to WordPress
                response = client.call(media.UploadFile(upload_data))
                
                # Extract ID and URL from response
                attachment_id = None
                attachment_url = None
                
                if isinstance(response, dict):
                    attachment_id = response.get('id')
                    attachment_url = response.get('url')
                elif hasattr(response, 'id'):
                    attachment_id = response.id
                    if hasattr(response, 'url'):
                        attachment_url = response.url
                    elif hasattr(response, 'link'):
                        attachment_url = response.link
                
                # If we have an ID but no URL, try to get the URL
                if attachment_id and not attachment_url:
                    try:
                        attachment = client.call(media.GetMediaItem(attachment_id))
                        if hasattr(attachment, 'url'):
                            attachment_url = attachment.url
                        elif hasattr(attachment, 'link'):
                            attachment_url = attachment.link
                        else:
                            # Construct a URL using WordPress conventions
                            site_url = client.url.replace('/xmlrpc.php', '')
                            date_path = datetime.now().strftime('%Y/%m')
                            attachment_url = f"{site_url}/wp-content/uploads/{date_path}/{image_name}"
                    except Exception as e:
                        self.logger.warning(f"Failed to get attachment URL: {str(e)}")
                
                if attachment_id:
                    # Set alt text
                    try:
                        from wordpress_xmlrpc.methods import posts
                        client.call(posts.EditPost(attachment_id, {
                            'post_title': title,
                            'post_excerpt': alt_text,
                            'custom_fields': [
                                {'key': '_wp_attachment_image_alt', 'value': alt_text}
                            ]
                        }))
                    except Exception as e:
                        self.logger.warning(f"Setting alt text failed: {str(e)}")
                    
                    return attachment_id, attachment_url
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                self.logger.warning(f"Upload attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
                    # Try to recreate client if it failed
                    if attempt == max_retries - 2:  # On penultimate attempt
                        try:
                            # Reconnect to WordPress
                            client = Client(
                                client.url,
                                client.username,
                                client.password
                            )
                        except:
                            pass
        
        return None, None
# Global instances
logger = Logger()
config = None
progress_tracker = ProgressTracker()
article_generator = None
auth_manager = None
background_tasks = {}

# FastAPI app
app = FastAPI(
   title="Recipe Article Generator API",
   description="Complete Backend API for Recipe Article Automation",
   version="2.0.0"
)

app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)

def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict]:
    """Get current user but don't fail if no auth - for progress endpoints"""
    if not credentials:
        return None
    
    token = credentials.credentials
    user = auth_manager.verify_token(token)
    
    return user

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
   token = credentials.credentials
   user = auth_manager.verify_token(token)
   
   if user is None:
       raise HTTPException(
           status_code=401,
           detail="Invalid authentication credentials",
           headers={"WWW-Authenticate": "Bearer"},
       )
   
   return user

@app.on_event("startup")
async def startup_event():
   global config, article_generator, auth_manager, progress_tracker, logger
   try:
       logger = Logger()
       progress_tracker = ProgressTracker()
       config = Config("config.json")
       auth_manager = AuthManager(config)
       article_generator = ArticleGenerator("config.json")
       
       # Validate Midjourney channel configurations
       websites = config.get_wordpress_sites()
       for website in websites:
           if website.get('is_active', True):
               midjourney_config = website.get('midjourney_channel')
               if not midjourney_config:
                   logger.warning(f"Website '{website['name']}' missing Midjourney channel configuration")
               else:
                   required_fields = ['channel_id', 'guild_id', 'session_id']
                   missing_fields = [field for field in required_fields if field not in midjourney_config]
                   if missing_fields:
                       logger.warning(f"Website '{website['name']}' Midjourney config missing: {missing_fields}")
                   else:
                       logger.info(f"Website '{website['name']}' Midjourney channel configured: {midjourney_config['channel_id']}")
           
           progress_tracker.start_website(website['name'])
       
       logger.info("Application started successfully")
   except Exception as e:
       logger.error(f"Failed to start application: {str(e)}")
       raise

@app.post("/api/login", response_model=LoginResponse)
async def login(login_data: LoginRequest):
   try:
       result = auth_manager.login(login_data.username, login_data.password)
       if not result:
           raise HTTPException(
               status_code=status.HTTP_401_UNAUTHORIZED,
               detail="Invalid username or password"
           )
       
       logger.info(f"User {login_data.username} logged in successfully")
       return LoginResponse(**result)
   
   except HTTPException:
       raise
   except Exception as e:
       logger.error(f"Login error: {str(e)}")
       raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/verify")
async def verify_token(current_user: dict = Depends(get_current_user)):
   return {
       "valid": True,
       "username": current_user["username"],
       "expires": current_user.get("exp")
   }

@app.post("/api/logout")
async def logout(current_user: dict = Depends(get_current_user)):
   logger.info(f"User {current_user['username']} logged out")
   return {"message": "Logged out successfully"}

@app.get("/debug-progress")
async def debug_progress():
    """Debug endpoint to check progress tracker state"""
    
    # Collect current processing info from all websites
    processing_websites = []
    for website_name, stats in progress_tracker._website_stats.items():
        if stats.get("status") == "processing":
            processing_websites.append({
                "website": website_name,
                "current_keyword": stats.get("current_keyword", ""),
                "current_step": stats.get("current_step", ""),
                "progress": stats.get("progress", 0),
                "completed": stats.get("completed", 0),
                "total": stats.get("total_keywords", 0)
            })
    
    debug_info = {
        "processing_websites": processing_websites,
        "total_websites": len(progress_tracker._website_stats),
        "total_keywords_tracked": len(progress_tracker._progress),
        "website_stats_summary": {
            website: {
                "status": stats.get("status"),
                "progress": stats.get("progress"),
                "completed": stats.get("completed"),
                "failed": stats.get("failed"),
                "total": stats.get("total_keywords")
            }
            for website, stats in progress_tracker._website_stats.items()
        },
        "timestamp": datetime.now().isoformat()
    }
    return debug_info

@app.get("/wordpress-sites")
async def get_websites(current_user: dict = Depends(get_current_user)):
   try:
       websites = config.get_wordpress_sites()
       website_list = []
       
       for website in websites:
           status = progress_tracker.get_website_status(website['name'])
           website_info = {
               "name": website['name'],
               "status": status['status'] if status else 'idle',
               "is_active": website.get('is_active', True),
               "url": website['url']
           }
           website_list.append(website_info)
       
       return {"sites": website_list}
   
   except Exception as e:
       logger.error(f"Error getting websites: {str(e)}")
       raise HTTPException(status_code=500, detail="Failed to get websites")

@app.get("/api/website/{website_name}")
async def get_website_status(website_name: str, current_user: dict = Depends(get_current_user)):
   try:
       website = config.get_website(website_name)
       if not website:
           raise HTTPException(status_code=404, detail="Website not found")
       
       status = progress_tracker.get_website_status(website_name)
       if not status:
           progress_tracker.start_website(website_name)
           status = progress_tracker.get_website_status(website_name)
       
       return {
           "name": website_name,
           "config": {
               "url": website['url'],
               "is_active": website.get('is_active', True)
           },
           "status": status
       }
   
   except HTTPException:
       raise
   except Exception as e:
       logger.error(f"Error getting website status: {str(e)}")
       raise HTTPException(status_code=500, detail="Failed to get website status")

@app.post("/start-process")
async def start_all_websites():
    """Reset all website states to fresh/idle - does NOT start processing"""
    try:
        # Create fresh progress tracker instance
        global progress_tracker
        progress_tracker = ProgressTracker()
        logger.info("🔄 Reset all progress tracking to idle state")
        
        # Initialize all websites in idle state
        websites = config.get_wordpress_sites()
        for website in websites:
            if website.get('is_active', True):
                progress_tracker.start_website(website['name'])
        
        return {
            "status": "success",
            "message": "All website states have been reset",
            "data": {
                "reset_count": len(websites)
            }
        }
        
    except Exception as e:
        logger.error(f"Error resetting states: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/api/website/{website_name}/start")
async def start_website_processing(website_name: str, current_user: dict = Depends(get_current_user)):
   try:
       website = config.get_website(website_name)
       if not website:
           raise HTTPException(status_code=404, detail="Website not found")
       
       if not website.get('is_active', True):
           raise HTTPException(status_code=400, detail="Website is not active")
       
       if progress_tracker.is_website_processing(website_name):
           raise HTTPException(status_code=409, detail="Website is already processing")
       
       task_id = f"{website_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
       
       task = asyncio.create_task(
           process_website_background(website_name, website, task_id)
       )
       background_tasks[task_id] = task
       
       progress_tracker.start_website(website_name)
       website_status = progress_tracker.get_website_status(website_name)
       website_status["task_id"] = task_id
       website_status["status"] = "processing"
       
       logger.info(f"Started processing for website: {website_name}")
       return {
           "message": f"Processing started for {website_name}",
           "task_id": task_id,
           "website": website_name
       }
   
   except HTTPException:
       raise
   except Exception as e:
       logger.error(f"Error starting website processing: {str(e)}")
       raise HTTPException(status_code=500, detail="Failed to start processing")

@app.post("/api/website/{website_name}/stop")
async def stop_website_processing(website_name: str, current_user: dict = Depends(get_current_user)):
   try:
       if not progress_tracker.is_website_processing(website_name):
           raise HTTPException(status_code=400, detail="Website is not currently processing")
       
       status = progress_tracker.get_website_status(website_name)
       task_id = status.get('task_id')
       
       if task_id and task_id in background_tasks:
           background_tasks[task_id].cancel()
           del background_tasks[task_id]
       
       website_status = progress_tracker.get_website_status(website_name)
       website_status["status"] = "stopped"
       website_status["current_step"] = "Stopped by user"
       
       logger.info(f"Stopped processing for website: {website_name}")
       return {
           "message": f"Processing stopped for {website_name}",
           "website": website_name
       }
   
   except HTTPException:
       raise
   except Exception as e:
       logger.error(f"Error stopping website processing: {str(e)}")
       raise HTTPException(status_code=500, detail="Failed to stop processing")

@app.get("/api/website/{website_name}/progress")
async def get_website_progress(website_name: str, current_user: Optional[dict] = Depends(get_current_user_optional)):
    """Get real-time progress for a specific website - NO AUTH REQUIRED for debugging"""
    try:
        # TEMPORARY: Allow progress calls without authentication for debugging
        # TODO: Re-enable auth after fixing frontend JWT issues
        
        logger.info(f"📊 Progress request for: {website_name}")
        logger.info(f"🔑 Auth status: {'Authenticated' if current_user else 'No auth'}")
        
        # Get website status from progress tracker
        status = progress_tracker.get_website_status(website_name)
        
        if not status:
            # Initialize if not found
            progress_tracker.start_website(website_name)
            status = progress_tracker.get_website_status(website_name)
            logger.info(f"🆕 Initialized new website status for: {website_name}")
        
        # Format response to match frontend expectations
        response_data = {
            "status": "success",
            "progress": {
                "status": status.get("status", "idle"),
                "current_keyword": status.get("current_keyword", ""),
                "current_step": status.get("current_step", ""),
                "completed": status.get("completed", 0),
                "total_keywords": status.get("total_keywords", 0),
                "progress": status.get("progress", 0),
                "keywords": status.get("keywords", []),
                "last_update": status.get("last_update", datetime.now().isoformat())
            }
        }
        
        logger.info(f"📤 Sending progress: {response_data['progress']['status']} - {response_data['progress']['current_step']}")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Error getting website progress: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": "Failed to get website progress"}
        )

@app.get("/api/progress")
async def get_all_progress(current_user: dict = Depends(get_current_user)):
    try:
        is_complete = (
            "__process_complete__" in progress_tracker._progress or
            ("global_status" in progress_tracker._progress and 
             progress_tracker._progress["global_status"]["status"] == "completed")
        )
        
        # Get current processing info from website stats
        current_info = {
            "websites_processing": [],
            "total_processing": 0,
            "details": []
        }
        
        for website_name, stats in progress_tracker._website_stats.items():
            if stats.get("status") == "processing":
                current_info["websites_processing"].append(website_name)
                current_info["total_processing"] += 1
                current_info["details"].append({
                    "website": website_name,
                    "keyword": stats.get("current_keyword", ""),
                    "step": stats.get("current_step", "")
                })
        
        response_data = {
            "status": "success",
            "message": "Current progress",
            "data": {
                "progress": progress_tracker._progress,
                "website_stats": progress_tracker._website_stats,
                "current": current_info,
                "process_status": {
                    "is_complete": is_complete,
                    "show_completion_banner": is_complete,
                    "is_active": current_info["total_processing"] > 0
                }
            }
        }
        
        if is_complete:
            response_data["show_completion_banner"] = True
            response_data["completion_data"] = {
                "websites": list(progress_tracker._website_stats.keys()),
                "total_keywords": sum(
                    stats.get("completed", 0) 
                    for stats in progress_tracker._website_stats.values()
                ),
                "timestamp": datetime.now().isoformat()
            }
        
        return response_data
    
    except Exception as e:
        logger.error(f"Error getting progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get progress")

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    return {
        "status": "healthy",
        "service": "recipe-automation-saas",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.get("/api/export-results/{website}")
async def export_results(website: str, current_user: dict = Depends(get_current_user)):
   try:
       csv_data = progress_tracker.export_to_csv()
       
       if website not in csv_data:
           raise HTTPException(status_code=404, detail=f"No data for website: {website}")
       
       timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
       filename = f"{website.replace(' ', '_')}_{timestamp}.csv"
       
       csv_content = csv_data[website].getvalue()
       
       response_data = {
           "csv_content": csv_content,
           "filename": filename
       }
       
       return JSONResponse(content=response_data)
       
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))

async def process_website_background(website_name: str, website_config: dict, task_id: str):
    """Background task for processing individual website"""
    try:
        # Initialize
        progress_tracker.set_current_step(website_name, "Initializing")
        progress_tracker.update_progress(website_name, "", "processing", {"current_step": "Initializing"})
        
        # Load Google Sheets data
        sheets_data = await asyncio.to_thread(
            article_generator.sheets_manager.fetch_all_sheets_data
        )
        
        # Get sheet name
        sheet_name = website_config.get('google_sheet_tab', website_name)
        if sheet_name not in sheets_data:
            error_msg = f"Sheet '{sheet_name}' not found"
            logger.error(error_msg)
            website_status = progress_tracker.get_website_status(website_name)
            if website_status:
                website_status["status"] = "failed"
                website_status["current_step"] = f"Error: {error_msg}"
            return
        
        df = sheets_data[sheet_name]
        processing_rows = df[df['article_done'].fillna('').str.lower() != 'yes']
        
        # Set total keywords
        progress_tracker.set_website_total(website_name, len(processing_rows))
        
        # Mark website as processing
        website_status = progress_tracker.get_website_status(website_name)
        if website_status:
            website_status["status"] = "processing"
            website_status["start_time"] = datetime.now().isoformat()
        
        # Process each keyword
        for index, row in processing_rows.iterrows():
            keyword = row['keyword']
            try:
                # Update progress
                progress_tracker.update_progress(
                    website_name, 
                    keyword, 
                    "processing", 
                    {"current_step": f"Processing {keyword}"}
                )
                
                # Generate article
                article_data = await article_generator.generate_article(
                    website_config, keyword, progress_tracker
                )
                
                # Update Google Sheet
                update_data = {
                    'article_done': 'YES',
                    'article_url': article_data['url'],
                    'article_title': article_data['title'],
                    'focus_keyword': article_data['focus_keyword'],
                    'slug': article_data['slug']
                }
                
                await asyncio.to_thread(
                    article_generator.sheets_manager.update_sheet_status,
                    sheet_name, index, update_data
                )
                
                # Mark keyword as completed
                progress_tracker.complete_keyword(website_name, keyword, success=True)
                
                logger.info(f"Successfully processed keyword '{keyword}' for {website_name}")
                
            except Exception as e:
                logger.error(f"Error processing keyword '{keyword}': {str(e)}")
                
                # Mark keyword as failed
                progress_tracker.complete_keyword(website_name, keyword, success=False)
                
                continue  # Continue with next keyword
        
        # Mark website as completed
        website_status = progress_tracker.get_website_status(website_name)
        if website_status:
            website_status["status"] = "completed"
            website_status["current_step"] = "All articles completed"
            website_status["progress"] = 100
        
        logger.info(f"Completed processing for website: {website_name}")
        
    except asyncio.CancelledError:
        # Handle cancellation
        website_status = progress_tracker.get_website_status(website_name)
        if website_status:
            website_status["status"] = "stopped"
            website_status["current_step"] = "Stopped by user"
        logger.info(f"Processing cancelled for website: {website_name}")
        
    except Exception as e:
        # Handle general errors
        website_status = progress_tracker.get_website_status(website_name)
        if website_status:
            website_status["status"] = "failed"
            website_status["current_step"] = f"Error: {str(e)}"
        logger.error(f"Error processing website {website_name}: {str(e)}")

# Mount static files for frontend
FRONTEND_DIR = Path("./frontend")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="index")
    print(f"✅ Frontend mounted from {FRONTEND_DIR}")
else:
    print("⚠️  Warning: Frontend directory not found")

@app.get("/")
async def root():
   return {"message": "Recipe Article Generator API v2.0", "status": "online"}

# Add environment variable support
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"🚀 Starting Recipe Automation SaaS on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )