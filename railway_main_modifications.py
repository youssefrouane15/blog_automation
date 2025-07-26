# Add these modifications to your main.py for Railway deployment

import os
import base64
import json
from pathlib import Path

# ========== ADD THIS SECTION AT THE TOP AFTER IMPORTS ==========

# Create required directories
os.makedirs('logs', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('cache', exist_ok=True)

# Handle Google credentials from environment variable
GOOGLE_CREDS_PATH = 'lbc-automation-process.json'
if os.getenv('GOOGLE_CREDENTIALS_BASE64') and not os.path.exists(GOOGLE_CREDS_PATH):
    try:
        credentials_base64 = os.getenv('GOOGLE_CREDENTIALS_BASE64')
        credentials_json = base64.b64decode(credentials_base64).decode('utf-8')
        
        # Write to file
        with open(GOOGLE_CREDS_PATH, 'w') as f:
            f.write(credentials_json)
        print(f"✓ Google credentials file created from environment variable")
    except Exception as e:
        print(f"❌ Error creating Google credentials file: {e}")

# Update configuration to use environment variables
def get_config():
    """Get configuration with environment variable overrides"""
    
    # Load base config
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with environment variables
    if os.getenv('OPENAI_API_KEY'):
        config['openai_api_key'] = os.getenv('OPENAI_API_KEY')
    
    if os.getenv('ANTHROPIC_API_KEY'):
        config['anthropic_api_key'] = os.getenv('ANTHROPIC_API_KEY')
    
    if os.getenv('JWT_SECRET_KEY'):
        config['secret_key'] = os.getenv('JWT_SECRET_KEY')
    
    # Override WordPress credentials from environment
    if 'wordpress_sites' in config:
        for i, site in enumerate(config['wordpress_sites']):
            site_num = i + 1
            username_key = f'WP_USERNAME_SITE{site_num}'
            password_key = f'WP_PASSWORD_SITE{site_num}'
            
            if os.getenv(username_key):
                site['username'] = os.getenv(username_key)
            if os.getenv(password_key):
                site['password'] = os.getenv(password_key)
    
    return config

# ========== MODIFY YOUR EXISTING CODE ==========

# Replace any line that loads config.json with:
config = get_config()

# ========== ADD THIS AT THE VERY END OF main.py ==========

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Railway provides this)
    port = int(os.environ.get("PORT", 8000))
    
    # Get host - 0.0.0.0 for Railway
    host = "0.0.0.0"
    
    print(f"🚀 Starting server on {host}:{port}")
    
    # Run the app
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        # Don't use reload in production
        reload=False,
        # Access logs
        access_log=True
    )