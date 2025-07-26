#!/usr/bin/env python3
"""
Script to add Railway port configuration to the end of main.py
"""

import os

# The content to add at the end of main.py
railway_port_config = '''

# Add this at the very end of main.py for Railway deployment
if __name__ == "__main__":
    import os
    
    # Railway provides PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    
    # Run the app
    uvicorn.run("main:app", host="0.0.0.0", port=port)
'''

# Read the current main.py file
with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check if the port configuration is already there
if 'if __name__ == "__main__":' in content and 'uvicorn.run("main:app"' in content:
    print("Port configuration already exists in main.py")
else:
    # Add the configuration
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
        f.write(railway_port_config)
    print("Successfully added Railway port configuration to main.py")
