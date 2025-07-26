<<<<<<< HEAD
# blog_automation
this project automates the blog articles creation on Wordpress (Recipes and food blogging niche)
=======
# Blog Automation - Recipe Article Generator

A FastAPI-based backend for automated recipe article generation using Claude AI, with image generation via Midjourney and WordPress publishing.

## 🚀 Quick Setup

### 1. Environment Setup

**First, copy the environment template:**
```bash
cp .env.example .env
```

**Then edit `.env` and add your API keys:**
```bash
# Required API Keys
CLAUDE_API_KEY=sk-ant-api03-your-actual-claude-key-here
SERPAPI_API_KEY=your-serpapi-key-here
DISCORD_TOKEN=your-discord-bot-token-here
RECRAFT_API_KEY=your-recraft-api-key-here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python main.py
```

The application will be available at `http://localhost:8000`

## 🔐 Security & Secrets Management

This project uses **environment variables** for all sensitive information like API keys. Here's what's been set up:

### ✅ What's Protected:
- **Claude API Key** → `CLAUDE_API_KEY`
- **SERP API Key** → `SERPAPI_API_KEY`  
- **Discord Bot Token** → `DISCORD_TOKEN`
- **Recraft API Key** → `RECRAFT_API_KEY`

### ✅ What's Safe in Git:
- Configuration structure (without secrets)
- Application code
- Templates and schemas

### ❌ What's NEVER committed:
- `.env` file (contains your actual keys)
- `config.json` (now loads from environment variables)
- Any files with actual API keys

## 🐳 Docker Deployment

### Build and Run with Docker Compose

```bash
# Build and start services
docker compose up --build

# Stop services
docker compose down
```

### Key Docker Features:
- **Persistent Data:** Logs, uploads, and cache are persisted between restarts
- **Health Checks:** Built-in health monitoring at `/health`
- **Security:** Runs as non-root user
- **Port:** Accessible at `http://localhost:8000`

## 🛠 Configuration

### Required Files:
1. **`.env`** - Your API keys and environment variables
2. **`lbc-automation-process.json`** - Google Sheets credentials
3. **`config.json`** - Application configuration (auto-loads from env vars)

### Environment Variables:
```bash
# Application
ENVIRONMENT=production
API_HOST=0.0.0.0
API_PORT=8000

# API Keys (REQUIRED)
CLAUDE_API_KEY=your-claude-key
SERPAPI_API_KEY=your-serpapi-key
DISCORD_TOKEN=your-discord-token
RECRAFT_API_KEY=your-recraft-key

# Security
SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_HOURS=24
```

## 📊 Features

- **🤖 AI Content Generation** - Claude AI for article writing
- **🎨 Image Generation** - Midjourney integration via Discord
- **📝 WordPress Publishing** - Automated post creation
- **📈 SEO Optimization** - Built-in SEO rules and validation
- **📊 Google Sheets Integration** - Keyword management
- **🔍 SERP Analysis** - Competitor research
- **📱 Web Interface** - FastAPI frontend
- **📈 Progress Tracking** - Real-time generation status

## 🔧 Development

### Local Development:
1. Copy `.env.example` to `.env`
2. Add your API keys to `.env`
3. Run `python main.py`

### Production Deployment:
1. Set environment variables in your hosting platform
2. Deploy using Docker or direct Python
3. Ensure `.env` file is not deployed (use platform env vars)

## 🆘 Troubleshooting

### Common Issues:

**❌ "Missing required config keys"**
- Make sure your `.env` file has all required API keys
- Check that `load_dotenv()` is working

**❌ Git push rejected due to secrets**
- Remove any API keys from `config.json`
- Use environment variables instead
- Check that `.env` is in `.gitignore`

**❌ Application not starting**
- Verify all environment variables are set
- Check `lbc-automation-process.json` exists
- Ensure all dependencies are installed

## 📚 API Documentation

Once running, visit:
- **API Docs:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health`
- **Web Interface:** `http://localhost:8000`

## 🔐 Security Best Practices

1. **Never commit `.env` files**
2. **Use different keys for development/production**
3. **Rotate API keys regularly**
4. **Use environment variables in production**
5. **Keep `lbc-automation-process.json` private**

---

**For production deployment on Railway/Heroku/etc:**
Set the environment variables in your platform's settings instead of using a `.env` file.
>>>>>>> 868f480 (Initial commit - Blog Automation with config)
