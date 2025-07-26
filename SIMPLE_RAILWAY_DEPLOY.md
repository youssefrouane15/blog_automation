# 🚀 Simple Railway Deployment - Using config.json

## Quick Deploy Steps

### 1. Create GitHub Repository
- Go to github.com
- Create new PRIVATE repository named "blog-automation"
- Keep it PRIVATE since config.json has your API keys

### 2. Upload Your Code
```bash
cd C:\Users\rouan\OneDrive\Desktop\WORKSPACE\Automations\Blog_Automation

git init
git add .
git commit -m "Initial commit with config"
git remote add origin https://github.com/YOUR_USERNAME/blog-automation.git
git push -u origin main
```

### 3. Deploy to Railway
1. Go to railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your PRIVATE "blog-automation" repository
5. Railway will automatically deploy!

### 4. Upload Google Credentials
Since Railway needs the Google credentials file:

**Option A: Include in Git (Easiest)**
- Rename `lbc-automation-process.json` to `google-creds.json`
- Update config.json to use the new name
- Commit and push

**Option B: Use Railway Variables (More Secure)**
- In Railway dashboard → Variables
- Add: `GOOGLE_CREDENTIALS_BASE64` with your base64 encoded JSON

### That's it! 🎉

## What Railway Will Do:
- ✅ Read your config.json automatically
- ✅ Install requirements.txt
- ✅ Start your app using Procfile
- ✅ Give you a public URL

## Important Notes:
- Keep your GitHub repo PRIVATE
- Your config.json will be used as-is
- No environment variables needed!
- Railway will handle everything else

## To Update Your App:
```bash
git add .
git commit -m "Update"
git push
```

Railway auto-deploys on every push!