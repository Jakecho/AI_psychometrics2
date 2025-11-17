# Deploying NCLEX Item Generator to Streamlit Community Cloud

## Prerequisites

1. **GitHub Account**: Your repository is already set up at https://github.com/Jakecho/nclex-item-generator
2. **Streamlit Account**: Sign up at https://streamlit.io/cloud (use your GitHub account)
3. **OpenAI API Key**: Have your OpenAI API key ready

## Deployment Steps

### Step 1: Push Latest Changes to GitHub

```bash
# Make sure all changes are committed and pushed
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

✅ **Already completed** - Your repository is up to date!

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: https://share.streamlit.io/

2. **Sign in with GitHub**

3. **Click "New app"**

4. **Fill in the deployment form**:
   - **Repository**: `Jakecho/nclex-item-generator`
   - **Branch**: `main`
   - **Main file path**: `AIG_NCLEX.py`
   - **App URL** (optional): Choose a custom URL like `nclex-item-generator`

5. **Advanced settings** (click "Advanced settings"):
   - **Python version**: 3.11 or 3.12
   - **Secrets**: Add your OpenAI API key (see below)

### Step 3: Configure Secrets

In the **Secrets** section, add:

```toml
OPENAI_API_KEY = "sk-proj-YOUR-API-KEY-HERE"
```

Replace `sk-proj-YOUR-API-KEY-HERE` with your actual OpenAI API key.

### Step 4: Deploy!

Click **Deploy** and wait 2-5 minutes for the app to build and launch.

## Deploy the Second App (Vector Search)

After deploying `AIG_NCLEX.py`, you can deploy the vector search system:

1. **Click "New app"** again

2. **Fill in the deployment form**:
   - **Repository**: `Jakecho/nclex-item-generator`
   - **Branch**: `main`
   - **Main file path**: `NCLEX_RN_VectorDB.py`
   - **App URL**: Choose a different URL like `nclex-vector-search`

3. **Advanced settings**:
   - Add PostgreSQL database connection if needed (see below)

## PostgreSQL Database Configuration (for Vector Search)

⚠️ **Important**: The vector search app requires a PostgreSQL database with pgvector extension.

### Option 1: Use Streamlit's PostgreSQL (Recommended for testing)

Streamlit doesn't provide built-in PostgreSQL, so you'll need an external database.

### Option 2: Use Neon (Free PostgreSQL with pgvector)

1. Sign up at https://neon.tech
2. Create a new project
3. Enable pgvector extension:
   ```sql
   CREATE EXTENSION vector;
   ```
4. Add connection string to Streamlit secrets:

```toml
[postgresql]
host = "your-project.neon.tech"
database = "neondb"
user = "your-username"
password = "your-password"
port = "5432"
```

5. Update `NCLEX_RN_VectorDB.py` to read from secrets:

```python
# Replace hardcoded connection with:
conn = psycopg2.connect(
    host=st.secrets["postgresql"]["host"],
    database=st.secrets["postgresql"]["database"],
    user=st.secrets["postgresql"]["user"],
    password=st.secrets["postgresql"]["password"],
    port=st.secrets["postgresql"]["port"]
)
```

### Option 3: Use Supabase (Free PostgreSQL with pgvector)

1. Sign up at https://supabase.com
2. Create a new project
3. Get connection string from Settings > Database
4. Add to Streamlit secrets (same format as above)

## Post-Deployment

### Your Apps Will Be Available At:

- **AI Item Generator**: `https://your-app-name.streamlit.app`
- **Vector Search**: `https://your-search-app.streamlit.app`

### Managing Your Apps

1. **View logs**: Click on your app in the dashboard
2. **Restart app**: Use the "Reboot" button
3. **Update app**: Just push to GitHub - auto-deploys!
4. **Manage secrets**: Edit in app settings

## Troubleshooting

### Common Issues

**1. Build fails with "No module named X"**
- Solution: Add the module to `requirements.txt`

**2. App crashes on startup**
- Check logs in Streamlit Cloud dashboard
- Verify secrets are configured correctly

**3. Database connection fails**
- Verify PostgreSQL connection string in secrets
- Check if pgvector extension is installed
- Ensure database allows external connections

**4. Out of memory errors**
- sentence-transformers models can be large
- Consider using a smaller model or hosting database separately

### Support

- **Streamlit Docs**: https://docs.streamlit.io/streamlit-community-cloud
- **Streamlit Forum**: https://discuss.streamlit.io/

## Notes for Production

- **Free tier limitations**: 1GB RAM, 1 CPU core per app
- **Auto-sleep**: Apps sleep after inactivity (wakes on visit)
- **Private apps**: Upgrade to Teams plan for authentication
- **Custom domain**: Available on Teams plan

## Alternative: Deploy AI Generator Only

If you want to deploy just the AI Item Generator (without vector search):

1. Deploy only `AIG_NCLEX.py` (no database needed!)
2. Users can generate items using OpenAI
3. No PostgreSQL configuration required
4. Much simpler deployment

This is the recommended approach for ICE Exchange 2025 presentation!

---

**Quick Start**: Just deploy `AIG_NCLEX.py` with your OpenAI API key in secrets - you're done! 🚀
