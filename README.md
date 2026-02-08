# MercadoLibre_project
```markdown
# Setup
pip install -r requirements.txt

# Configure API key
# Here is my way: I created env in juypterlab directory using the following code.
# 1. check dir
current_dir = os.getcwd()
print(f"dir: {current_dir}")

# 3. create .env 
try:
    env_path = os.path.join(current_dir, '.env')
    with open(env_path, 'w') as f:
        f.write('OPENAI_API_KEY=sk-e9588fb1d1914dd6bab3e6093f4940fc\n')
    print(f"✅ successfully created .env: {env_path}")
except PermissionError as e:
    print(f"❌ error: {e}")


# Run notebooks
jupyter notebook

# Run dashboard
import function from dashboard.py
```
