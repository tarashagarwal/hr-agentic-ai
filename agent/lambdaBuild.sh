# 1. Create a folder for packages
mkdir python

# 2. Install into that folder
pip install -r requirements.txt -t python/

# 3. Zip it all up
cd python
zip -r ../function.zip .
cd ..
cp function1.py lambda_handler.py
zip -g function.zip lambda_handler.py
