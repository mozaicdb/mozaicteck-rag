import json
import os
import pymongo
from dotenv import load_dotenv

# 1. Step out of the room to find the secret keys FIRST
load_dotenv("../.env")

# 2. Print to confirm it loaded correctly
print(f"Testing URI: {os.getenv('MONGO_URI')}")

# 3. Get the secret connection string
mongo_uri = os.getenv("MONGO_URI")

# 4. Set up the client to talk to the cloud
client = pymongo.MongoClient(mongo_uri)

# 5. Point to your specific database and shelf
db = client["mozaic_db"]
collection = db["prompts"]

def import_data():
    try:
        # 6. Open the JSON file in 'read' mode
        with open("prompts.json", "r") as file:
            data = json.load(file)
            
        all_prompts = []
        
        # 7. Look inside each category bag and grab the "prompts" list
        for category in data["categories"]:
            all_prompts.extend(category["prompts"])
            
        # 8. Push the whole collection to MongoDB
        if all_prompts:
            result = collection.insert_many(all_prompts)
            print(f"Oshey! Successfully inserted {len(result.inserted_ids)} prompts into MongoDB.")
        else:
            print("The bag was empty, boss.")
            
    except Exception as e:
        print(f"Wait, an error occurred: {e}")

if __name__ == "__main__":
    import_data()