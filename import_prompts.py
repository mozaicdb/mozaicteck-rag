import json
import os
import pymongo
from dotenv import load_dotenv

# 1. Load secret keys first
load_dotenv("../.env")

# 2. Get connection string
mongo_uri = os.getenv("MONGO_URI")

# 3. Connect to MongoDB
client = pymongo.MongoClient(mongo_uri)

# 4. Point to database and collection
db = client["mozaic_db"]
collection = db["prompts"]

def import_data():
    try:
        # 5. Delete all existing prompts first
        collection.delete_many({})
        print("Old prompts cleared successfully.")

        # 6. Open the JSON file
        with open("prompts.json", "r") as file:
            data = json.load(file)

        all_prompts = []

        # 7. Loop through categories and attach label to each prompt
        for category in data["categories"]:
            for prompt in category["prompts"]:
                prompt["category_id"] = category["id"]
                prompt["category_label"] = category["label"]
                all_prompts.append(prompt)

        # 8. Insert all prompts with category labels
        if all_prompts:
            result = collection.insert_many(all_prompts)
            print(f"Oshey! Successfully inserted {len(result.inserted_ids)} prompts.")
        else:
            print("No prompts found.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import_data()