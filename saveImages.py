import asyncio
import aiohttp
import pandas as pd
import os
import random

# --- Config ---
csv_files = [
            #  "ltd-flooring-surfaces-materials-image-csv/materials.csv",
            #  "ltd-flooring-surfaces-materials-image-csv/surface_images.csv",
             "ltd-flooring-surfaces-materials-image-csv/lighting.csv",
             ]
output_dir = "images-ltd-1"
os.makedirs(output_dir, exist_ok=True)
num_samples = 1500


semaphore = asyncio.Semaphore(20)

# --- Function to download one image ---
async def download_image(session, url, save_path):
    async with semaphore:
        try:
            await asyncio.sleep(random.uniform(0.5, 2))  # random delay
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(save_path, "wb") as f:
                        f.write(content)
                else:
                    print(f"Failed {url}: {response.status}")
        except Exception as e:
            print(f"Error {url}: {e}")

# --- Download images from a list ---
async def download_images(image_list, folder_name):
    folder_path = os.path.join(output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, row in enumerate(image_list):
            name = row[0].replace(" ", "_")  # clean filename
            url = row[1]
            save_path = os.path.join(folder_path, f"{i}_{name}.jpg")
            tasks.append(download_image(session, url, save_path))
        await asyncio.gather(*tasks)

# --- Main ---
async def main():
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df_sampled = df.sample(n=min(num_samples, len(df)), random_state=42)
        images = df_sampled.values.tolist()
        folder_name = os.path.splitext(os.path.basename(csv_file))[0]
        print(f"Downloading {len(images)} images from {csv_file}...")
        await download_images(images, folder_name)

# Run the async event loop
asyncio.run(main())
