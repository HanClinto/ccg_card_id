# Pull down the latest version of Scryfall Bulk Data
# If it is newer than our most recent local cache, then download the new version

# GET https://api.scryfall.com/bulk-data

"""
HTTP 200 success
Content-Type: application/json; charset=utf-8

{
  "object": "list",
  "has_more": false,
  "data": [
    {
      "object": "bulk_data",
      "id": "27bf3214-1271-490b-bdfe-c0be6c23d02e",
      "type": "oracle_cards",
      "updated_at": "2025-11-19T22:05:15.945+00:00",
      "uri": "https://api.scryfall.com/bulk-data/27bf3214-1271-490b-bdfe-c0be6c23d02e",
      "name": "Oracle Cards",
      "description": "A JSON file containing one Scryfall card object for each Oracle ID on Scryfall. The chosen sets for the cards are an attempt to return the most up-to-date recognizable version of the card.",
      "size": 166326785,
      "download_uri": "https://data.scryfall.io/oracle-cards/oracle-cards-20251119220515.json",
      "content_type": "application/json",
      "content_encoding": "gzip"
    },
    {
      "object": "bulk_data",
      "id": "6bbcf976-6369-4401-88fc-3a9e4984c305",
      "type": "unique_artwork",
      "updated_at": "2025-11-19T10:06:09.185+00:00",
      "uri": "https://api.scryfall.com/bulk-data/6bbcf976-6369-4401-88fc-3a9e4984c305",
      "name": "Unique Artwork",
      "description": "A JSON file of Scryfall card objects that together contain all unique artworks. The chosen cards promote the best image scans.",
      "size": 240490331,
      "download_uri": "https://data.scryfall.io/unique-artwork/unique-artwork-20251119100609.json",
      "content_type": "application/json",
      "content_encoding": "gzip"
    },
    {
      "object": "bulk_data",
      "id": "e2ef41e3-5778-4bc2-af3f-78eca4dd9c23",
      "type": "default_cards",
      "updated_at": "2025-11-19T22:20:54.097+00:00",
      "uri": "https://api.scryfall.com/bulk-data/e2ef41e3-5778-4bc2-af3f-78eca4dd9c23",
      "name": "Default Cards",
      "description": "A JSON file containing every card object on Scryfall in English or the printed language if the card is only available in one language.",
      "size": 519642590,
      "download_uri": "https://data.scryfall.io/default-cards/default-cards-20251119222054.json",
      "content_type": "application/json",
      "content_encoding": "gzip"
    },
    {
      "object": "bulk_data",
      "id": "922288cb-4bef-45e1-bb30-0c2bd3d3534f",
      "type": "all_cards",
      "updated_at": "2025-11-19T22:50:10.429+00:00",
      "uri": "https://api.scryfall.com/bulk-data/922288cb-4bef-45e1-bb30-0c2bd3d3534f",
      "name": "All Cards",
      "description": "A JSON file containing every card object on Scryfall in every language.",
      "size": 2440567891,
      "download_uri": "https://data.scryfall.io/all-cards/all-cards-20251119225010.json",
      "content_type": "application/json",
      "content_encoding": "gzip"
    },
    {
      "object": "bulk_data",
      "id": "06f54c0b-ab9c-452d-b35a-8297db5eb940",
      "type": "rulings",
      "updated_at": "2025-11-19T22:00:53.196+00:00",
      "uri": "https://api.scryfall.com/bulk-data/06f54c0b-ab9c-452d-b35a-8297db5eb940",
      "name": "Rulings",
      "description": "A JSON file containing all Rulings on Scryfall. Each ruling refers to cards via an `oracle_id`.",
      "size": 24329467,
      "download_uri": "https://data.scryfall.io/rulings/rulings-20251119220053.json",
      "content_type": "application/json",
      "content_encoding": "gzip"
    }
  ]
}
"""

# Local cache path is: ./cache/all_cards.json 
# If updated_at date is newer than the modified date on our local cache file OR local cache file does not exist, then download the newest version of the file and return TRUE

import os
import requests

bulkdata_type = "all_cards"

import json
from datetime import datetime
import datetime as dt

def sync_scryfall_bulkdata():
  """
  Checks Scryfall bulk-data endpoint for the latest 'all_cards' file.
  Downloads it if newer than local cache or if cache is missing.
  Returns True if a new file was downloaded, False otherwise.
  """
  API_URL = "https://api.scryfall.com/bulk-data"
  CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
  CACHE_PATH = os.path.join(CACHE_DIR, f"{bulkdata_type}.json")

  # Get bulk data metadata from Scryfall
  resp = requests.get(API_URL)
  resp.raise_for_status()
  data = resp.json()
  bulk_entry = None
  for entry in data.get("data", []):
    if entry.get("type") == bulkdata_type:
      bulk_entry = entry
      break
  if not bulk_entry:
    raise RuntimeError(f"Could not find bulk data type '{bulkdata_type}' in Scryfall response.")

  remote_updated_at = bulk_entry["updated_at"]
  remote_updated_dt = datetime.fromisoformat(remote_updated_at.replace("Z", "+00:00"))
  download_url = bulk_entry["download_uri"]

  print(f"Remote data updated at: {remote_updated_dt.isoformat()}")
  
  # Check local cache
  need_download = False
  if not os.path.exists(CACHE_PATH):
    print(f"Local cache not found: {CACHE_PATH}")
    need_download = True
  else:
    # Make local_mtime timezone-aware (UTC) using recommended method
    local_mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_PATH), tz=dt.UTC)
    print(f"Local cache updated at: {local_mtime.isoformat()}")
    # Compare remote updated_at to local file mtime
    if remote_updated_dt > local_mtime:
      need_download = True

  if need_download:
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Downloading new {bulkdata_type} data from Scryfall...")
    with requests.get(download_url, stream=True) as r:
      r.raise_for_status()
      with open(CACHE_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
          f.write(chunk)
    # Update mtime to match remote updated_at
    mtime = remote_updated_dt.timestamp()
    os.utime(CACHE_PATH, (mtime, mtime))
    return True
  else:
    print(f"Local cache is up to date: {CACHE_PATH}")
    return False

if __name__ == "__main__":
  changed = sync_scryfall_bulkdata()
  print(f"Data updated: {changed}")


