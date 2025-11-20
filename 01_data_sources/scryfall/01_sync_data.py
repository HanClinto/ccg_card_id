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


