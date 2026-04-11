TCGplayer exploration notes

Current usable public endpoints:
- `https://mpapi.tcgplayer.com/v2/Catalog/CatalogGroups`
- `https://mpapi.tcgplayer.com/v2/Catalog/CategoryFilters`
- `https://mp-search-api.tcgplayer.com/v1/search/productLines`
- `https://mp-search-api.tcgplayer.com/v1/search/request`
- `https://mp-search-api.tcgplayer.com/v1/product/{productId}/listings`  ← **key endpoint for historical listos**
- `https://mp-search-api.tcgplayer.com/v2/product/{productId}/details`
- `https://infinite-api.tcgplayer.com/priceguide/set/{setId}/cards/?rows=5000&productTypeID=1`
- `https://data.tcgplayer.com/autocomplete`

Current findings from live probes on March 26 and April 11, 2026:
- `/v1/search/request` is the most useful catalog endpoint. It supports game-level and set-level crawling with pagination via `from` and `size`.
- In this environment, `size=50` worked reliably for `/v1/search/request`; `size>=100` returned `400 Bad Request`.
- The search response contains a small sample of live listings for each product, including `listingType: "custom"` entries with `customData.images`.
- **`/v1/product/{productId}/listings` returns 200 without any authentication** — an earlier note that it required auth was incorrect (the 403 was environment-specific). This is a POST endpoint with the same JSON body structure as the listing filter in the search request.
- **`quantity.gte: 0` unlocks historical/sold-out listings.** The default `quantity.gte: 1` filter shows only currently in-stock listings. Setting `gte: 0` exposes zero-quantity records that remain in the index after selling out. For product 1388 (Black Lotus Alpha), this increased results from 99 → 4,946 total (4,847 zero-quantity). Custom photo listos: 5 live → 259 total (254 zero-quantity), with listing IDs as low as ~200k suggesting genuinely old records.
- Page size limit for `/v1/product/{productId}/listings` is also **50**; `size>=100` returns `400 Bad Request`. Paginate via `from`.
- The CDN URL pattern for custom listing images (UUID-keyed) is not yet resolved. `tcgplayer-cdn.tcgplayer.com/custom-listing-images/{uuid}.jpg` returns 403 — note that this CDN returns 403 for both access-denied AND not-found, so this is ambiguous. The image UUID IDs from the API are valid references; image URL resolution is a TODO.
- The `listingType: "custom"` term filter works to isolate photo listings specifically.

Quick start:

```bash
python3 01_data_sources/tcgplayer/01_sync_data.py --max-games 3 --max-sets 1
python3 01_data_sources/tcgplayer/01_sync_data.py --exclude "Magic"
python3 01_data_sources/tcgplayer/01_sync_data_fast.py --product-line "Magic: The Gathering" --set-name "Teenage Mutant Ninja Turtles"
python3 01_data_sources/tcgplayer/01_sync_data.py --product-line Pokemon --max-sets 1
python3 01_data_sources/tcgplayer/01_sync_data.py --product-line Pokemon --set-name "SV: Scarlet & Violet 151"
python3 01_data_sources/tcgplayer/02_sync_images.py
python3 01_data_sources/tcgplayer/02_sync_images.py --product-lines "Pokemon"
python3 01_data_sources/tcgplayer/03_sync_product_details.py
python3 01_data_sources/tcgplayer/04_build_catalog_db.py --fast-src /path/to/popular_games_fast.json
python3 01_data_sources/tcgplayer/04_build_catalog_db.py --catalog-src /path/to/popular_games.json --details-dir /path/to/product_details --images-dir /path/to/images
python3 01_data_sources/tcgplayer/01_sync_data.py --details-product-id 502558 --skip-catalog
```

Default behavior:
- The script now defaults to `--popular-games`, using TCGplayer's curated navigation JSON from `marketplace-navigation-search-feature.json`.
- Passing `--product-line "..."` overrides that default and crawls only the specified line.
- `01_sync_data.py` now also supports `--exclude "Magic"` style filters so you can skip deeper slow crawls for selected games.
- `01_sync_data_fast.py` uses one lightweight search probe per set to resolve `setId`, then pulls the bulk product/price rows from the faster price-guide API.
- `01_sync_data_fast.py` now writes aggregate popular-games output as:
  - a lightweight manifest at `catalogs/popular_games_fast.json`
  - game-level source manifests under `catalogs/games/<game>/priceguide.json`
  - set-level source payloads under `catalogs/games/<game>/sets/<set>/priceguide.json`
- `01_sync_data.py` writes its aggregate slow catalog the same way:
  - a lightweight manifest at `catalogs/popular_games.json`
  - game-level source manifests under `catalogs/games/<game>/mp_search.json`
  - set-level source payloads under `catalogs/games/<game>/sets/<set>/mp_search.json`
- JSON writes use crash-safe temp-file + backup replacement, so partial writes should no longer destroy the previous file contents.

Reference images:
- `02_sync_images.py` downloads high-resolution catalog/reference images using:
  `https://tcgplayer-cdn.tcgplayer.com/product/{productId}_in_1000x1000.jpg`
- Images are written under `catalog/tcgplayer/images/product/`.
- This is only for catalog/reference art. Listings-with-photos (`listos`) should be treated as dataset material later under `02_data_sets/`.

Listos archive:
- `01_sync_data.py` now archives sampled custom listings from the slow search crawl under:
  `datasets/tcgplayer_listos/data/listings/<digit>/<digit>/<customListingId>.json`
- These files are retained even after the listings disappear from current search results.
- Each file tracks `firstSeenAt`, `lastSeenAt`, the latest sampled payload, and the observed product/set sightings.

Historical listo collection via `/v1/product/{productId}/listings`:
- This endpoint can be queried per-product with `quantity.gte: 0` to retrieve ALL custom listings — including sold-out/historical ones — not just the small sample embedded in search results.
- Workflow: iterate all known productIds from the catalog DB, POST to `/v1/product/{id}/listings` with `listingType: "custom"` and `quantity.gte: 0`, paginate with `from`/`size` (max size=50).
- No authentication required. Rate-limit politely.
- Image UUID resolution (CDN URL pattern) is still a TODO — collect the UUIDs now and resolve the download URL later once the pattern is found.
- Example request body:
  ```json
  {
    "filters": {
      "term": {"sellerStatus": "Live", "channelId": 0, "listingType": "custom"},
      "range": {"quantity": {"gte": 0}},
      "exclude": {"channelExclusion": 0}
    },
    "context": {"shippingCountry": "US", "cart": {}},
    "from": 0,
    "size": 50
  }
  ```

SKU enrichment + SQLite:
- `03_sync_product_details.py` downloads per-product details payloads, including the `skus` table.
- `04_build_catalog_db.py` is incremental now:
  - fast sync updates base catalog + `product_priceguide_rows`
  - slow sync updates `product_metadata` + `sample_listings`
  - details sync updates `product_skus`
  - image sync updates `product_images`
