# CCG Card ID
Computer vision dataset and library for visual recognition of collectible cards (such as Magic, Pokemon, Yugioh, and more)

Represents the entire pipeline, from data gathering and cleaning, to card detection (finding the location of a card in an image), to card vectorization, and finally, to card lookup.

Data is cached in each step, and scripts are intended to be run once, in sequence, and then re-run later to update with the latest data available (such as when new sets are printed, or when new real-life camera images are available and added).

Each script should check the dates of each piece of data downloaded or calculated, and only redownload / recalculate if our cached data is missing or outdated.

In this way, we can run the full pipeline of scripts on a nightly basis, and only have to spend time on a minimum of recalculation and download.
