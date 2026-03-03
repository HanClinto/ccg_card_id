# Sol Ring Dataset
(c) 2026, HanClinto Games, LLC

A collection of 307 reference images for benchmarking Magic: The Gathering card identification.

## Purpose
To provide a meaningful metric for measuring and comparing the accuracy of card recognition algorithms.

## Theory
In Magic: The Gathering, the format of Commander is the most popular way to play the game.

In Commander, the single most-popular card (ranked #1 on EDHRec) is Sol Ring.

The Mike Bierik artwork for Sol Ring is the most-reprinted artwork in the entire game.

As such, this card presents a unique challenge for measuring card identification accuracy. It is both the most-played card in the most-played format, but it also the card whose printings are most easily mistaken for similar sets or printings.

Therefore, this card presents a unique challenge for card identification algorithms, and represents a meaningful standard for set identification accuracy -- not just on fringe cards, but on a very important and popular card against a wide swath of modern sets and editions.

## Technique

21 different printings of Sol Ring were acquired through TCGplayer.com -- each one from a different edition, but each one also with the iconic Mike Bierik artwork.

Video was then taken of each of these cards using a mobile phone against a plain white background, to capture dozens of reference frames for each card in various lightings, blurs, and angles.

Each video is labeled with the known Scryfall ID of the correct reference ID of each card.

Keyframes are extracted from these videos, and blur-detection is used to filter out unwanted frames.

Resulting sharp frames ("good" frames) are then organized into the uncropped data folder.

This represents the "uncropped" version of the dataset.

To create the "aligned" version of the dataset, each "good" frame is then processed through a homography detection pipeline to crop and align the card in the frame to a standard (ScryFall) reference image.

We then de-warp the card to a standard size / rotation and place it in the "aligned" folder.

This represents the "aligned" version of the dataset.

## Usage

If you are only testing card identification accuracy, you can use the "aligned" version of the dataset, which has already been cropped and aligned to a standard reference image.

If you also want to test card detection, you can use the "uncropped" version of the dataset, which contains the original frames with the card in various positions and angles.

## License

This content is available under the Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0). You are free to share and adapt this material for any purpose, even commercially, as long as you provide appropriate credit and distribute your contributions under the same license.

License note: You are *explicitly free to use this dataset for commercial purposes*, so long as you do not redistribute the original data as your own, and any redistribution must include proper attribution to the original creators. That is all that is required.

Again, the purpose is to provide a universal, common standard for measuring card identification accuracy, and to encourage the use of this dataset as a benchmark for comparing different algorithms -- across both closed-source and open-source solutions. If for any reason the above terms are not acceptable to you, please reach out and we can get you license terms that work for your needs.

Contributions are accepted. Any additions or modifications to this dataset back to the original are appreciated, but not required.