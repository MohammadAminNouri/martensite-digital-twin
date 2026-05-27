# Open data ingestion plan

The app does not bundle large third-party datasets. Instead, it records sources and provides the structure for later ingestion.

## Priority sources

1. Zenodo: In-situ Heating-Stage EBSD Validation of Algorithms for Prior-Austenite Grain Reconstruction in Steel.
2. Zenodo: Prior Austenite Grain Measurement supplementary material.
3. MTEX martensite parent grain reconstruction example.
4. orix-data orientation mapping datasets.
5. Cayron NiTi B2→B19′ open-access paper as crystallographic reference.

## Why not bundle everything?

- Raw EBSD and image datasets can be large.
- Every dataset has its own license and citation requirements.
- Some articles are open-access but do not include raw data.

## Near-term ingestion targets

- Add a `data/raw/zenodo_8348372/` ingestion script.
- Parse CTF files to orientation CSV using orix/kikuchipy/MTEX export.
- Store provenance in `data/metadata/samples.csv`.
- Add benchmark notebooks comparing reconstructed parent grains to measured high-temperature parent maps.
