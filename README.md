# Kairos

Python repository for end-to-end listed-option volatility analysis:

- ingest daily prices and option-chain snapshots
- clean and validate the data
- compute implied vols and Greeks
- fit volatility smiles and a basic surface
- compare implied and realized volatility

## Ingestion Paths

Daily prices can be fetched from the IBKR Web API and from Twelve Data.