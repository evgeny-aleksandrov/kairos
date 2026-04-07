# Kairos

Python repository for end-to-end listed-option volatility analysis:

- ingest daily prices and option-chain snapshots
- clean and validate the data
- compute implied vols and Greeks
- fit volatility smiles and a basic surface
- compare implied and realized volatility

## Ingestion Paths

Daily prices and option-chain snapshots can be fetched from the IBKR Web API.

## Starting The IBKR Gateway

Before using the IBKR Web API scripts, start the Client Portal Gateway locally and complete the browser login flow.

From the extracted `clientportal.gw` directory on Windows:

```powershell
cd C:\path\to\clientportal.gw
.\bin\run.bat .\root\conf.yaml
```

Then authenticate in the browser at:

```text
https://localhost:5000
```
