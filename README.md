# Showcase application für function calling in TypeScript

Diese Anwendung zeigt am Beispiel einer einfachen Arbeitszeitberechnung wie Function Calling funtioniert.
Sie ist als Begleitprojekt zu einem Blogartikel der Serie "GenAI für Full Stack EntwicklerInnen" entstanden.

## Setup

Um die Anwendung zu starten wird ein Azure OpenAI deployment benötigt. Erstelle ein file `azure.env` mit folgendem Inhalt:

```bash
export AZURE_OPENAI_API_INSTANCE_NAME=<AZURE_REGION>
export AZURE_OPENAI_API_DEPLOYMENT_NAME=<AZURE_DEPLOYMENT>
export AZURE_OPENAI_API_KEY=<AZURE_AUTH_KEY> # One of either Key1 / Key2 from your azure openAI instance
export AZURE_OPENAI_API_VERSION="2024-02-01"
export AZURE_OPENAI_BASE_PATH=https://<AZURE_DOMAIN>/openai/deployments
```

Dann kannst du die Anwendung starten:

```shell
yarn install
yarn dev
```

Auf http://localhost:3333 kann die Anwendung aufgerufen werden. Bei Beschreibungen eines Arbeitstages antwortet sie mit der berechneten Arbeitszeit.
Dabei verlässt sie sich nicht auf die Berechnungen des Modells, sondern auf eine eigene Berechnungsmethode.

## Genutzte Frameworks

- Azure OpenAI
- Backroad
- LangChain (JS)
- zod
