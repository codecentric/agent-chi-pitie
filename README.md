# Showcase application für Agenten in LangChain (TS)

Diese Anwendung zeigt am Beispiel eines Geheimagenten wie einfache Agentensysteme mit LangChain und TypeScript umgesetzt werden können.
Sie ist als Begleitprojekt zu einem Blogartikel der Serie "GenAI für Full Stack EntwicklerInnen" entstanden.

## Setup

Um die Anwendung zu starten wird ein Azure OpenAI Deployment benötigt. Erstelle ein file `azure.env` mit folgendem Inhalt:

```bash
export AZURE_OPENAI_API_INSTANCE_NAME=<AZURE_REGION>
export AZURE_OPENAI_API_DEPLOYMENT_NAME=<AZURE_DEPLOYMENT>
export AZURE_OPENAI_API_KEY=<AZURE_AUTH_KEY> # One of either Key1 / Key2 from your azure openAI instance
export AZURE_OPENAI_API_VERSION="2024-02-01"
export AZURE_OPENAI_BASE_PATH=https://<AZURE_DOMAIN>/openai/deployments
export TAVILY_API_KEY=<TAVILY-KEY>

```

Dann kannst du die Anwendung starten:

```shell
yarn install
yarn dev
```

Auf http://localhost:3333 kann die Anwendung aufgerufen werden. Der Agent fragt nach einer Rolle die er einnehmen soll. Sobald er eine erhält versucht er mithilfe von Wikipedia und Tavily die Rolle einzunehmen.

## Genutzte Frameworks

- Azure OpenAI
- Backroad
- LangChain (JS)
- zod
