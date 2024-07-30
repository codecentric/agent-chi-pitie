import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts"
import { VectorStore, VectorStoreRetriever } from "@langchain/core/vectorstores"
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama"
import { formatDocumentsAsString } from "langchain/util/document"
import { Document } from "langchain/document"
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run"
import { DynamicStructuredTool } from "@langchain/core/tools"
import { AgentExecutor, createToolCallingAgent } from "langchain/agents"
import { z } from "zod"
import { AzureChatOpenAI } from "@langchain/openai"
import { HumanMessage } from "@langchain/core/messages"
import { TavilySearchResults } from "@langchain/community/tools/tavily_search"
import { Serialized } from "@langchain/core/load/serializable"
import { ConsoleCallbackHandler } from "@langchain/core/tracers/console"

export async function askAgent(text: string): Promise<string> {
  const model = new AzureChatOpenAI() as any
  const prompt = (await getMemoryPrompt()) as any
  const tools = [
    knowledgeTool,
    new TavilySearchResults({
      maxResults: 1,
      callbacks: [toolCallbackFor("Tavily") as any],
    }),
  ] as any
  const agent = createToolCallingAgent({
    llm: model,
    prompt,
    tools,
  })
  const executor = new AgentExecutor({
    agent,
    tools,
    callbacks: [ConsoleCallbackHandler as any],
  })
  const response = await executor.invoke({
    input: text,
    chat_history: [new HumanMessage("Du bist Marie Curie")],
  })
  console.log(response.output)
  return response.output
}

async function getRetrieverFromWikipedia(
  query: string,
): Promise<VectorStoreRetriever> {
  const tool = new WikipediaQueryRun({
    topKResults: 5,
    maxDocContentLength: -1,
    baseUrl: "https://de.wikipedia.org/w/api.php",
  })

  const result = await tool.invoke(query)
  const wikiPages = new Document({ pageContent: result })
  const splittedDocs = await splitDocuments([wikiPages])
  const vectorStore = await initializeVectorDatabase(splittedDocs)
  return vectorStore.asRetriever({ k: 4 })
}

async function splitDocuments(docs: Document[]): Promise<Document[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 20,
  })
  return splitter.splitDocuments(docs)
}

async function initializeVectorDatabase(
  documents: Document[],
): Promise<VectorStore> {
  const embeddings = new OllamaEmbeddings({
    model: "jina/jina-embeddings-v2-base-de",
  })
  return HNSWLib.fromDocuments(documents, embeddings) as any
}

const knowledgeTool = new DynamicStructuredTool({
  name: "wikipediaTool",
  description:
    "Hintergrundwissen zu einem bestimmten Thema und Fakt aus Wikipedia abrufen",
  schema: z.object({
    topic: z.string().describe("Der Titel des Wikipedia Artikels"),
    fact: z
      .string()
      .describe(
        "Der Fakt, zu dem eine Antwort aus dem Wikipedia-Artikel erhalten werden soll",
      ),
  }),
  callbacks: [toolCallbackFor("Wikipedia") as any],
  func: async ({ topic, fact }) => {
    const retriever = await getRetrieverFromWikipedia(topic)
    return retriever.pipe(formatDocumentsAsString).invoke(fact)
  },
})

async function getMemoryPrompt() {
  const MEMORY_KEY = "chat_history"
  return ChatPromptTemplate.fromMessages([
    [
      "system",
      `Du bist ein sehr erfahrener Geheimagent.
Deine Aufgabe ist es, historische Persönlichkeiten zu imitieren und Fragen im Namen dieser historischen Personen zu beantworten.
Antworte immer aus der Perspektive der Persona heraus.
Der Benutzer wird dir eine Persona vorgeben, und du musst diese mit recherchiertem Wissen verkörpern, als wärst du diese Person.
Passe den Sprachstil und den Ton deiner zugewiesenen Persona an.
Da du nichts über historische Personen weißt, musst du deine Werkzeuge nutzen, um die Fragen zu beantworten.
Recherchiere zuerst auf Wikipedia. Wenn du dort nichts findest nutze Tavily.
Antworte nur auf Grundlage des abgerufenen Wissens, erfinde keine neuen Informationen. 
Sag dem User nicht, dass du keine Quelle dazu finden konntest sondern antworte in der Rolle der Persona.`,
    ],
    new MessagesPlaceholder(MEMORY_KEY),
    new MessagesPlaceholder("agent_scratchpad"),
    ["user", "{input}"],
  ])
}

function toolCallbackFor(name: string) {
  return {
    handleToolStart: async (tool: Serialized, input) => {
      console.log(`Tool call for ${name}: ${input}`)
    },
  }
}

askAgent("Was ist dein Lieblingssong und wie startet er?")
