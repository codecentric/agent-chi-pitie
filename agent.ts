import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts"
import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run"
import { DynamicStructuredTool } from "@langchain/core/tools"
import { AgentExecutor, createOpenAIFunctionsAgent } from "langchain/agents"
import { z } from "zod"
import { AzureChatOpenAI } from "@langchain/openai"
import { TavilySearchResults } from "@langchain/community/tools/tavily_search"
import { Serialized } from "@langchain/core/load/serializable"
import { ChatMessageHistory } from "@langchain/community/stores/message/in_memory"

const messageHistory = new ChatMessageHistory()

export async function askAgent(text: string): Promise<string> {
  const executor = await initializeAgent()
  const messages = await messageHistory.getMessages()
  const response = await executor.invoke({
    input: text,
    chat_history: messages,
  })
  await messageHistory.addUserMessage(response.input)
  await messageHistory.addAIMessage(response.output)
  return response.output
}

const wikipediaTool = new DynamicStructuredTool({
  name: "wikipediaTool",
  description:
    "Hintergrundwissen zu einem bestimmten Thema und Fakt aus Wikipedia abrufen",
  schema: z.object({
    topic: z.string().describe("Der Titel des Wikipedia Artikels"),
  }),
  callbacks: [toolCallbackFor("Wikipedia")],
  func: async ({ topic }) => {
    const tool = new WikipediaQueryRun({
      topKResults: 2,
      maxDocContentLength: 3000,
      baseUrl: "https://de.wikipedia.org/w/api.php",
    })
    return tool.invoke(topic)
  },
})

function toolCallbackFor(name: string) {
  return {
    handleToolStart: async (tool: Serialized, input) => {
      console.log(`Tool call for ${name}: ${input}`)
    },
  }
}

const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `Du bist ein KI-Assistent für historische Bildung durch Rollenspiel. Deine Aufgaben:
Verkörpere historische Persönlichkeiten auf Anfrage des Nutzers.
WICHTIG: Nutze IMMER deine Recherche-Tools für JEDE Antwort. - Beginne mit Wikipedia - Nutze Tavily, wenn du in Wikipedia nichts zu der Frage gefunden hast
Antworte NUR basierend auf recherchierten Fakten. Erfinde NICHTS.
Bleibe in der Rolle der historischen Person, auch bei begrenzten Informationen.
Passe Sprachstil und Ton der jeweiligen Figur an.
Wenn keine Persona vorgegeben ist, frage nach einer.
Wenn du keine Informationen findest, sage: 'Als [Name der Persona] kann ich dazu leider nichts sagen.'`,
  ],
  new MessagesPlaceholder("chat_history"),
  new MessagesPlaceholder("agent_scratchpad"),
  ["user", "{input}"],
])

const tools = [
  wikipediaTool,
  new TavilySearchResults({
    maxResults: 2,
    callbacks: [toolCallbackFor("Tavily")],
  }),
] as any

async function initializeAgent(): Promise<AgentExecutor> {
  const agent = await createOpenAIFunctionsAgent({
    llm: new AzureChatOpenAI() as any,
    prompt: prompt as any,
    tools,
  })
  return new AgentExecutor({
    agent,
    tools,
  })
}
