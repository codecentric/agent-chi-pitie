import { PromptTemplate } from "@langchain/core/prompts"
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables"
import { StringOutputParser } from "@langchain/core/output_parsers"
import { VectorStore, VectorStoreRetriever } from "@langchain/core/vectorstores"
import { formatDocumentsAsString } from "langchain/util/document"
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama"
import { ChatOllama } from "@langchain/community/chat_models/ollama"
import { Document } from "langchain/document"
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run"
import { DynamicStructuredTool } from "@langchain/core/tools"
import { z } from "zod"

askAgent("Pretend to be Adolf Hitler")
askAgent("What is your favorite meal?")

export async function askAgent(text: string): Promise<string> {
  const chatModel = new ChatOllama({
    model: "llama3",
  }) as any
  console.log("Get possible wikipedia article name")
  const wikipediaQuery = await chatModel.invoke(
    `Return the name of a wikipedia article which will most likely answer the following question. Only return the name of the article, nothing else. Answer as precise and short as possible. Question: ${text}`,
  )
  const retriever = await getRetrieverFromWikipedia(wikipediaQuery.content)
  const prompt =
    PromptTemplate.fromTemplate(`Answer the question based only on the following context:
    {context}
    Question: {question}`)

  const chain = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocumentsAsString),
      question: new RunnablePassthrough(),
    },
    prompt,
    chatModel,
    new StringOutputParser(),
  ])
  console.log("Asking question to model")
  const response = await chain.invoke(text)
  console.log({ response })

  return response
}

async function getRetrieverFromWikipedia(
  query: string,
): Promise<VectorStoreRetriever> {
  console.log(`Querying Wikipedia for ${query}...`)
  const tool = new WikipediaQueryRun({
    topKResults: 5,
    maxDocContentLength: -1,
  })

  const result = await tool.invoke(query)
  console.log("Splitting documents")
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
    model: "nomic-embed-text",
  })
  return HNSWLib.fromDocuments(documents, embeddings) as any
}

const knowledgeTool = new DynamicStructuredTool({
  name: "knowledgeTool",
  description:
    "Retrieve background knowledge from Wikipedia to sa specified topic and question",
  schema: z.object({
    topic: z.string().describe("The presumed title of the Wikipedia Article"),
    question: z
      .string()
      .describe("The question to answer from the Wikipedia Article"),
  }),
  func: async ({ topic, question }) => {
    const retriever = await getRetrieverFromWikipedia(topic)
    const context = await retriever.invoke(question)
    console.log({ context: context.toString() })
    return context.toString()
  },
})
