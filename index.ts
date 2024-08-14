import { run } from "@backroad/backroad"
import { askAgent } from "./agent"

async function main() {
  startChatUI(askAgent)
}

function startChatUI(askAgent: Function) {
  run(async (br) => {
    const messages = br.getOrDefault("messages", [
      { by: "ai", content: "Hallo! Ich bin PiTie. ... Chi PiTie! 😎" },
    ])

    messages.forEach((message) => {
      br.chatMessage({ by: message.by }).write({ body: message.content })
    })

    const input = br.chatInput({ id: "input" })
    if (input) {
      const response = await askAgent(input)
      br.setValue("messages", [
        ...messages,
        { by: "human", content: input },
        { by: "ai", content: response },
      ])
    }
  })
}

main()
