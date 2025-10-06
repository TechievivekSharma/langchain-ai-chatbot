import { Ollama } from "@langchain/community/llms/ollama";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { Document } from "langchain/document";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// 1️⃣ Local Model (Llama 3)
const llm = new Ollama({
  model: "llama3",
});

// 2️⃣ Use Ollama for Embeddings (local)
const embeddings = new OllamaEmbeddings({ model: "nomic-embed-text" });

// 3️⃣ Custom Data
const docs = [
  new Document({
    pageContent:
      "BlinkFixServe is a home service booking platform for AC repair, plumbing, painting, etc.",
  }),
  new Document({
    pageContent:
      "Users can easily book local professionals and track their service status in real time.",
  }),
];

// 4️⃣ Create vector store
const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);

const retriever = vectorStore.asRetriever();

// 5️⃣ Query
const query = "What is BlinkFixServe?";
const retrievedDocs = await retriever.getRelevantDocuments(query);
const context = retrievedDocs.map((d) => d.pageContent).join("\n");

console.log("🔎 Context Used:\n", context);

const prompt = `
You are an assistant who answers based only on the given context.
Context:
${context}
Question: ${query}
`;

const response = await llm.invoke(prompt);
console.log("\n🧠 AI Response:\n", response);
