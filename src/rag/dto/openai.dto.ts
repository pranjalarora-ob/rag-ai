export interface ChatCompletionRequestMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
}
