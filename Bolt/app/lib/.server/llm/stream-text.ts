import type { Message } from 'ai';

interface StreamTextOptions {
  messages: Message[];
  system?: string;
  temperature?: number;
  maxTokens?: number;
}

interface GeminiResponse {
  candidates: Array<{
    content: {
      parts: Array<{
        text: string;
      }>;
    };
  }>;
}

export async function streamText(options: StreamTextOptions) {
  const { messages = [], system, temperature = 0.7, maxTokens = 1000 } = options;

  if (!messages || !Array.isArray(messages)) {
    throw new Error('Messages must be an array');
  }

  // Get API key from environment
  const apiKey = process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    throw new Error('GOOGLE_API_KEY is not set in environment variables');
  }

  // Prepare the conversation history
  const history = messages.map(m => ({
    role: m.role,
    parts: [{ text: m.content }]
  }));

  // Add system message if provided
  if (system) {
    history.unshift({
      role: 'system',
      parts: [{ text: system }]
    });
  }

  try {
    // Make direct API call to Gemini 2.0 Flash
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contents: history,
          generationConfig: {
            temperature,
            maxOutputTokens: maxTokens,
          },
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }

    const data = await response.json() as GeminiResponse;
    return data.candidates[0].content.parts[0].text;
  } catch (error) {
    console.error('Error generating response:', error);
    throw error;
  }
}
