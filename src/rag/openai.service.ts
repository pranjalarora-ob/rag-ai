import { Injectable, InternalServerErrorException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';
import { Response } from 'express';
import { ChatCompletionRequestMessage } from './dto/openai.dto';
import {
  START,
  END,
  MessagesAnnotation,
  StateGraph,
  Annotation,
  MemorySaver,
  messagesStateReducer,
} from '@langchain/langgraph';
import { QdrantService } from './qdrant.service';
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
  BaseMessage
} from '@langchain/core/messages';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';

/**
 * Generation and Embeddings go through OpenRouter (OpenAI-compatible).
 */
@Injectable()
export class OpenaiService {
  private readonly OPENROUTER_API_KEY: string;
  private readonly OPENROUTER_MODEL: string;
  private readonly EMBEDDING_MODEL: string;
  private readonly model: ChatGoogleGenerativeAI;

  private readonly OPENROUTER_EMBEDDINGS_URL = 'https://openrouter.ai/api/v1/embeddings';
  private readonly OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';

  constructor(
    private readonly configService: ConfigService,
    private readonly qdrantService: QdrantService,
  ) {
    this.OPENROUTER_API_KEY = this.configService.get('OPEN_ROUTER_API_KEY') || '';
    this.OPENROUTER_MODEL = this.configService.get('OPEN_ROUTER_MODEL') || 'openai/gpt-4o-mini';
    this.EMBEDDING_MODEL = this.configService.get('EMBEDDING_MODEL') || 'openai/text-embedding-3-small';
  }

  private get openRouterHeaders() {
    return {
      Authorization: `Bearer ${this.OPENROUTER_API_KEY}`,
      'Content-Type': 'application/json',
    };
  }

  // ---- Embeddings (OpenRouter/OpenAI) ----
  async generateEmbedding(input: string): Promise<number[]> {
    try {
      const response = await axios.post(
        this.OPENROUTER_EMBEDDINGS_URL,
        {
          model: this.EMBEDDING_MODEL,
          input,
        },
        {
          headers: this.openRouterHeaders,
        },
      );
      const embedding = response.data.data?.[0]?.embedding;
      if (!embedding || !Array.isArray(embedding)) {
        throw new Error('No embedding values returned from OpenRouter');
      }
      return embedding;
    } catch (error: any) {
      console.error('OpenRouter Embedding Error:', error?.response?.data || error?.message);
      throw new InternalServerErrorException(`Failed to generate embedding: ${error?.message}`);
    }
  }

  // ---- Batched embeddings — one HTTP call per `batchSize` texts (used by ingestion) ----
  async generateEmbeddings(inputs: string[], batchSize = 96): Promise<number[][]> {
    const out: number[][] = [];
    for (let i = 0; i < inputs.length; i += batchSize) {
      const batch = inputs.slice(i, i + batchSize);
      try {
        const response = await axios.post(
          this.OPENROUTER_EMBEDDINGS_URL,
          { model: this.EMBEDDING_MODEL, input: batch },
          { headers: this.openRouterHeaders },
        );
        const data = response.data?.data;
        if (!Array.isArray(data) || data.length !== batch.length) {
          throw new Error('Embedding count mismatch from provider');
        }
        // The API may return items out of order — sort by `index` before collecting.
        [...data].sort((a, b) => a.index - b.index).forEach((d) => out.push(d.embedding));
      } catch (error: any) {
        console.error('OpenRouter Batch Embedding Error:', error?.response?.data || error?.message);
        throw new InternalServerErrorException(`Failed to generate embeddings: ${error?.message}`);
      }
    }
    return out;
  }

  // ---- Streaming chat (OpenRouter) ----
  async streamOpenRouter(messages: ChatCompletionRequestMessage[], res: Response) {
    return new Promise(async (resolve) => {
      let result = '';
      const errors: any[] = [];

      res.setHeader('Content-Type', 'text/plain');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.flushHeaders();

      let response: any;
      try {
        response = await axios.post(
          this.OPENROUTER_URL,
          { model: this.OPENROUTER_MODEL, messages, temperature: 0.2, stream: true },
          { responseType: 'stream', headers: this.openRouterHeaders },
        );
      } catch (err: any) {
        console.error('Failed to start OpenRouter stream:', err?.response?.data || err?.message);
        errors.push(err);
        if (!res.writableEnded) res.status(500).end();
        return resolve({ result, errors });
      }

      let buffer = '';
      response.data.on('data', (chunk: Buffer) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const payload = trimmed.replace(/^data:\s*/, '');
          if (payload === '[DONE]') {
            if (!res.writableEnded) res.end();
            return resolve({ result, errors });
          }
          try {
            const data = JSON.parse(payload);
            const content = data.choices?.[0]?.delta?.content;
            if (content) {
              res.write(content);
              (res as any).flush?.();
              result += content;
            }
          } catch {
            // ignore keep-alive comments / partial chunks
          }
        }
      });
      response.data.on('end', () => {
        if (!res.writableEnded) res.end();
        resolve({ result, errors });
      });
      response.data.on('error', (err: any) => {
        console.error('OpenRouter stream error:', err);
        errors.push(err);
        if (!res.writableEnded) res.status(500).end();
        resolve({ result, errors });
      });
      res.on('close', () => {
        if (!res.writableEnded) res.end();
        resolve({ result, errors });
      });
    });
  }

  // ---- Non-streaming text (OpenRouter) — used for routing/classification ----
  async openRouterGenerate(prompt: string, temperature = 0): Promise<string> {
    try {
      const res = await axios.post(
        this.OPENROUTER_URL,
        { model: this.OPENROUTER_MODEL, messages: [{ role: 'user', content: prompt }], temperature },
        { headers: this.openRouterHeaders },
      );
      return res.data?.choices?.[0]?.message?.content || '';
    } catch (error: any) {
      console.error('OpenRouter generate error:', error?.response?.data || error?.message);
      return '';
    }
  }

  // ---- One tool-calling turn (OpenRouter) — returns the assistant message (may hold tool_calls) ----
  async openRouterToolTurn(messages: any[], tools: any[]): Promise<any> {
    const res = await axios.post(
      this.OPENROUTER_URL,
      { model: this.OPENROUTER_MODEL, messages, tools, tool_choice: 'auto', temperature: 0.2 },
      { headers: this.openRouterHeaders },
    );
    return res.data?.choices?.[0]?.message;
  }

  // Greedily join chunks up to an approximate token budget (no tokenizer dependency).
  async truncateByTokens(texts: string[], maxTokens = Infinity) {
    const selected: string[] = [];
    let total = 0;
    for (const text of texts) {
      const tokens = Math.ceil((text?.split(/\s+/).length || 0) * 1.3);
      if (total + tokens > maxTokens) break;
      selected.push(text);
      total += tokens;
    }
    return selected.join('\n');
  }

  async normalizeSemanticQuestion(
    text: string,
  ): Promise<string> {
    const stopWords = new Set([
      'what',
      'which',
      'does',
      'do',
      'is',
      'are',
      'the',
      'a',
      'an',
      'how',
      'why',
      'when',
      'where',
      'who',
      'kind',
      'type',
      'of',
      'to',
      'and',
      'or',
      'in',
      'on',
      'for',
      'this',
      'that',
      'these',
      'those',
    ]);

    const synonymMap: Record<
      string,
      string
    > = {
      meeting: 'interview',
      discussion: 'interview',

      takeaways: 'points',
      takeaway: 'points',

      summary: 'overview',
    };

    const words =
      text
        .toLowerCase()
        .match(/\b[a-z0-9-]+\b/g) || [];

    const normalizedWords =
      words
        .filter(
          word => !stopWords.has(word),
        )
        .map(
          word =>
            synonymMap[word] || word,
        );

    return normalizedWords.join(' ');
  }

  async generateGeminiEmbedding(input: string): Promise<number[]> {
    try {

      // const response = await axios.post(`${this.GEMINI_EMBEDDING_URL}?key=${this.GEMINI_API_KEY}`, {
      //   // model: 'models/text-embedding-004',
      //   content: { parts: [{ text: input }] },
      //   taskType: 'RETRIEVAL_DOCUMENT',
      //   // outputDimensionality: 1536,
      //   outputDimensionality: 3072,
      // });

      const response = await axios.post(
        'https://openrouter.ai/api/v1/embeddings',
        {
          model: 'google/gemini-embedding-001',
          input,
          dimensions: 3072,
          encoding_format: 'float',
        },
        {
          headers: {
            Authorization: `Bearer ${this.OPENROUTER_API_KEY}`,
            'Content-Type': 'application/json',
          },
        },
      );
      // console.log('embeding output', response)
      const embedding = response.data.data?.[0]?.embedding;


      // const embedding = response.data.embedding?.values;

      if (!embedding || !Array.isArray(embedding)) {
        throw new Error('No embedding values returned from Gemini');
      }

      return embedding;
    } catch (error) {
      console.error('Gemini Embedding Error:', error);
      throw new InternalServerErrorException(`Failed to generate Gemini embedding: ${error}`);
    }
  }

  agentGraphV8() {
    // ============ 1. GRAPH ANNOTATION ============

    const GraphAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
      }),
      question: Annotation<string>(),
      normalizedQuestion: Annotation<string>(),
      embedding: Annotation<number[]>(),
      route: Annotation<string>(),
      customerId: Annotation<string>(),
      context: Annotation<string>(),
      summary: Annotation<string>(),
      costAnalysis: Annotation<string>(),
      zoneAnalysis: Annotation<string>(),
      result: Annotation<string>(),
      metadata: Annotation<any>(),
    });

    type GraphState = typeof GraphAnnotation.State;

    // ============ 2. NEW: EXTRACT STAGE ============

    const extractStage = (question: string): string | undefined => {
      const lowerQuestion = question.toLowerCase();

      const stageMap: Record<string, string> = {
        // Pre-Sales
        'pre sales': 'Pre-Sales',
        'pre-sales': 'Pre-Sales',
        'presales': 'Pre-Sales',
        'pre sale': 'Pre-Sales',
        'sales': 'Pre-Sales',

        // Design-Sales
        'design sales': 'Design-Sales',
        'design-sales': 'Design-Sales',
        'designsales': 'Design-Sales',
        'sales design': 'Design-Sales',

        // Design Delivery
        'design delivery': 'Design Delivery',
        'design-delivery': 'Design Delivery',
        'designdelivery': 'Design Delivery',
        'delivery': 'Design Delivery',
        'design': 'Design Delivery',

        // Execution
        'execution': 'Execution',
        'executing': 'Execution',
        'execute': 'Execution',
        'executed': 'Execution',
        'in execution': 'Execution',
        'under execution': 'Execution',

        // Handover
        'handover': 'Handover',
        'hand over': 'Handover',
        'hand-over': 'Handover',
        'handed over': 'Handover',
        'completed': 'Handover',
        'completion': 'Handover',
        'finished': 'Handover',
        'done': 'Handover',
      };

      for (const [keyword, stageValue] of Object.entries(stageMap)) {
        if (lowerQuestion.includes(keyword)) {
          console.log(`📍 Stage extracted: "${keyword}" → "${stageValue}"`);
          return stageValue;
        }
      }

      return undefined;
    };

    // ============ 3. NEW: EXTRACT STATUS ============

    const extractStatus = (question: string): string | undefined => {
      const lowerQuestion = question.toLowerCase();

      const statusMap: Record<string, string> = {
        // Cancelled
        'cancelled': 'Cancelled',
        'canceled': 'Cancelled',
        'cancel': 'Cancelled',
        'terminated': 'Cancelled',
        'dropped': 'Cancelled',
        'abandoned': 'Cancelled',

        // Completed
        'completed': 'Completed',
        'complete': 'Completed',
        'completion': 'Completed',
        'finished': 'Completed',
        'finish': 'Completed',
        'done': 'Completed',
        'closed': 'Completed',

        // InProgress
        'in progress': 'InProgress',
        'in-progress': 'InProgress',
        'inprogress': 'InProgress',
        'ongoing': 'InProgress',
        'active': 'InProgress',
        'running': 'InProgress',
        'under progress': 'InProgress',
        'work in progress': 'InProgress',

        // OnHold
        'on hold': 'OnHold',
        'onhold': 'OnHold',
        'hold': 'OnHold',
        'paused': 'OnHold',
        'pause': 'OnHold',
        'suspended': 'OnHold',
        'deferred': 'OnHold',
      };

      for (const [keyword, statusValue] of Object.entries(statusMap)) {
        if (lowerQuestion.includes(keyword)) {
          console.log(`🔵 Status extracted: "${keyword}" → "${statusValue}"`);
          return statusValue;
        }
      }

      return undefined;
    };

    // ============ 4. NEW: EXTRACT DYNAMIC KEYWORDS ============

    interface DynamicKeyword {
      field: string;
      value: string;
      type: 'exact' | 'contains';
    }

    // const extractDynamicKeywords = (question: string): DynamicKeyword[] => {
    //   const keywords: DynamicKeyword[] = [];
    //   const lowerQuestion = question.toLowerCase();

    //   // Map cities to their storage values (matching your LEAD_ZONES)
    //   const fieldMappings: Record<string, Record<string, string>> = {
    //     zone: {
    //       'bangalore': 'Bangalore',
    //       'kolkata': 'Kolkata',
    //       'mumbai': 'Mumbai',
    //       'gurgaon': 'Gurgaon',
    //       'unassigned': 'Unassigned',
    //     },
    //     city: {
    //       'bangalore': 'bangalore',
    //       'kolkata': 'kolkata',
    //       'mumbai': 'mumbai',
    //       'gurgaon': 'gurgaon',
    //       'delhi': 'delhi',
    //       'pune': 'pune',
    //       'hyderabad': 'hyderabad',
    //       'vadodara': 'vadodara',
    //     },
    //     stage: {
    //       'execution': 'execution',
    //       'planning': 'planning',
    //       'completed': 'completed',
    //       'in progress': 'in_progress',
    //       'initiated': 'initiated',
    //     },
    //     status: {
    //       'active': 'active',
    //       'inactive': 'inactive',
    //       'pending': 'pending',
    //       'approved': 'approved',
    //     },
    //     channel: {
    //       'direct': 'direct',
    //       'indirect': 'indirect',
    //       'digital': 'digital',
    //       'referral': 'referral',
    //     },
    //     leadType: {
    //       'new office': 'new_office',
    //       'new branch': 'new_branch',
    //       'expansion': 'expansion',
    //       'retail': 'retail',
    //       'commercial': 'commercial',
    //     },
    //   };

    //   for (const [field, keywords_map] of Object.entries(fieldMappings)) {
    //     for (const [keyword, value] of Object.entries(keywords_map)) {
    //       if (lowerQuestion.includes(keyword)) {
    //         keywords.push({
    //           field,
    //           value,
    //           type: 'exact',
    //         });
    //         console.log(`🏷️ Dynamic keyword: field="${field}", value="${value}"`);
    //       }
    //     }
    //   }

    //   return keywords;
    // };

    // ============ 5. EXTRACT COST RANGE ============

    const extractCostRange = (question: string): {
      min?: number;
      max?: number;
      operator?: string;
      hasCostKeyword?: boolean;
    } => {
      const result: {
        min?: number;
        max?: number;
        operator?: string;
        hasCostKeyword?: boolean;
      } = {};

      const lowerQuestion = question.toLowerCase();

      const costKeywords = [
        'cost',
        'boqvalue',
        'boquvalue',
        'boq',
        'budget',
        'price',
        'financial',
        'amount',
        'value',
        'overrun',
        'spend',
        'expense',
        'payment',
      ];

      const hasCostKeyword = costKeywords.some((keyword) =>
        lowerQuestion.includes(keyword)
      );

      if (!hasCostKeyword) {
        console.log(`🚫 Cost extraction skipped: no cost keyword found`);
        return result;
      }

      result.hasCostKeyword = true;

      const convertToRupees = (value: number, unit: string): number => {
        const normalizedUnit = unit.toLowerCase();

        if (normalizedUnit === 'k') {
          return value * 1000;
        }

        if (normalizedUnit === 'c' || normalizedUnit === 'crore') {
          return value * 10000000;
        }

        if (
          normalizedUnit === 'l' ||
          normalizedUnit === 'lakh' ||
          normalizedUnit === 'lac'
        ) {
          return value * 100000;
        }

        return value;
      };

      const greaterMatch = question.match(
        /(?:cost|boqvalue|boquvalue|boq|budget|price|financial|amount|value|overrun|spend|expense|payment)\s+(?:greater\s+than|more\s+than|>|above|exceeds?)\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(k|lakh|lac|l|crore|c)?/i
      );

      if (greaterMatch) {
        const numberMatch = greaterMatch[0].match(/(\d+(?:\.\d+)?)\s*(k|lakh|lac|l|crore|c)?/i);

        if (numberMatch) {
          let value = parseFloat(numberMatch[1]);
          const unit = numberMatch[2] || 'lakh';

          value = convertToRupees(value, unit);
          result.min = value;
          result.operator = 'greater_than';

          console.log(`💰 Cost extracted (cost > pattern): ${numberMatch[1]} ${unit} = ₹${value.toLocaleString('en-IN')}`);
        }
      }

      const lessMatch = question.match(
        /(?:cost|boqvalue|boquvalue|boq|budget|price|financial|amount|value|overrun|spend|expense|payment)\s+(?:less\s+than|under|<|below|upto|up\s+to)\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(k|lakh|lac|l|crore|c)?/i
      );

      if (lessMatch) {
        const numberMatch = lessMatch[0].match(/(\d+(?:\.\d+)?)\s*(k|lakh|lac|l|crore|c)?/i);

        if (numberMatch) {
          let value = parseFloat(numberMatch[1]);
          const unit = numberMatch[2] || 'lakh';

          value = convertToRupees(value, unit);
          result.max = value;
          result.operator = 'less_than';

          console.log(`💰 Cost extracted (cost < pattern): ${numberMatch[1]} ${unit} = ₹${value.toLocaleString('en-IN')}`);
        }
      }

      const betweenMatch = question.match(
        /(?:cost|boqvalue|boquvalue|boq|budget|price|financial|amount|value|overrun|spend|expense|payment)\s+(?:between|from|through)?\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(k|lakh|lac|l|crore|c)?/i
      );

      if (betweenMatch) {
        const min = parseFloat(betweenMatch[1]);
        const max = parseFloat(betweenMatch[2]);
        const unit = betweenMatch[3] || 'lakh';

        result.min = convertToRupees(min, unit);
        result.max = convertToRupees(max, unit);
        result.operator = 'between';

        console.log(
          `💰 Cost extracted (cost between): ${min}-${max} ${unit} = ₹${result.min.toLocaleString('en-IN')}-₹${result.max.toLocaleString('en-IN')}`
        );
      }

      const currencyMatch = question.match(
        /(?:cost|boqvalue|boquvalue|boq|budget|price|financial|amount|value|overrun|spend|expense|payment)\s+[^\d]*₹\s*(\d+(?:\.\d+)?)\s*(k|lakh|lac|l|crore|c)?/i
      );

      if (currencyMatch) {
        const numberMatch = currencyMatch[0].match(/₹\s*(\d+(?:\.\d+)?)\s*(k|lakh|lac|l|crore|c)?/i);

        if (numberMatch) {
          let value = parseFloat(numberMatch[1]);
          const unit = numberMatch[2] || 'lakh';

          value = convertToRupees(value, unit);

          if (!result.min && !result.max) {
            result.min = value;
            console.log(`💰 Cost extracted (currency): ₹${value.toLocaleString('en-IN')}`);
          }
        }
      }

      if (result.min || result.max || result.operator) {
        console.log('✅ Final cost range:', {
          min: result.min ? `₹${result.min.toLocaleString('en-IN')}` : undefined,
          max: result.max ? `₹${result.max.toLocaleString('en-IN')}` : undefined,
          operator: result.operator,
          hasCostKeyword: result.hasCostKeyword,
        });
      }

      return result;
    };

    const extractMappedValue = (
      question: string,
      map: Record<string, string>,
    ): string[] => {
      const lower = question.toLowerCase();

      return [
        ...new Set(
          Object.entries(map)
            .filter(([alias]) => lower.includes(alias))
            .map(([, value]) => value),
        ),
      ];
    };

    // ============ 6. EXTRACT AREA RANGE ============

    const extractAreaRange = (question: string): { min?: number; max?: number; unit?: string; operator?: string } => {
      const result: { min?: number; max?: number; unit?: string; operator?: string } = {};

      const greaterMatch = question.match(/(?:greater\s+than|more\s+than|>|above)\s*(?:₹\s*)?(\d+(?:\.\d+)?)\s*(sq\.?m|sqm|sq\.?ft|sqft)?/i);
      if (greaterMatch) {
        let value = parseFloat(greaterMatch[1]);
        const unit = greaterMatch[2]?.toLowerCase().replace(/\./g, '') || "sqft";

        if (unit === 'sqm') {
          value = value * 10.764;
        }

        result.min = value;
        result.unit = "sqft";
        result.operator = "greater_than";
      }

      const lessMatch = question.match(/(?:less\s+than|under|<|below)\s*(?:₹\s*)?(\d+(?:\.\d+)?)\s*(sq\.?m|sqm|sq\.?ft|sqft)?/i);
      if (lessMatch) {
        let value = parseFloat(lessMatch[1]);
        const unit = lessMatch[2]?.toLowerCase().replace(/\./g, '') || "sqft";

        if (unit === 'sqm') {
          value = value * 10.764;
        }

        result.max = value;
        result.unit = "sqft";
        result.operator = "less_than";
      }

      const betweenMatch = question.match(/between\s*(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*(\d+(?:\.\d+)?)\s*(sq\.?m|sqm|sq\.?ft|sqft)?/i);
      if (betweenMatch) {
        const unit = betweenMatch[3]?.toLowerCase().replace(/\./g, '') || "sqft";
        let min = parseFloat(betweenMatch[1]);
        let max = parseFloat(betweenMatch[2]);

        if (unit === 'sqm') {
          min = min * 10.764;
          max = max * 10.764;
        }

        result.min = min;
        result.max = max;
        result.unit = "sqft";
        result.operator = "between";
      }

      const aboveMatch = question.match(/above\s*(\d+(?:\.\d+)?)\s*(sq\.?m|sqm|sq\.?ft|sqft)?/i);
      if (aboveMatch) {
        let value = parseFloat(aboveMatch[1]);
        const unit = aboveMatch[2]?.toLowerCase().replace(/\./g, '') || "sqft";

        if (unit === 'sqm') {
          value = value * 10.764;
        }

        result.min = value;
        result.unit = "sqft";
        result.operator = "above";
      }

      if (Object.keys(result).length > 0) {
        console.log("📐 Extracted area range:", result);
      }

      return result;
    };

    // ============ 7. INTENT DETECTION (UPDATED) ============

    const detectIntent = (question: string): any => {
      const lowerQuestion = question.toLowerCase();

      const projectIdMatch = question.match(
        /project\s+([A-Za-z0-9_-]+)|project\s+([A-Za-z0-9_-]+)/i
      );
      const projectId = projectIdMatch?.[1] || projectIdMatch?.[2];

      const leadTypeMap: Record<string, string> = {
        // New Office
        'new office': 'New Office',
        'newoffice': 'New Office',
        'new': 'New Office',

        // Office Renovation
        'office renovation': 'Office Renovation',
        'office-renovation': 'Office Renovation',
        'officereno': 'Office Renovation',

        // Renovation
        'renovation': 'Renovation',
        'renovate': 'Renovation',
        'reno': 'Renovation',
      };

      const ownerMap: Record<string, string> = {
        // Enterprise Team
        'enterprise team': 'Enterprise Team',
        'enterprise': 'Enterprise Team',

        // Product Team
        'product team': 'Product Team',
        'product': 'Product Team',

        // Random Owner 2
        'random owner 2': 'Random Owner 2',
        'random owner2': 'Random Owner 2',
        'owner 2': 'Random Owner 2',

        // SKV
        'skv': 'SKV',

        // SMB Team
        'smb team': 'SMB Team',
        'smb': 'SMB Team',

        // Team B
        'team b': 'Team B',
        'teamb': 'Team B',
      };

      const LEAD_ZONES = {
        Bangalore: "South",
        Kolkata: "East",
        Mumbai: "West",
        Gurgaon: "North",
        Unassigned: "Unassigned",
      } as const;

      const REGION_TO_ZONES = Object.entries(LEAD_ZONES).reduce(
        (acc, [zoneName, region]) => {
          const key = region.toLowerCase();

          if (!acc[key]) {
            acc[key] = [];
          }

          acc[key].push(zoneName);

          return acc;
        },
        {} as Record<string, string[]>,
      );

      const zoneMatches = question.match(
        /\b(north|south|east|west|north-?east|north-?west|south-?east|south-?west)\b/gi,
      );

      let zones: string[] | undefined;

      if (zoneMatches) {
        const regions = Array.from(new Set(zoneMatches.map((z) => z.toLowerCase())));

        const expandedRegions = regions
          .flatMap((region) => {
            switch (region) {
              case "north-east":
              case "northeast":
                return ["north", "east"];

              case "north-west":
              case "northwest":
                return ["north", "west"];

              case "south-east":
              case "southeast":
                return ["south", "east"];

              case "south-west":
              case "southwest":
                return ["south", "west"];

              default:
                return [region];
            }
          });

        zones = [
          ...new Set(
            expandedRegions.flatMap(
              (region) => REGION_TO_ZONES[region] ?? [],
            ),
          ),
        ];
      }

      console.log("Detected zones:", zones);

      const cityMatches = question.match(
        /(vadodara|bangalore|delhi|mumbai|pune|hyderabad|chennai|kolkata)/gi
      );
      const cities = cityMatches
        ? Array.from(new Set(cityMatches.map((c) => c.toLowerCase())))
        : undefined;

      // NEW: Extract stage, status, and dynamic keywords
      const stage = extractStage(question);
      const status = extractStatus(question);
      // const dynamicKeywords = extractDynamicKeywords(question);

      const costRange = extractCostRange(question);
      const areaRange = extractAreaRange(question);
      const leadTypes = extractMappedValue(question, leadTypeMap);
      const owners = extractMappedValue(question, ownerMap);

      let type = "rag";

      if (
        (/across|compare|analysis|breakdown|performance/i.test(lowerQuestion) &&
          (zones || cities)) ||
        /\b(north|south|east|west)\b.*\b(north|south|east|west)\b/i.test(question)
      ) {
        type = "zone_analysis";
      } else if (/cost|budget|overrun|price|value|financial|expensive|lakh|crore/i.test(lowerQuestion)) {
        type = "cost_analysis";
      } else if (/summary|overview|status|progress|achievement/i.test(lowerQuestion)) {
        type = "summary";
      }

      if (zones?.length) {
        for (const [zone, region] of Object.entries(LEAD_ZONES)) {
          const regex = new RegExp(`\\b${region}\\b`, "gi");
          question = question.replace(regex, zone);
        }
      }

      return {
        type,
        projectId,
        zones,
        cities,
        stage,          // NEW!
        status,         // NEW!
        // dynamicKeywords, // NEW!
        leadTypes,
        owners,
        costRange,
        areaRange,
        limit: type === "zone_analysis" ? 25 : type === "rag" ? 7 : 15,
        question,
      };
    };

    // ============ 8. ORGANIZE CHUNKS ============

    const organizeChunksByType = (chunks: any[]): Record<string, any[]> => {
      const organized: Record<string, any[]> = {
        details: [],
        flow: [],
        financial: [],
      };

      chunks.forEach((chunk) => {
        const docType = chunk.payload?.document_type || "details";
        if (!organized[docType]) {
          organized[docType] = [];
        }
        organized[docType].push(chunk);
      });

      return organized;
    };

    // ============ 9. ZONE ANALYSIS AGGREGATION ============

    const aggregateByZone = (chunks: any[]): Record<string, any[]> => {
      const byZone: Record<string, any[]> = {};

      chunks.forEach((chunk) => {
        const zone = chunk.payload?.zone || "Unknown";
        if (!byZone[zone]) {
          byZone[zone] = [];
        }
        byZone[zone].push(chunk);
      });

      return byZone;
    };

    const aggregateByCity = (chunks: any[]): Record<string, any[]> => {
      const byCity: Record<string, any[]> = {};

      chunks.forEach((chunk) => {
        const city = chunk.payload?.city || "Unknown";
        if (!byCity[city]) {
          byCity[city] = [];
        }
        byCity[city].push(chunk);
      });

      return byCity;
    };

    // ============ 10. FORMAT FUNCTIONS ============

    const formatCurrency = (value: number): string => {
      if (value >= 10000000) {
        return `₹${(value / 10000000).toFixed(1)} Cr`;
      } else if (value >= 100000) {
        return `₹${(value / 100000).toFixed(1)} L`;
      } else {
        return `₹${value.toLocaleString("en-IN")}`;
      }
    };

    const formatAreaRangeContext = (
      chunksByType: Record<string, any[]>,
      areaRange: { min?: number; max?: number; unit?: string }
    ): string => {
      const sections: string[] = [];

      sections.push("AREA ANALYSIS");
      sections.push("=".repeat(80));
      sections.push("");

      if (areaRange.min && areaRange.max) {
        sections.push(`Filter: ${areaRange.min.toFixed(0)} - ${areaRange.max.toFixed(0)} sqft`);
      } else if (areaRange.min) {
        sections.push(`Filter: Area > ${areaRange.min.toFixed(0)} sqft`);
      } else if (areaRange.max) {
        sections.push(`Filter: Area < ${areaRange.max.toFixed(0)} sqft`);
      }
      sections.push("");

      if (chunksByType.details && chunksByType.details.length > 0) {
        sections.push("LARGE-AREA PROJECTS:");
        sections.push("-".repeat(80));

        const sorted = [...chunksByType.details].sort((a, b) => {
          const aArea = a.payload?.area || 0;
          const bArea = b.payload?.area || 0;
          return bArea - aArea;
        });

        sorted.forEach((chunk, idx) => {
          const area = chunk.payload?.area || 0;
          const projectCode = chunk.payload?.projectCode || "Unknown";
          const zone = chunk.payload?.zone || "Unknown";

          sections.push(`\n[Project ${idx + 1}] ${projectCode} (${zone})`);
          sections.push(`Area: ${area.toFixed(0)} sqft`);
          sections.push(chunk.payload?.text || "");
        });
      }

      return sections.join("\n");
    };

    const formatCostRangeContext = (
      chunksByType: Record<string, any[]>,
      costRange: { min?: number; max?: number; operator?: string }
    ): string => {
      const sections: string[] = [];

      sections.push("COST RANGE ANALYSIS");
      sections.push("=".repeat(80));
      sections.push("");

      if (costRange.min && costRange.max) {
        sections.push(`Filter: ${formatCurrency(costRange.min)} - ${formatCurrency(costRange.max)}`);
      } else if (costRange.min) {
        sections.push(`Filter: Cost > ${formatCurrency(costRange.min)}`);
      } else if (costRange.max) {
        sections.push(`Filter: Cost < ${formatCurrency(costRange.max)}`);
      }
      sections.push("");

      // if (chunksByType.details && chunksByType.details.length > 0) {
      //   sections.push("HIGH-VALUE PROJECTS:");
      //   sections.push("-".repeat(80));

      //   const sorted = [...chunksByType.details].sort((a, b) => {
      //     const aValue = a.payload?.boqValue || 0;
      //     const bValue = b.payload?.boqValue || 0;
      //     return bValue - aValue;
      //   });

      //   sorted.forEach((chunk, idx) => {
      //     const value = chunk.payload?.boqValue || 0;
      //     const projectCode = chunk.payload?.projectCode || "Unknown";
      //     const zone = chunk.payload?.zone || "Unknown";

      //     sections.push(`\n[Project ${idx + 1}] ${projectCode} (${zone})`);
      //     sections.push(`Cost: ${formatCurrency(value)}`);
      //     sections.push(`Area: ${chunk.payload?.area ? chunk.payload.area.toFixed(0) + " sqft" : "N/A"}`);
      //     sections.push(chunk.payload?.text || "");
      //   });
      // }

       if (chunksByType.financial && chunksByType.financial.length > 0) {
        sections.push("HIGH-VALUE PROJECTS:");
        sections.push("-".repeat(80));

        const sorted = [...chunksByType.financial].sort((a, b) => {
          const aValue = a.payload?.boqValue || 0;
          const bValue = b.payload?.boqValue || 0;
          return bValue - aValue;
        });

        sorted.forEach((chunk, idx) => {
          const value = chunk.payload?.boqValue || 0;
          const projectCode = chunk.payload?.projectCode || "Unknown";
          const zone = chunk.payload?.zone || "Unknown";

          sections.push(`\n[Project ${idx + 1}] ${projectCode} (${zone})`);
          sections.push(`Cost: ${formatCurrency(value)}`);
          sections.push(`area: ${chunk.payload?.area ? chunk.payload.area.toFixed(0) + " sqft" : "N/A"}`);
          sections.push(chunk.payload?.text || "");

        });
      }

      if (chunksByType.details && chunksByType.details.length > 0) {
        sections.push("\nPROJECT DETAILS:");
        sections.push("-".repeat(80));
        chunksByType.details.forEach((chunk) => {
          const text = chunk.payload?.text || "";
          const boqValue = chunk.payload?.boqValue || 0;
          const relevantLines = text
            .split("\n")
            .filter((line: string) =>
              /project|budget|cost|estimated|actual|value|area|scope/i.test(line)
            );
          if (relevantLines.length > 0) {
            sections.push(relevantLines.join("\n"));
            sections.push(`Cost of ${chunk.payload?.projectCode || "Unknown"}: ${formatCurrency(boqValue)}`);
          }
        });
      }

      return sections.join("\n");
    };

    const formatCostAnalysisContext = (
      chunksByType: Record<string, any[]>,
      costRange?: { min?: number; max?: number }
    ): string => {
      if (costRange && (costRange.min !== undefined || costRange.max !== undefined)) {
        return formatCostRangeContext(chunksByType, costRange);
      }

      const sections: string[] = [];

      sections.push("FINANCIAL ANALYSIS CONTEXT");
      sections.push("=".repeat(80));
      sections.push("");

      // if (chunksByType.details && chunksByType.details.length > 0) {
      //   sections.push("COST DATA:");
      //   sections.push("-".repeat(80));
      //   chunksByType.details.forEach((chunk, idx) => {
      //     sections.push(`[Document ${idx + 1}]`);
      //     sections.push(chunk.payload?.text || "");
      //     sections.push("");
      //   });
      // }


      if (chunksByType.financial && chunksByType.financial.length > 0) {
        sections.push("COST DATA:");
        sections.push("-".repeat(80));
        chunksByType.financial.forEach((chunk, idx) => {
          sections.push(`[Document ${idx + 1}]`);
          sections.push(chunk.payload?.text || "");
          sections.push("");
        });
      }

      if (chunksByType.details && chunksByType.details.length > 0) {
        sections.push("PROJECT DETAILS:");
        sections.push("-".repeat(80));
        chunksByType.details.forEach((chunk) => {
          const text = chunk.payload?.text || "";
          const relevantLines = text
            .split("\n")
            .filter((line: string) =>
              /project|budget|cost|estimated|actual|value|area|scope/i.test(line)
            );
          if (relevantLines.length > 0) {
            sections.push(relevantLines.join("\n"));
          }
        });
      }

      return sections.join("\n");
    };

    const formatSummaryContext = (
      chunksByType: Record<string, any[]>
    ): string => {
      const sections: string[] = [];

      sections.push("PROJECT SUMMARY CONTEXT");
      sections.push("=".repeat(80));
      sections.push("");

      if (chunksByType.details && chunksByType.details.length > 0) {
        sections.push("PROJECT INFORMATION:");
        sections.push("-".repeat(80));
        chunksByType.details.forEach((chunk) => {
          sections.push(chunk.payload?.text || "");
        });
        sections.push("");
      }

      if (chunksByType.flow && chunksByType.flow.length > 0) {
        sections.push("PROJECT STATUS:");
        sections.push("-".repeat(80));
        chunksByType.flow.forEach((chunk) => {
          // sections.push(chunk.payload?.text || "");
        });
      }

      return sections.join("\n");
    };

    const formatZoneAnalysisContext = (
      byZone: Record<string, any[]>,
      byCity: Record<string, any[]>,
      intent: any
    ): string => {
      const sections: string[] = [];

      sections.push("ZONE/REGION ANALYSIS CONTEXT");
      sections.push("=".repeat(80));
      sections.push("");

      if (intent.zones && intent.zones.length > 0) {
        sections.push("DATA BY ZONE:");
        sections.push("-".repeat(80));

        intent.zones.forEach((zone: string) => {
          const zoneChunks = byZone[zone] || [];
          if (zoneChunks.length > 0) {
            sections.push(`\n📍 ZONE: ${zone.toUpperCase()}`);
            sections.push(`   Projects: ${zoneChunks.length}`);

            const projects = new Set(zoneChunks.map((c) => c.payload?.projectId));
            sections.push(`   Unique Projects: ${projects.size}`);

            zoneChunks.slice(0, 2).forEach((chunk) => {
              const text = chunk.payload?.text || "";
              const projectName = chunk.payload?.metadata?.projectName || "Unknown Project";
              const customerName = chunk.payload?.metadata?.customerName || "Unknown Customer";
              let summary = `Project: ${projectName}, Customer: ${customerName}, cost: ${chunk.payload?.boqValue}`;
              summary = text.split("\n").slice(0, 3).join("\n");
              sections.push(`   ${summary}`);
            });
          }
        });
        sections.push("");
      }

      if (intent.cities && intent.cities.length > 0) {
        sections.push("DATA BY CITY:");
        sections.push("-".repeat(80));

        intent.cities.forEach((city: string) => {
          const cityChunks = byCity[city] || [];
          if (cityChunks.length > 0) {
            sections.push(`\n🏙️ CITY: ${city.toUpperCase()}`);

            const projects = new Set(cityChunks.map((c) => c.payload?.projectId));
            sections.push(`   Unique Projects: ${projects.size}`);

            const totalCost = cityChunks.reduce((sum, c) => {
              return sum + (c.payload?.boqValue || 0);
            }, 0);
            sections.push(`   Total Value: ${formatCurrency(totalCost)}`);
          }
        });
        sections.push("");
      }

      sections.push("AGGREGATED METRICS:");
      sections.push("-".repeat(80));

      const allChunks = Object.values(byZone).flat();
      const uniqueProjects = new Set(allChunks.map((c) => c.payload?.projectId)).size;
      const totalValue = allChunks.reduce(
        (sum, c) => sum + (c.payload?.boqValue || 0),
        0
      );
      const completedProjects = allChunks.filter(
        (c) => c.payload?.status === "Completed"
      ).length;

      sections.push(`Total Projects Analyzed: ${uniqueProjects}`);
      sections.push(`Total Value: ${formatCurrency(totalValue)}`);
      sections.push(`Completed: ${completedProjects}`);
      sections.push(`In Progress: ${allChunks.length - completedProjects}`);

      return sections.join("\n");
    };

    const formatGeneralContext = (
      chunksByType: Record<string, any[]>
    ): string => {
      const sections: string[] = [];

      sections.push("PROJECT CONTEXT");
      sections.push("=".repeat(80));
      sections.push("");

      if (chunksByType.details && chunksByType.details.length > 0) {
        sections.push("PROJECT DETAILS:");
        sections.push("-".repeat(80));
        chunksByType.details.slice(0, 2).forEach((chunk) => {
          sections.push(chunk.payload?.text || "");
        });
        sections.push("");
      }

      if (chunksByType.flow && chunksByType.flow.length > 0) {
        sections.push("PROJECT FLOW:");
        sections.push("-".repeat(80));
        chunksByType.flow.slice(0, 2).forEach((chunk) => {
          sections.push(chunk.payload?.text || "");
        });
      }

      if (chunksByType.financial && chunksByType.financial.length > 0) {
        sections.push("FINANCIAL INFO:");
        sections.push("-".repeat(80));
        chunksByType.financial.slice(0, 2).forEach((chunk) => {
          sections.push(chunk.payload?.text || "");
        });
      }

      return sections.join("\n");
    };

    // ============ 11. GET SYSTEM PROMPT ============

    const getSystemPrompt = (intentType: string): string => {
      const basePrompt = `You are a helpful AI assistant analyzing project data.
        Answer the user's question based ONLY on the provided context.
        Be specific, data-driven, and actionable.
        If information is not in the context, say so clearly. should be like conversational`;

      const prompts: Record<string, string> = {
        cost_analysis: `${basePrompt}

        For cost analysis, provide:
        1. Budget vs Actual comparison
        2. Cost overrun/savings percentage
        3. Key cost drivers
        4. Recommendations for optimization`,

                summary: `${basePrompt}

        For project summary, provide:
        1. Project overview and objectives
        2. Current status and progress
        3. Key achievements
        4. Outstanding issues
        5. Next steps`,

        zone_analysis: `${basePrompt}

        For zone/region analysis, provide:
        1. Summary of each zone/city analyzed
        2. Total project count and value
        3. Performance comparison across zones
        4. Top performers and underperformers
        5. Key insights and recommendations`,
      };

      return prompts[intentType] || basePrompt;
    };

    // ============ 12. NODE FUNCTIONS ============

    const normalizeQuery = async (state: GraphState) => {
      const normalizedQuestion =
        await this.normalizeSemanticQuestion(state.question);
      return { normalizedQuestion };
    };

    const generateEmbedding = async (state: GraphState) => {
      const embedding = await this.generateGeminiEmbedding(
        state.normalizedQuestion!
      );
      return { embedding };
    };

    const retrieveContext = async (state: GraphState) => {
      try {
        const intent = detectIntent(state.question);
        console.log("🔍 Intent detected:", intent);

        const mustFilters: any[] = [];

        if (intent.projectId) {
          mustFilters.push({
            key: "projectCode",
            match: { value: +intent.projectId },
          });
        }

        if (intent.zones && intent.zones.length > 0) {
          mustFilters.push({
            should: intent.zones.map((zone: string) => ({
              key: "zone",
              match: { value: zone },
            })),
          });
        }

        if (intent.owners?.length) {
          mustFilters.push({
            should: intent.owners.map((owner: string) => ({
              key: "metadata.owner",
              match: { value: owner },
            })),
          });
        }

        if (intent.leadTypes?.length) {
          mustFilters.push({
            should: intent.leadTypes.map((leadType: string) => ({
              key: "metadata.leadType",
              match: { value: leadType },
            })),
          });
}

        if (intent.cities && intent.cities.length > 0) {
          mustFilters.push({
            should: intent.cities.map((city: string) => ({
              key: "city",
              match: { value: city },
            })),
          });
        }

        // NEW: Add stage filter
        if (intent.stage) {
          mustFilters.push({
            key: 'stage',
            match: { value: intent.stage },
          });
          console.log(`📍 Stage filter applied: ${intent.stage}`);
        }

        // NEW: Add status filter
        if (intent.status) {
          mustFilters.push({
            key: 'status',
            match: { value: intent.status },
          });
          console.log(`🔵 Status filter applied: ${intent.status}`);
        }

        // NEW: Add dynamic keyword filters
        if (intent.dynamicKeywords && intent.dynamicKeywords.length > 0) {
          intent.dynamicKeywords.forEach((kw: DynamicKeyword) => {
            mustFilters.push({
              key: kw.field,
              match: { value: kw.value },
            });
            console.log(`🏷️ Dynamic filter applied: ${kw.field}="${kw.value}"`);
          });
        }

        // Add cost range filter
        if (intent.costRange && (intent.costRange.min !== undefined || intent.costRange.max !== undefined)) {
          const rangeFilter: any = {
            key: "boqValue",
            range: {},
          };

          if (intent.costRange.min !== undefined) {
            rangeFilter.range.gte = intent.costRange.min;
          }

          if (intent.costRange.max !== undefined) {
            rangeFilter.range.lte = intent.costRange.max;
          }

          mustFilters.push(rangeFilter);
          console.log("💰 Cost range filter applied:", rangeFilter);
        }

        if (intent.areaRange && (intent.areaRange.min !== undefined || intent.areaRange.max !== undefined)) {
          const areaFilter: any = {
            key: "area",
            range: {},
          };
          if (intent.areaRange.min !== undefined) {
            areaFilter.range.gte = intent.areaRange.min;
          }
          if (intent.areaRange.max !== undefined) {
            areaFilter.range.lte = intent.areaRange.max;
          }
          mustFilters.push(areaFilter);
          console.log("📐 Area range filter applied:", areaFilter);
        }

        const allowedDocTypes = ["details", "flow", "financial"];
        mustFilters.push({
          should: allowedDocTypes.map((docType) => ({
            key: "document_type",
            match: { value: docType },
          })),
        });

        console.log("🔽 Qdrant filters:", JSON.stringify(mustFilters));

        const qdrantResults = await this.qdrantService.search(
          "collection_gemini",
          {
            vector: state.embedding,
            limit: intent.limit,
            filter: mustFilters.length > 0 ? { must: mustFilters } : undefined,
            score_threshold: 0.5,
          }
        );

        console.log(`📦 Retrieved ${qdrantResults.length} documents`);

        if (!qdrantResults || qdrantResults.length === 0) {
          return {
            context: "No projects found matching your criteria.",
            metadata: { documentsFound: 0, intent },
          };
        }

        let formattedContext: string;
        let chunksByType = organizeChunksByType(qdrantResults);

        if (intent.type === "zone_analysis") {
          const byZone = aggregateByZone(qdrantResults);
          const byCity = aggregateByCity(qdrantResults);
          formattedContext = formatZoneAnalysisContext(byZone, byCity, intent);
        } else if (intent.type === "cost_analysis") {
          formattedContext = formatCostAnalysisContext(chunksByType, intent.costRange);
        } else if (intent.type === "summary") {
          formattedContext = formatSummaryContext(chunksByType);
        } else if (/area|sqft|sqm|space|size/i.test(state.question)) {
          formattedContext = formatAreaRangeContext(chunksByType, intent.areaRange);
        } else {
          formattedContext = formatGeneralContext(chunksByType);
        }

        console.log("📝 Formatted context generated", formattedContext);

        return {
          context: formattedContext,
          question: intent.question,
          metadata: {
            documentsFound: qdrantResults.length,
            intent,
            chunksByType: Object.keys(chunksByType),
          },
        };
      } catch (error) {
        console.error("❌ Error retrieving context:", error);
        return {
          context: "Error retrieving context. Please try again.",
          metadata: { error: String(error) },
        };
      }
    };

    const supervisorAgent = async (state: GraphState) => {
      const response = await this.model.invoke([
        new SystemMessage(`Route request into ONE:

        rag
        summary
        cost
        zone_analysis

        Return only the route word, nothing else.`),
        new HumanMessage(state.question),
      ]);

      console.log("supervisorAgent response:", response.content);

      return {
        route: response.content.toString().trim().toLowerCase(),
      };
    };

    const ragAgent = async (state: GraphState) => {
      try {
        console.log("🤖 RAG Agent processing...");

        const response = await this.model.invoke([
          new SystemMessage(`${getSystemPrompt("rag")}

          CONTEXT:
          ${state.context}`),
          new HumanMessage(state.question),
        ]);

        const result =
          typeof response.content === "string"
            ? response.content
            : JSON.stringify(response.content);

        console.log("✅ RAG response generated");

        return {
          result,
          messages: [response],
        };
      } catch (error) {
        console.error("❌ RAG Agent error:", error);
        return {
          result: `Error: ${String(error)}`,
          messages: [],
        };
      }
    };

    const summaryAgent = async (state: GraphState) => {
      try {
        console.log("📋 Summary Agent processing...");

        const response = await this.model.invoke([
          new SystemMessage(`${getSystemPrompt("summary")}

          CONTEXT:
          ${state.context}`),
          new HumanMessage(state.question),
        ]);

        const result = response.content.toString();

        return {
          summary: result,
          result,
          messages: [response],
        };
      } catch (error) {
        console.error("❌ Summary Agent error:", error);
        return {
          summary: `Error: ${String(error)}`,
          result: `Error: ${String(error)}`,
          messages: [],
        };
      }
    };

    const costAgent = async (state: GraphState) => {
      try {
        console.log("💰 Cost Agent processing...");

        const response = await this.model.invoke([
          new SystemMessage(`${getSystemPrompt("cost_analysis")}

          CONTEXT:
          ${state.context}`),
          new HumanMessage(state.question),
        ]);

        const result = response.content.toString();

        return {
          costAnalysis: result,
          result,
          messages: [response],
        };
      } catch (error) {
        console.error("❌ Cost Agent error:", error);
        return {
          costAnalysis: `Error: ${String(error)}`,
          result: `Error: ${String(error)}`,
          messages: [],
        };
      }
    };

    const zoneAnalysisAgent = async (state: GraphState) => {
      try {
        console.log("🗺️ Zone Analysis Agent processing...");

        const response = await this.model.invoke([
          new SystemMessage(`${getSystemPrompt("zone_analysis")}

          CONTEXT:
          ${state.context}`),
          new HumanMessage(state.question),
        ]);

        const result = response.content.toString();

        return {
          zoneAnalysis: result,
          result,
          messages: [response],
        };
      } catch (error) {
        console.error("❌ Zone Analysis Agent error:", error);
        return {
          zoneAnalysis: `Error: ${String(error)}`,
          result: `Error: ${String(error)}`,
          messages: [],
        };
      }
    };

    const saveCache = async (state: GraphState) => {
      if (!state.result) {
        return {};
      }

      try {
        await this.qdrantService.upsert("chat_cache", [
          {
            id: crypto.randomUUID(),
            vector: state.embedding,
            payload: {
              question: state.question,
              normalizedQuestion: state.normalizedQuestion,
              answer: state.result,
              customerId: state.customerId,
              createdAt: new Date().getTime(),
            },
          },
        ]);
        console.log("✅ Response cached");
      } catch (error) {
        console.error("❌ Cache save error:", error);
      }

      return {};
    };

    const routeDecision = (state: GraphState) => {
      switch (state.route) {
        case "summary":
          return "summary";
        case "cost":
          return "cost";
        case "zone_analysis":
          return "zone_analysis";
        default:
          return "rag";
      }
    };

    // ============ 13. BUILD WORKFLOW ============

    const workflow = new StateGraph(GraphAnnotation)
      .addNode("normalize", normalizeQuery)
      .addNode("embeddings", generateEmbedding)
      .addNode("retrieve", retrieveContext)
      .addNode("supervisor", supervisorAgent)
      .addNode("rag", ragAgent)
      .addNode("summaryV1", summaryAgent)
      .addNode("cost", costAgent)
      .addNode("zone_analysis", zoneAnalysisAgent)
      .addNode("saveCache", saveCache)
      .addEdge(START, "normalize")
      .addEdge("normalize", "embeddings")
      .addEdge("embeddings", "retrieve")
      .addEdge("retrieve", "supervisor")
      .addConditionalEdges("supervisor", routeDecision, {
        rag: "rag",
        summary: "summaryV1",
        cost: "cost",
        zone_analysis: "zone_analysis",
      })
      .addEdge("rag", "saveCache")
      .addEdge("summaryV1", "saveCache")
      .addEdge("cost", "saveCache")
      .addEdge("zone_analysis", "saveCache")
      .addEdge("saveCache", END);

    const memory = new MemorySaver();
    return workflow.compile({ checkpointer: memory });
  }
}
