import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import axios from 'axios';

@Injectable()
export class RerankService {
  private readonly COHERE_API_KEY: string;
  private readonly COHERE_MODEL: string;

  constructor(private readonly configService: ConfigService) {
    this.COHERE_API_KEY = this.configService.get('COHERE_API_KEY') || '';
    this.COHERE_MODEL = this.configService.get('COHERE_MODEL') || 'rerank-english-v3.0';
  }

  /**
   * Rerank retrieved document chunks against the user query.
   * If COHERE_API_KEY is not set, it falls back to passing documents through without reranking.
   */
  async rerank(query: string, documents: string[], topN = 10): Promise<string[]> {
    if (!documents || documents.length === 0) {
      return [];
    }

    if (!this.COHERE_API_KEY) {
      console.warn('COHERE_API_KEY not configured. Bypassing rerank step.');
      return documents;
    }

    try {
      const response = await axios.post(
        'https://api.cohere.com/v1/rerank',
        {
          model: this.COHERE_MODEL,
          query,
          documents,
          top_n: topN,
        },
        {
          headers: {
            Authorization: `Bearer ${this.COHERE_API_KEY}`,
            'Content-Type': 'application/json',
          },
        },
      );
      const results = response.data?.results || [];
      // Sort the results and map back to the original documents using their indices
      const rerankedDocs = results
        .sort((a: any, b: any) => b.relevance_score - a.relevance_score)
        .map((r: any) => documents[r.index]);

      return rerankedDocs;
    } catch (error: any) {
      console.error('Cohere Rerank API Error:', error?.response?.data || error?.message);
      // Fallback: return top documents by vector search score as-is
      return documents;
    }
  }
}
