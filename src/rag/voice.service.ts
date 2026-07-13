import { Injectable, Logger } from '@nestjs/common';
import axios from 'axios';
import FormData from 'form-data';

/**
 * Voice-to-text via Ringg Parrot STT (REST). Takes an uploaded audio buffer,
 * forwards it to Ringg's transcription endpoint, and returns the plain transcript.
 * The transcript then feeds the normal RAG chat flow — STT is just a front door.
 *
 * Config (.env): RINGG_API_KEY (required), RINGG_STT_URL (optional override).
 */
@Injectable()
export class VoiceService {
  private readonly logger = new Logger(VoiceService.name);
  private readonly url =
    process.env.RINGG_STT_URL || 'https://prod-api.ringg.ai/stt/v1/transcriptions';

  async transcribe(buffer: Buffer, filename = 'audio.wav', language = 'en'): Promise<string> {
    const apiKey = process.env.RINGG_API_KEY;
    if (!apiKey) {
      throw new Error('RINGG_API_KEY is not configured — add it to .env to enable voice input.');
    }

    const form = new FormData();
    form.append('file', buffer, { filename, contentType: 'audio/wav' });
    form.append('language', language);
    form.append('enable_cap_punc', 'true');

    const res = await axios.post(this.url, form, {
      headers: { 'x-api-key': apiKey, ...form.getHeaders() },
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      timeout: 30000,
    });

    // Ringg's response shape may vary; pick the transcript defensively and log if
    // none of the expected fields are present so we can adjust to the real schema.
    const data: any = res.data;
    const transcript =
      data?.transcript ?? data?.text ?? data?.transcription ?? data?.result ??
      (typeof data === 'string' ? data : '');
    if (!transcript) {
      this.logger.warn(`Unexpected STT response shape: ${JSON.stringify(data).slice(0, 300)}`);
    }
    return String(transcript || '').trim();
  }
}
