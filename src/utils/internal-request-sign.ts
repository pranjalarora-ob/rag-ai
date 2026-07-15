import * as crypto from 'crypto';

export function signingPayload(method: string, path: string, timestampStr: string, bodyUtf8: string): string {
  return `${String(method).toUpperCase()}\n${path}\n${timestampStr}\n${bodyUtf8 == null ? '' : bodyUtf8}`;
}

export function buildInternalSignature(secret: string, method: string, path: string, timestampStr: string, bodyUtf8: string): string {
  return crypto.createHmac('sha256', secret).update(signingPayload(method, path, timestampStr, bodyUtf8)).digest('hex');
}

export function getInternalServiceHeaders(opts: { secret: string; method: string; path: string; rawBody: string }): Record<string, string> {
  const tsStr = String(Math.floor(Date.now() / 1000));
  const bodyStr = opts.rawBody == null ? '' : Buffer.isBuffer(opts.rawBody) ? (opts.rawBody as Buffer).toString('utf8') : String(opts.rawBody);
  const sig = buildInternalSignature(opts.secret, opts.method, opts.path, tsStr, bodyStr);
  return {
    'X-Internal-Timestamp': tsStr,
    'X-Internal-Signature': sig,
  };
}
