const fs = require('fs');
const path = require('path');
const axios = require('axios');

function parseEnv() {
  const envPath = path.join(__dirname, '.env');
  const content = fs.readFileSync(envPath, 'utf8');
  const config = {};
  content.split('\n').forEach(line => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) return;
    const parts = trimmed.split('=');
    const key = parts[0].trim();
    const value = parts.slice(1).join('=').trim();
    config[key] = value;
  });
  return config;
}

const env = parseEnv();
const key = process.env.CLAUDE_API_KEY || process.env.ANTHROPIC_API_KEY || env.CLAUDE_API_KEY || env.ANTHROPIC_API_KEY;
const model = process.env.CLAUDE_MODEL || env.CLAUDE_MODEL || 'claude-3-5-sonnet-20240620';

console.log('Testing Claude API...');
console.log('Model:', model);
console.log('Key length:', key ? key.length : 0);
console.log('Key ending with:', key ? key.slice(-10) : 'none');

if (!key) {
  console.error('No API key found!');
  process.exit(1);
}

axios.post('https://api.anthropic.com/v1/messages', {
  model: model,
  max_tokens: 1024,
  messages: [{ role: 'user', content: 'Hello! Respond with "API Connection Successful!" if you can read this.' }]
}, {
  headers: {
    'x-api-key': key,
    'anthropic-version': '2023-06-01',
    'Content-Type': 'application/json'
  }
}).then(res => {
  console.log('\n✅ SUCCESS!');
  console.log('Claude response:', res.data?.content?.[0]?.text);
}).catch(err => {
  console.error('\n❌ FAILED!');
  if (err.response) {
    console.error('Status Code:', err.response.status);
    console.error('Response Data:', JSON.stringify(err.response.data, null, 2));
  } else {
    console.error('Error:', err.message);
  }
});
